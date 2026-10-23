
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import torch

from config import default_config
from data import load_fixed_split
from metrics import metrics_by_group, print_results, result_rows, select_threshold, write_results
from model import build_model
from training import (
    FocalDiceLoss,
    choose_device,
    clone_state,
    freeze_base,
    predict,
    repeated_batches,
    set_seed,
    train_base_epoch,
    train_residual_step,
    unwrap_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CRLC-MFTP in two stages")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda[:index]")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--base-checkpoint", default=None,
        help="Optional base.pt previously produced by stage 1; skips stage-1 retraining.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--base-epochs", type=int, default=None)
    parser.add_argument("--residual-steps", type=int, default=None)
    parser.add_argument("--validate-every", type=int, default=None)
    return parser.parse_args()


def resolve(root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def residual_selection_key(metrics: Dict[str, Dict], limits: Dict[str, float]) -> Tuple:
    """Prefer multifunctional recovery, subject to base-performance safeguards."""
    feasible = (
        metrics["Overall"]["Accuracy"] >= limits["overall_accuracy"]
        and metrics["Overall"]["AT"] >= limits["overall_at"]
        and metrics["1-label"]["AT"] >= limits["single_at"]
        and metrics["Overall"]["AF"] <= limits["overall_af"]
    )
    return (
        int(feasible),
        metrics[">=2-label"]["AT"],
        metrics[">=2-label"]["Accuracy"],
        metrics[">=2-label"]["Coverage"],
        metrics["Overall"]["Accuracy"],
        metrics["Overall"]["AT"],
        -metrics["Overall"]["AF"],
    )


def train_base(config: Dict, data_dir: Path, output_dir: Path, device: torch.device):
    """Optimize and save the stage-1 HPE/BPN checkpoint."""
    set_seed(config["seed"])
    settings = config["base_training"]
    train_loader, val_loader, _, metadata = load_fixed_split(data_dir, config["batch_size"])
    model = build_model(config, metadata["esm_dim"], enable_residual=False).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings["epochs"])
    loss_fn = FocalDiceLoss()
    best_accuracy = -1.0
    best_epoch = 0
    best_state = None

    for epoch in range(1, settings["epochs"] + 1):
        train_base_epoch(
            model, train_loader, optimizer, scheduler, device, loss_fn, settings["lambda_probe"]
        )
        validation = predict(model, val_loader, device)
        grouped = metrics_by_group(
            validation["scores"], validation["labels"], config["fixed_threshold"]
        )
        accuracy = grouped["Overall"]["Accuracy"]
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_epoch = epoch
            best_state = clone_state(model)
        if epoch == 1 or epoch % 10 == 0 or epoch == settings["epochs"]:
            print(
                f"Base epoch {epoch:3d}/{settings['epochs']}  "
                f"val ACC={accuracy:.4f}  best_epoch={best_epoch}"
            )

    if best_state is None:
        raise RuntimeError("Base training did not produce a checkpoint.")
    model.load_state_dict(best_state, strict=True)
    torch.save(
        {"state_dict": best_state, "best_epoch": best_epoch, "config": config},
        output_dir / "base.pt",
    )
    return model, predict(model, val_loader, device), metadata


def load_base_checkpoint(
    path: Path, config: Dict, data_dir: Path, device: torch.device
):
    """Optionally resume at stage 2 from an ordinary stage-1 checkpoint."""
    set_seed(config["seed"])
    _, val_loader, _, metadata = load_fixed_split(data_dir, config["batch_size"])
    model = build_model(config, metadata["esm_dim"], enable_residual=False).to(device)
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(unwrap_state(checkpoint), strict=True)
    print(f"Loaded stage-1 base model from {path}")
    return model, predict(model, val_loader, device), metadata


def initialize_residual_from_base(base_model, residual_model) -> None:
    """Transfer only the selected HPE/BPN weights into stage 2.

    CSE and RCH remain independently initialized. This uses no special RNG
    snapshot or residual-initialization checkpoint.
    """
    if base_model.enable_residual or not residual_model.enable_residual:
        raise ValueError("Expected a base-only model and a residual-enabled model.")
    residual_model.hpe.load_state_dict(base_model.hpe.state_dict(), strict=True)
    residual_model.bpn.load_state_dict(base_model.bpn.state_dict(), strict=True)
    freeze_base(residual_model)


def train_residual(
    config: Dict,
    data_dir: Path,
    base_model,
    base_validation: Dict,
    esm_dim: int,
    device: torch.device,
):
    """Train CSE/RCH with fixed HPE/BPN and validation-based selection."""
    set_seed(config["seed"])
    settings = config["residual_training"]
    train_loader, val_loader, test_loader, _ = load_fixed_split(data_dir, config["batch_size"])
    model = build_model(config, esm_dim, enable_residual=True).to(device)
    initialize_residual_from_base(base_model, model)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings["max_steps"]
    )
    loss_fn = FocalDiceLoss()
    batches = repeated_batches(train_loader)
    base_metrics = metrics_by_group(
        base_validation["scores"], base_validation["labels"], config["fixed_threshold"]
    )
    limits = {
        "overall_accuracy": base_metrics["Overall"]["Accuracy"] - 0.02,
        "overall_at": base_metrics["Overall"]["AT"] - 0.02,
        "single_at": base_metrics["1-label"]["AT"] - 0.03,
        "overall_af": base_metrics["Overall"]["AF"] + 0.005,
    }
    best_key = None
    best_step = 0
    best_state = clone_state(model)

    def validate(step: int) -> None:
        nonlocal best_key, best_step, best_state
        validation = predict(model, val_loader, device)
        grouped = metrics_by_group(
            validation["scores"], validation["labels"], config["fixed_threshold"]
        )
        key = residual_selection_key(grouped, limits)
        if best_key is None or key > best_key:
            best_key = key
            best_step = step
            best_state = clone_state(model)
        print(
            f"Residual step {step:3d}/{settings['max_steps']}  "
            f">=2 AT={grouped['>=2-label']['AT']:.4f}  "
            f">=2 ACC={grouped['>=2-label']['Accuracy']:.4f}  best_step={best_step}"
        )

    validate(0)
    for step in range(1, settings["max_steps"] + 1):
        train_residual_step(
            model=model,
            batch=next(batches),
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            cls_loss_fn=loss_fn,
            lambda_cardinality=settings["lambda_cardinality"],
            lambda_mfc=settings["lambda_mfc"],
            negative_weight=settings["negative_weight"],
        )
        if step % settings["validate_every"] == 0 or step == settings["max_steps"]:
            validate(step)

    model.load_state_dict(best_state, strict=True)
    return model, predict(model, val_loader, device), predict(model, test_loader, device), best_step


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    config = default_config()
    if args.seed is not None:
        config["seed"] = args.seed
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.base_epochs is not None:
        config["base_training"]["epochs"] = args.base_epochs
    if args.residual_steps is not None:
        config["residual_training"]["max_steps"] = args.residual_steps
    if args.validate_every is not None:
        config["residual_training"]["validate_every"] = args.validate_every
    if config["base_training"]["epochs"] < 1 or config["residual_training"]["max_steps"] < 1:
        raise ValueError("Training epochs and steps must both be positive.")
    if config["residual_training"]["validate_every"] < 1 or config["batch_size"] < 1:
        raise ValueError("Validation cadence and batch size must both be positive.")

    data_dir = resolve(root, args.data_dir)
    output_dir = resolve(root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    print(f"Device: {device}")

    if args.base_checkpoint is None:
        base_model, base_validation, metadata = train_base(config, data_dir, output_dir, device)
    else:
        base_model, base_validation, metadata = load_base_checkpoint(
            resolve(root, args.base_checkpoint), config, data_dir, device
        )

    model, validation, test, best_step = train_residual(
        config, data_dir, base_model, base_validation, metadata["esm_dim"], device
    )
    threshold = select_threshold(
        validation["scores"], validation["labels"], config["fixed_threshold"]
    )
    rows = result_rows(test["scores"], test["labels"], threshold)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "best_step": best_step,
            "selected_threshold": threshold,
            "config": config,
        },
        output_dir / "crlc.pt",
    )
    write_results(output_dir / "paper_metrics.csv", rows)
    print_results(rows, threshold)
    print(f"\nSaved: {output_dir}")


if __name__ == "__main__":
    main()
