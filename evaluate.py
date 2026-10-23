
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from config import default_config
from data import load_fixed_split
from metrics import print_results, result_rows, select_threshold, write_results
from model import build_model
from training import choose_device, predict, unwrap_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained CRLC-MFTP model")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--checkpoint", default="outputs/crlc.pt")
    parser.add_argument("--output", default="outputs/evaluation_metrics.csv")
    return parser.parse_args()


def resolve(root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    device = choose_device(args.device)
    checkpoint = torch.load(resolve(root, args.checkpoint), map_location=device, weights_only=True)
    config = checkpoint.get("config", default_config()) if isinstance(checkpoint, dict) else default_config()

    _, val_loader, test_loader, metadata = load_fixed_split(
        resolve(root, args.data_dir), batch_size=config["batch_size"]
    )
    model = build_model(config, metadata["esm_dim"], enable_residual=True).to(device)
    model.load_state_dict(unwrap_state(checkpoint), strict=True)

    test = predict(model, test_loader, device)
    # Reuse the threshold selected on validation when the model was saved.
    # For older checkpoints without a stored threshold, select it from val.
    threshold = checkpoint.get("selected_threshold") if isinstance(checkpoint, dict) else None
    if threshold is None:
        validation = predict(model, val_loader, device)
        threshold = select_threshold(
            validation["scores"], validation["labels"], config["fixed_threshold"]
        )
    rows = result_rows(test["scores"], test["labels"], float(threshold))
    output = resolve(root, args.output)
    write_results(output, rows)
    print_results(rows, float(threshold))
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
