
from __future__ import annotations

import os
import random
import subprocess
from copy import deepcopy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalDiceLoss(nn.Module):
    """Multi-label Focal Dice Loss used in both training stages."""

    def __init__(self, p_pos=2, p_neg=2, clip_pos=0.7, clip_neg=0.5, pos_weight=0.3):
        super().__init__()
        self.p_pos = p_pos
        self.p_neg = p_neg
        self.clip_pos = clip_pos
        self.clip_neg = clip_neg
        self.pos_weight = pos_weight

    def forward(self, input, target):
        prediction = torch.sigmoid(input).contiguous().view(input.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        positive_prediction = prediction
        if self.clip_pos is not None and self.clip_pos >= 0:
            positive_prediction = (positive_prediction + self.clip_pos).clamp(max=1) * positive_prediction
        positive_num = torch.sum(positive_prediction * target, dim=1)
        positive_den = torch.sum(positive_prediction.pow(self.p_pos) + target.pow(self.p_pos), dim=1).clamp_min(1e-8)

        negative_prediction = 1 - prediction
        if self.clip_neg is not None and self.clip_neg >= 0:
            negative_prediction = (negative_prediction + self.clip_neg).clamp(max=1) * negative_prediction
        negative_num = torch.sum(negative_prediction * (1 - target), dim=1)
        negative_den = torch.sum(negative_prediction.pow(self.p_neg) + (1 - target).pow(self.p_neg), dim=1).clamp_min(1e-8)

        positive_loss = 1 - (2 * positive_num) / positive_den
        negative_loss = 1 - (2 * negative_num) / negative_den
        return (positive_loss * self.pos_weight + negative_loss * (1 - self.pos_weight)).mean()


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(device_name: str = "auto") -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        query = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
        free_memory = [int(line.strip()) for line in query.splitlines() if line.strip()]
        if free_memory:
            return torch.device(f"cuda:{int(np.argmax(free_memory))}")
    except Exception:
        pass
    return torch.device("cuda:0")


def positive_negative_low_bce(logits: torch.Tensor, labels: torch.Tensor, neg_weight: float = 0.05) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    weights = torch.where(labels > 0.5, torch.ones_like(labels), torch.full_like(labels, float(neg_weight)))
    return (loss * weights).mean()


def unwrap_state(checkpoint):
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def train_base_epoch(model, loader, optimizer, scheduler, device, cls_loss_fn, lambda_probe: float) -> Dict[str, float]:
    model.train()
    total = {"loss": 0.0, "cls": 0.0, "probe": 0.0}
    n_batches = 0
    for sequence_ids, esm_features, labels in loader:
        sequence_ids = sequence_ids.to(device)
        esm_features = esm_features.to(device)
        labels = labels.to(device).float()
        output = model(sequence_ids, esm_features)
        cls_loss = cls_loss_fn(output["final_logits"], labels)
        probe_loss = F.binary_cross_entropy_with_logits(output["probe_logits"], labels)
        loss = cls_loss + float(lambda_probe) * probe_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total["loss"] += float(loss.item())
        total["cls"] += float(cls_loss.item())
        total["probe"] += float(probe_loss.item())
        n_batches += 1
    scheduler.step()
    return {key: value / max(n_batches, 1) for key, value in total.items()}



def train_residual_step(
    model,
    batch,
    optimizer,
    scheduler,
    device,
    cls_loss_fn,
    lambda_cardinality: float,
    lambda_mfc: float,
    negative_weight: float,
) -> Dict[str, float]:
    model.train()
    # Fixed base predictions must not be randomized by Dropout in stage two.
    model.hpe.eval()
    model.bpn.eval()
    sequence_ids, esm_features, labels = batch
    sequence_ids = sequence_ids.to(device)
    esm_features = esm_features.to(device)
    labels = labels.to(device).float()
    output = model(sequence_ids, esm_features)
    final_logits = output["final_logits"]
    target = (labels.sum(dim=1) >= 2).float()

    cls_loss = cls_loss_fn(final_logits, labels)
    card_loss = F.binary_cross_entropy_with_logits(output["cardinality_logit"], target)
    multifunctional = target > 0.5
    if multifunctional.any():
        mfc_loss = positive_negative_low_bce(final_logits[multifunctional], labels[multifunctional], neg_weight=negative_weight)
    else:
        mfc_loss = final_logits.sum() * 0.0
    loss = cls_loss + float(lambda_cardinality) * card_loss + float(lambda_mfc) * mfc_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    scheduler.step()
    return {
        "loss": float(loss.item()),
        "cls": float(cls_loss.item()),
        "cardinality": float(card_loss.item()),
        "mfc": float(mfc_loss.item()),
    }


def repeated_batches(loader):
    """Yield training batches indefinitely, reshuffling at each new pass."""
    while True:
        yield from loader


@torch.no_grad()
def predict(model, loader, device) -> Dict[str, np.ndarray]:
    model.eval()
    out = {"labels": [], "scores": []}
    for sequence_ids, esm_features, labels in loader:
        sequence_ids = sequence_ids.to(device)
        esm_features = esm_features.to(device)
        output = model(sequence_ids, esm_features)
        final_logits = output["final_logits"]
        out["labels"].append(labels.cpu().numpy())
        out["scores"].append(torch.sigmoid(final_logits).cpu().numpy())
    return {key: np.concatenate(value, axis=0) for key, value in out.items()}


def freeze_base(model) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(("cse.", "rlcm."))


def clone_state(model):
    return deepcopy(model.state_dict())
