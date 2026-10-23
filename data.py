
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

AMINO_ACIDS = "XACDEFGHIKLMNPQRSTVWY"
AA_TO_ID = {aa: i for i, aa in enumerate(AMINO_ACIDS)}


def load_mftp_txt(file_path: str | Path) -> Tuple[List[str], np.ndarray]:
    path = Path(file_path)
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) % 2 != 0:
        raise ValueError(f"Dataset file should contain label/sequence pairs: {path}")
    sequences, labels = [], []
    for i in range(0, len(lines), 2):
        if not lines[i].startswith(">"):
            raise ValueError(f"Expected label line at line {i + 1}: {path}")
        label = np.asarray([int(c) for c in lines[i][1:]], dtype=np.float32)
        if label.shape[0] != 21:
            raise ValueError(f"Expected 21 labels, got {label.shape[0]}: {path}")
        sequences.append(lines[i + 1].upper())
        labels.append(label)
    return sequences, np.asarray(labels, dtype=np.float32)


def load_esm_npy(path: str | Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    arr = np.load(path, allow_pickle=True)
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        arr = np.stack(arr.tolist(), axis=0)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected ESM feature shape [N, L, D], got {arr.shape}: {path}")
    return arr


def seq_to_ids(sequences: List[str], max_len: int) -> torch.LongTensor:
    encoded = []
    for seq in sequences:
        ids = [AA_TO_ID.get(aa, 0) for aa in seq[:max_len]]
        if len(ids) < max_len:
            ids.extend([0] * (max_len - len(ids)))
        encoded.append(ids)
    return torch.LongTensor(np.asarray(encoded, dtype=np.int64))


def make_tensor_dataset(
    sequences: List[str],
    labels: np.ndarray,
    esm_features: np.ndarray,
    max_len: int,
) -> TensorDataset:
    if len(sequences) != labels.shape[0] or len(sequences) != esm_features.shape[0]:
        raise ValueError("Sequence, label, and ESM feature counts are not aligned.")
    return TensorDataset(
        seq_to_ids(sequences, max_len),
        torch.tensor(esm_features, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
    )


def load_fixed_split(data_dir: str | Path, batch_size: int = 256) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    data_dir = Path(data_dir)
    train_sequences, train_labels = load_mftp_txt(data_dir / "train.txt")
    val_sequences, val_labels = load_mftp_txt(data_dir / "val.txt")
    test_sequences, test_labels = load_mftp_txt(data_dir / "test.txt")

    train_esm = load_esm_npy(data_dir / "train_esm.npy")
    val_esm = load_esm_npy(data_dir / "val_esm.npy")
    test_esm = load_esm_npy(data_dir / "test_esm.npy")
    max_len = int(train_esm.shape[1])
    esm_dim = int(train_esm.shape[2])

    train_ds = make_tensor_dataset(train_sequences, train_labels, train_esm, max_len)
    val_ds = make_tensor_dataset(val_sequences, val_labels, val_esm, max_len)
    test_ds = make_tensor_dataset(test_sequences, test_labels, test_esm, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    meta = {
        "num_train": len(train_ds),
        "num_val": len(val_ds),
        "num_test": len(test_ds),
        "max_len": max_len,
        "esm_dim": esm_dim,
        "data_dir": str(data_dir),
    }
    expected = {"num_train": 6691, "num_val": 1182, "num_test": 1969}
    actual = {key: meta[key] for key in expected}
    if actual != expected:
        raise RuntimeError(f"Fixed split check failed: expected={expected}, actual={actual}")
    return train_loader, val_loader, test_loader, meta
