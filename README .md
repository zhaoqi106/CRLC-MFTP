# CRLC-MFTP: Cardinality-Guided Residual Label Completion for Multi-Functional Therapeutic Peptide Prediction

## Overview

This repository provides the implementation of CRLC-MFTP for multi-label therapeutic peptide prediction on the MFTP dataset, which contains 21 functional labels.

CRLC-MFTP combines amino acid embeddings and ESM-2 representations with a multi-scale TextCNN. A cardinality score estimator guides residual label completion to improve the prediction of multifunctional peptides. The model is trained in two stages: base prediction and residual completion.

---

## Repository Structure

```text
CRLC-MFTP/
├── data/
│   ├── train.txt
│   ├── val.txt
│   ├── test.txt
│   └── label_order.txt
├── config.py
├── data.py
├── model.py
├── training.py
├── train.py
├── evaluate.py
├── metrics.py
└── README.md
```

---

## Task Description

MFTP prediction is a multi-label classification task. Each peptide sequence may have one or more of the 21 therapeutic functions.

The dataset contains 6,691 training samples, 1,182 validation samples, and 1,969 test samples.

---

## Data Files

The sequence and label files are:

```text
train.txt
val.txt
test.txt
```

Each sample consists of a 21-dimensional binary label vector and an amino acid sequence on two consecutive lines:

```text
>000000001000000000000
GLSDGEWQ
```

`label_order.txt` specifies the order of the functional labels.

The data loader also reads precomputed ESM-2 features for each split:

```text
train_esm.npy
val_esm.npy
test_esm.npy
```

---

## Environment Setup

### Recommended Software

```text
Python 3.10 or later
PyTorch
NumPy
```

### Create a Conda Environment

```bash
conda create -n crlc-mftp python=3.10 -y
conda activate crlc-mftp
```

### Install Dependencies

```bash
pip install torch numpy
```

---

## Running the Code

Run the commands from the `CRLC-MFTP` directory.

### Training

```bash
python train.py
```

### Evaluation

```bash
python evaluate.py
```

---

## Output Files

Training saves the following files in `outputs/`:

```text
base.pt
crlc.pt
paper_metrics.csv
```

- `base.pt`: base prediction model checkpoint.
- `crlc.pt`: final CRLC-MFTP model checkpoint.
- `paper_metrics.csv`: test evaluation results.

Evaluation saves `outputs/evaluation_metrics.csv`.

---

## Evaluation Results

The code reports five metrics:

```text
Aiming
Coverage
ACC
AT
AF
```

Results are reported for the overall test set and the 1-label, 2-label, 3-label, >=4-label, and >=2-label groups.
