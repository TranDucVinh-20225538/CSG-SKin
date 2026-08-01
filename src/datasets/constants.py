# src/datasets/constants.py
"""Shared label ordering and normalization (ImageNet)."""

# Index i <-> LABELS[i] must match metadata string column `label` and model logits dim i.
LABELS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS)}
INDEX_TO_LABEL = {idx: label for label, idx in LABEL_TO_INDEX.items()}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
