# src/models/baseline.py
# Baseline: ResNet50 backbone + FC head. Optional Lightning training wrapper.

import math
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy, MulticlassConfusionMatrix, MulticlassF1Score
from torchvision.models import ResNet50_Weights, resnet50

from src.datasets.constants import INDEX_TO_LABEL


def get_baseline_resnet50(num_classes=8, pretrained=True):
    """
    ResNet-50 multiclass head (ImageNet weights), backbone features before FC.
    Returns nn.Module with forward(x, return_features=False).
    """
    net = BaselineNet(num_classes=num_classes, pretrained=pretrained)
    return net


class BaselineNet(nn.Module):
    """ResNet50 trunk + linear classifier; returns logits and optional pooled features."""

    def __init__(self, num_classes=8, pretrained=True):
        super().__init__()
        if pretrained:
            # IMAGENET1K_V1 matches Ban_sao_datn get_resnet50 (robust training recipe).
            w = ResNet50_Weights.IMAGENET1K_V1
            backbone = resnet50(weights=w)
        else:
            backbone = resnet50(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x, return_features=False):
        features = self.backbone(x)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


class BaselineResNet50(pl.LightningModule):
    """Lightning wrapper for BaselineNet (same API as plain model for OOD eval)."""

    def __init__(
        self,
        num_classes=8,
        learning_rate=2e-4,
        pretrained=True,
        label_smoothing=0.03,
        weight_decay=1e-4,
        warmup_epochs=5,
        max_epochs=40,
        sanity_isic_eval_items=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["sanity_isic_eval_items"])
        self.num_classes = num_classes
        self.sanity_isic_eval_items = sanity_isic_eval_items or []
        self.learning_rate = learning_rate
        self.label_smoothing = label_smoothing
        self.net = get_baseline_resnet50(num_classes=num_classes, pretrained=pretrained)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_f1_macro = MulticlassF1Score(num_classes=num_classes, average="macro")
        self.val_confmat = MulticlassConfusionMatrix(num_classes=num_classes)
        self._first_batch_sanity_done = False
        self._did_isic_folder_sanity_print = False

    def forward(self, x, return_features=False):
        return self.net(x, return_features=return_features)

    def training_step(self, batch, batch_idx):
        images, targets = batch

        if not self._first_batch_sanity_done:
            self._first_batch_sanity_done = True
            with torch.no_grad():
                nan_m = torch.isnan(images).any().item()
                finite = torch.isfinite(images).all().item()
                flat = images.view(images.size(0), -1)
                all_zero = (flat.abs().sum(dim=1) < 1e-12).all().item()
                print(
                    "[sanity] first train batch: shape={} min={:.4f} max={:.4f} mean={:.4f} | "
                    "nan={} all_finite={} all_zero_batch={}".format(
                        tuple(images.shape),
                        float(images.min()),
                        float(images.max()),
                        float(images.mean()),
                        nan_m,
                        finite,
                        all_zero,
                    )
                )
                if nan_m or not finite:
                    print("[sanity][WARN] Tensor has NaN/Inf — check data loading / transforms.")
                if all_zero:
                    print("[sanity][WARN] Batch is all zeros — images may not be loaded correctly.")

        logits = self(images)
        loss = self.criterion(logits, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        logits = self(images)
        loss = self.criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, targets)
        self.val_f1_macro.update(preds, targets)
        self.val_confmat.update(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1_macro", self.val_f1_macro, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_start(self):
        """First real val epoch: predict on 5 JPEGs under ISIC training folder vs metadata labels."""
        trainer = self.trainer
        if trainer is None or getattr(trainer, "sanity_checking", False):
            return
        if self.current_epoch != 0 or self._did_isic_folder_sanity_print:
            return
        items = self.sanity_isic_eval_items
        if not items:
            return
        from PIL import Image

        from src.datasets.skin_dataset import build_val_transform_robust

        tf = build_val_transform_robust()
        device = self.device
        self.eval()
        print("\n=== Val (epoch 0): 5 samples from ISIC training folder — pred vs true ===")
        with torch.no_grad():
            for path, y_true, y_name in items:
                try:
                    img = Image.open(path).convert("RGB")
                except OSError as e:
                    print("  [skip] {} | {}".format(path, e))
                    continue
                x = tf(img).unsqueeze(0).to(device)
                logits = self(x)
                pred_i = int(torch.argmax(logits, dim=1).item())
                pred_name = INDEX_TO_LABEL.get(pred_i, str(pred_i))
                ok = "OK" if pred_i == y_true else "MISMATCH"
                print(
                    "  {} | true: {} (idx {}) | pred: {} (idx {}) | {}".format(
                        Path(path).name, y_name, y_true, pred_name, pred_i, ok
                    )
                )
        self._did_isic_folder_sanity_print = True

    def on_validation_epoch_end(self):
        cm = self.val_confmat.compute()
        row_sum = cm.sum(dim=1).clamp(min=1e-8)
        per_class_acc = (cm.diag() / row_sum).detach().cpu()
        lines = ["=== Val accuracy per class (true class -> acc) ==="]
        for i in range(self.num_classes):
            name = INDEX_TO_LABEL.get(i, str(i))
            acc_i = float(per_class_acc[i])
            self.log("val/acc_{}".format(name), acc_i, prog_bar=False)
            lines.append("  {} (idx {}): {:.4f}  (n in val ≈ {})".format(name, i, acc_i, int(row_sum[i].item())))
        print("\n".join(lines))
        self.val_confmat.reset()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        warmup = max(1, int(self.hparams.warmup_epochs))
        total = max(1, int(self.hparams.max_epochs))

        def lr_lambda(epoch):
            if epoch < warmup:
                return float(epoch + 1) / float(warmup)
            progress = (epoch - warmup) / float(max(1, total - warmup))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }
