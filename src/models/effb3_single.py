"""EfficientNet-B3 single-encoder control baseline (close to CSG lesion branch)."""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.classification import MulticlassAccuracy
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3


class EffB3SingleNet(nn.Module):
    """image -> gray3 -> EfficientNet-B3 -> projector+BN -> 8-class classifier."""

    def __init__(self, num_classes=8, latent_dim=16, pretrained=True):
        super().__init__()
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b3(weights=weights)
        feat_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.projector = nn.Linear(feat_dim, latent_dim)
        self.bn = nn.BatchNorm1d(latent_dim)
        self.classifier = nn.Linear(latent_dim, num_classes)

    @staticmethod
    def _rgb_to_gray3(x):
        gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
        return gray.repeat(1, 3, 1, 1)

    def extract_embedding(self, x):
        x_gray = self._rgb_to_gray3(x)
        feats = self.backbone(x_gray)
        z = self.projector(feats)
        return self.bn(z)

    def forward(self, x, return_features=False):
        z = self.extract_embedding(x)
        logits = self.classifier(z)
        if return_features:
            return logits, z
        return logits


class EffB3SingleLightning(pl.LightningModule):
    """Lightning wrapper with CSG-like optimization settings (AdamW, no scheduler)."""

    def __init__(
        self,
        num_classes=8,
        latent_dim=16,
        learning_rate=1e-4,
        weight_decay=1e-4,
        pretrained=True,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.net = EffB3SingleNet(
            num_classes=num_classes,
            latent_dim=latent_dim,
            pretrained=pretrained,
        )
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, x, return_features=False):
        return self.net(x, return_features=return_features)

    def training_step(self, batch, _batch_idx):
        images, targets = batch
        logits = self(images)
        loss = F.cross_entropy(logits, targets)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        return loss

    def validation_step(self, batch, _batch_idx):
        images, targets = batch
        logits = self(images)
        loss = F.cross_entropy(logits, targets)
        preds = torch.argmax(logits, dim=1)
        self.val_acc.update(preds, targets)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

