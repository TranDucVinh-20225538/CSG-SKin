# src/models/csg_lite.py
# CSG-lite Dual-Encoder: lesion/context backbones are fully separated.

import torch
from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B3_Weights,
    efficientnet_b0,
    efficientnet_b3,
)


class _GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambd=1.0):
        super().__init__()
        self.lambd = float(lambd)

    def set_lambd(self, lambd):
        self.lambd = float(lambd)

    def forward(self, x):
        return _GradientReversalFn.apply(x, self.lambd)


def _build_efficientnet_backbone(variant="b3", pretrained=True):
    v = str(variant).lower()
    if v == "b3":
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        net = efficientnet_b3(weights=weights)
    elif v == "b0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        net = efficientnet_b0(weights=weights)
    else:
        raise ValueError("Unsupported backbone variant: {} (use 'b3' or 'b0').".format(variant))

    feat_dim = net.classifier[1].in_features
    net.classifier = nn.Identity()
    return net, feat_dim


def get_csg_lite(
    lesion_classes=8,
    domain_classes=2,
    lesion_latent_dim=16,
    context_latent_dim=64,
    grl_lambda=1.0,
    pretrained=True,
    backbone_variant="b3",
):
    return CSGLite(
        lesion_classes=lesion_classes,
        domain_classes=domain_classes,
        lesion_latent_dim=lesion_latent_dim,
        context_latent_dim=context_latent_dim,
        grl_lambda=grl_lambda,
        pretrained=pretrained,
        backbone_variant=backbone_variant,
    )


class CSGLite(nn.Module):
    def __init__(
        self,
        lesion_classes=8,
        domain_classes=2,
        lesion_latent_dim=16,
        context_latent_dim=64,
        grl_lambda=1.0,
        pretrained=True,
        backbone_variant="b3",
    ):
        super().__init__()
        # Two independent encoders (no shared backbone).
        self.context_backbone, feat_dim_ctx = _build_efficientnet_backbone(backbone_variant, pretrained=pretrained)
        self.lesion_backbone, feat_dim_les = _build_efficientnet_backbone(backbone_variant, pretrained=pretrained)

        self.lesion_projector = nn.Linear(feat_dim_les, lesion_latent_dim)
        self.lesion_bn = nn.BatchNorm1d(lesion_latent_dim)
        self.lesion_classifier = nn.Linear(lesion_latent_dim, lesion_classes)

        self.context_projector = nn.Linear(feat_dim_ctx, context_latent_dim)
        self.context_predictor = nn.Linear(context_latent_dim, domain_classes)
        # Orth projector only for loss computation.
        self.context_orth_projector = nn.Linear(context_latent_dim, lesion_latent_dim, bias=False)

        self.grl = GradientReversalLayer(lambd=grl_lambda)
        adv_hidden_dim1 = max(lesion_latent_dim, 32)
        adv_hidden_dim2 = max(lesion_latent_dim // 2, 16)
        self.domain_classifier_adv = nn.Sequential(
            nn.Linear(lesion_latent_dim, adv_hidden_dim1),
            nn.BatchNorm1d(adv_hidden_dim1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(adv_hidden_dim1, adv_hidden_dim2),
            nn.BatchNorm1d(adv_hidden_dim2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(adv_hidden_dim2, domain_classes),
        )

    @staticmethod
    def _rgb_to_gray3(x):
        gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
        return gray.repeat(1, 3, 1, 1)

    def extract_context_features(self, x_context):
        return self.context_backbone(x_context)

    def extract_lesion_features(self, x_lesion):
        return self.lesion_backbone(x_lesion)

    def forward(self, x_context, x_lesion=None, return_latents=False):
        if x_lesion is None:
            x_lesion = self._rgb_to_gray3(x_context)

        features_ctx = self.extract_context_features(x_context)
        features_les = self.extract_lesion_features(x_lesion)

        z_lesion = self.lesion_projector(features_les)
        z_lesion_norm = self.lesion_bn(z_lesion)
        y_hat = self.lesion_classifier(z_lesion_norm)

        z_context = self.context_projector(features_ctx)
        d_ctx = self.context_predictor(z_context)

        z_lesion_adv = self.grl(z_lesion_norm)
        d_adv = self.domain_classifier_adv(z_lesion_adv)

        if return_latents:
            return y_hat, d_ctx, d_adv, z_lesion_norm, z_context
        return y_hat, d_ctx, d_adv

    def encode_z_lesion(self, x):
        x_gray = self._rgb_to_gray3(x)
        features = self.extract_lesion_features(x_gray)
        z_lesion = self.lesion_projector(features)
        return self.lesion_bn(z_lesion)

    def set_grl_lambda(self, lambd):
        self.grl.set_lambd(lambd)

    def project_context_for_orth(self, z_context):
        return self.context_orth_projector(z_context)
