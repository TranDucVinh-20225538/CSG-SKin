# src/losses/csg_losses.py
# CSG training losses (cross-entropy heads + covariance independence).

import torch
from torch import nn


# ====================== Supervised heads ======================


class ClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, y_hat, y_true):
        return self.criterion(y_hat, y_true)


class ContextLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, domain_hat_from_z_context, domain_true):
        return self.criterion(domain_hat_from_z_context, domain_true)


class AdvLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, domain_hat_from_z_lesion_adv, domain_true):
        return self.criterion(domain_hat_from_z_lesion_adv, domain_true)


# ====================== Independence (cross-covariance) ======================


class IndependenceLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, z_lesion, z_context):
        batch_size = z_lesion.size(0)
        if batch_size < 2:
            return z_lesion.new_tensor(0.0)

        z_lesion_centered = z_lesion - z_lesion.mean(dim=0, keepdim=True)
        z_context_centered = z_context - z_context.mean(dim=0, keepdim=True)
        cov = (z_lesion_centered.T @ z_context_centered) / (batch_size - 1 + self.eps)
        return cov.pow(2).mean()
