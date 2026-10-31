"""Paper B R5 models: backbone factory, dual-encoder CSG with any backbone, single-encoder DANN with the P9/P10
adversary toolbox, and the P11 moment-matching penalties.

Import after the transformers stub (import train_phase3_sweep first). Nothing here changes an existing module: the
dual-encoder class subclasses CSGLiteLightning and, for backbone 'effb3', keeps CSGLite itself.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.csg_lightning import CSGLiteLightning
from src.models.csg_lite import GradientReversalLayer

HUB = Path.home() / ".cache" / "torch" / "hub"
DINO_REPO = HUB / "facebookresearch_dinov2_main"
BACKBONES = ("effb3", "resnet50", "convnext_tiny", "dinov2_vits14", "dinov2_vitb14", "vit_small_in21k_ft_in1k")
VIT_LIKE = ("dinov2_vits14", "dinov2_vitb14", "vit_small_in21k_ft_in1k")
LN2 = float(math.log(2.0))


def weights_available(name: str) -> tuple[bool, str]:
    ck = HUB / "checkpoints"
    need = {
        "effb3": ck / "efficientnet_b3_rwightman-b3899882.pth",
        "resnet50": ck / "resnet50-0676ba61.pth",
        "convnext_tiny": ck / "convnext_tiny-983f1562.pth",
        "dinov2_vits14": ck / "dinov2_vits14_pretrain.pth",
        "dinov2_vitb14": ck / "dinov2_vitb14_pretrain.pth",
    }.get(name)
    if need is None:
        return True, "timm / Hugging Face cache (checked at load time)"
    return need.exists(), str(need)


def build_backbone(name: str, pretrained: bool = True):
    """Return (module mapping (B,3,224,224) -> (B, d), d). ImageNet normalisation for every backbone."""
    import torchvision.models as tvm

    if name == "effb3":
        net = tvm.efficientnet_b3(weights=tvm.EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None)
        d = net.classifier[1].in_features
        net.classifier = nn.Identity()
        return net, d
    if name == "resnet50":
        net = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        d = net.fc.in_features
        net.fc = nn.Identity()
        return net, d
    if name == "convnext_tiny":
        net = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        d = net.classifier[2].in_features
        net.classifier[2] = nn.Identity()
        return net, d
    if name in ("dinov2_vits14", "dinov2_vitb14"):
        net = torch.hub.load(str(DINO_REPO), name, source="local", pretrained=pretrained)
        return net, int(net.embed_dim)
    if name == "vit_small_in21k_ft_in1k":
        import timm

        net = timm.create_model("vit_small_patch16_224.augreg_in21k_ft_in1k", pretrained=pretrained, num_classes=0)
        return net, int(net.num_features)
    raise ValueError("unknown backbone {}".format(name))


# ----------------------------------------------------------------------------------------------------------------------
# Dual encoder (main model) with any backbone
# ----------------------------------------------------------------------------------------------------------------------


class CSGAnyNet(nn.Module):
    """CSGLite with the backbone swapped; heads, latents, GRL and the grey lesion path are unchanged."""

    def __init__(self, backbone, lesion_classes=8, domain_classes=2, lesion_latent_dim=16, context_latent_dim=64,
                 grl_lambda=1.0, pretrained=True):
        super().__init__()
        self.context_backbone, d_ctx = build_backbone(backbone, pretrained)
        self.lesion_backbone, d_les = build_backbone(backbone, pretrained)
        self.lesion_projector = nn.Linear(d_les, lesion_latent_dim)
        self.lesion_bn = nn.BatchNorm1d(lesion_latent_dim)
        self.lesion_classifier = nn.Linear(lesion_latent_dim, lesion_classes)
        self.context_projector = nn.Linear(d_ctx, context_latent_dim)
        self.context_predictor = nn.Linear(context_latent_dim, domain_classes)
        self.context_orth_projector = nn.Linear(context_latent_dim, lesion_latent_dim, bias=False)
        self.grl = GradientReversalLayer(lambd=grl_lambda)
        h1, h2 = max(lesion_latent_dim, 32), max(lesion_latent_dim // 2, 16)
        self.domain_classifier_adv = nn.Sequential(
            nn.Linear(lesion_latent_dim, h1), nn.BatchNorm1d(h1), nn.ReLU(inplace=True), nn.Dropout(p=0.5),
            nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.ReLU(inplace=True), nn.Dropout(p=0.5),
            nn.Linear(h2, domain_classes),
        )

    @staticmethod
    def _rgb_to_gray3(x):
        gray = 0.2989 * x[:, 0:1] + 0.5870 * x[:, 1:2] + 0.1140 * x[:, 2:3]
        return gray.repeat(1, 3, 1, 1)

    def extract_context_features(self, x):
        return self.context_backbone(x)

    def extract_lesion_features(self, x):
        return self.lesion_backbone(x)

    def forward(self, x_context, x_lesion=None, return_latents=False):
        if x_lesion is None:
            x_lesion = self._rgb_to_gray3(x_context)
        z_lesion_norm = self.lesion_bn(self.lesion_projector(self.extract_lesion_features(x_lesion)))
        y_hat = self.lesion_classifier(z_lesion_norm)
        z_context = self.context_projector(self.extract_context_features(x_context))
        d_ctx = self.context_predictor(z_context)
        d_adv = self.domain_classifier_adv(self.grl(z_lesion_norm))
        if return_latents:
            return y_hat, d_ctx, d_adv, z_lesion_norm, z_context
        return y_hat, d_ctx, d_adv

    def set_grl_lambda(self, lambd):
        self.grl.set_lambd(lambd)

    def project_context_for_orth(self, z_context):
        return self.context_orth_projector(z_context)


class CSGR5Lightning(CSGLiteLightning):
    """CSGLiteLightning plus: any backbone, class-weighted CE, an optional adversary row mask (R6-R1-style dose
    streams), an optional moment-matching penalty on z_lesion_norm (P11), a backbone LR multiplier (ViTs), and an
    epoch-level adversary log. With backbone='effb3', no mask, no penalty, unit weights and backbone_lr_mult=1 the loss
    and optimiser equal the parent's (lambda_supcon is 0 in every Paper B run; its term is dropped, not recomputed)."""

    def __init__(self, backbone="effb3", cls_weight=None, backbone_lr_mult=1.0, penalty="none", penalty_weight=0.0,
                 adv_log_path=None, **kw):
        pretrained = kw.pop("pretrained", True)
        kw["lambda_supcon"] = 0.0
        super().__init__(pretrained=pretrained and backbone == "effb3", backbone_variant="b3", **kw)
        self.backbone_name = backbone
        if backbone != "effb3":
            self.model = CSGAnyNet(backbone, lesion_classes=kw.get("lesion_classes", 8),
                                   domain_classes=kw.get("domain_classes", 2),
                                   lesion_latent_dim=kw.get("lesion_latent_dim", 16),
                                   context_latent_dim=kw.get("context_latent_dim", 64), pretrained=pretrained)
        w = None if cls_weight is None else torch.tensor(cls_weight, dtype=torch.float32)
        self.register_buffer("cls_weight", w if w is not None else torch.ones(kw.get("lesion_classes", 8)))
        self.use_cls_weight = cls_weight is not None
        self.backbone_lr_mult = float(backbone_lr_mult)
        self.penalty = penalty
        self.penalty_weight = float(penalty_weight)
        self.adv_log_path = Path(adv_log_path) if adv_log_path else None
        self._ep = {"loss_adv": [], "acc_adv": [], "loss_cls": [], "penalty": []}

    def training_step(self, batch, batch_idx):
        if len(batch) == 6:
            images_ctx, _les, y, _ys, domain, adv_mask = batch
        else:
            images_ctx, _les, y, _ys, domain = batch
            adv_mask = torch.ones_like(domain, dtype=torch.bool)
        alpha, progress = self._compute_grl_alpha()
        self.model.set_grl_lambda(alpha)
        self._last_alpha = float(alpha)
        y_hat, d_ctx, d_adv, z_l, z_c = self.model(images_ctx, x_lesion=None, return_latents=True)
        loss_cls = F.cross_entropy(y_hat, y, weight=self.cls_weight if self.use_cls_weight else None, ignore_index=-1)
        m = adv_mask.bool()
        loss_ctx = self.ctx_loss_mod(d_ctx[m], domain[m])
        loss_adv = self.adv_loss_mod(d_adv[m], domain[m])
        loss_orth = torch.mean(F.cosine_similarity(z_l, self.model.project_context_for_orth(z_c), dim=1).pow(2))
        pen = z_l.new_tensor(0.0)
        if self.penalty != "none":
            zs, zt = z_l[m & (domain == 0)].float(), z_l[m & (domain == 1)].float()
            if len(zs) > 1 and len(zt) > 1:
                pen = penalty_fn(self.penalty)(zs, zt)
        loss = (loss_cls + self.lambda_ctx * loss_ctx + self.lambda_adv * loss_adv + self.lambda_orth * loss_orth
                + self.penalty_weight * pen)
        with torch.no_grad():
            acc_adv = (d_adv[m].argmax(1) == domain[m]).float().mean()
        self._ep["loss_adv"].append(float(loss_adv.detach()))
        self._ep["acc_adv"].append(float(acc_adv))
        self._ep["loss_cls"].append(float(loss_cls.detach()))
        self._ep["penalty"].append(float(pen.detach()))
        self.log("train/loss", loss, on_step=True, on_epoch=True, batch_size=images_ctx.size(0))
        self.log("train/loss_adv", loss_adv, on_step=False, on_epoch=True, batch_size=images_ctx.size(0))
        self.log("train/acc_adv", acc_adv, on_step=False, on_epoch=True, batch_size=images_ctx.size(0))
        self.log("train/penalty", pen, on_step=False, on_epoch=True, batch_size=images_ctx.size(0))
        self.log("train/grl_progress", progress, on_step=True, on_epoch=False, batch_size=images_ctx.size(0))
        return loss

    def on_train_epoch_end(self):
        row = {"epoch": int(self.current_epoch) + 1, "grl_alpha": float(self._last_alpha), "lnK": LN2}
        for k, v in self._ep.items():
            row[k] = float(np.mean(v)) if v else None
        self._ep = {k: [] for k in self._ep}
        if self.adv_log_path:
            with self.adv_log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
        print("[epoch {:02d}] adv CE {} acc {} cls {} pen {} grl_a {:.3f}".format(
            row["epoch"], row["loss_adv"], row["acc_adv"], row["loss_cls"], row["penalty"], row["grl_alpha"]), flush=True)

    def configure_optimizers(self):
        if self.backbone_lr_mult == 1.0:
            return super().configure_optimizers()
        adv = list(self.model.domain_classifier_adv.parameters())
        cls = list(self.model.lesion_classifier.parameters())
        bb = list(self.model.lesion_backbone.parameters()) + list(self.model.context_backbone.parameters())
        skip = {id(p) for p in adv + cls + bb}
        rest = [p for p in self.model.parameters() if id(p) not in skip]
        lr, wd = self.learning_rate, self.weight_decay
        return torch.optim.AdamW([
            {"params": bb, "lr": lr * self.backbone_lr_mult, "weight_decay": wd},
            {"params": rest, "lr": lr, "weight_decay": wd},
            {"params": adv, "lr": lr * self.adv_lr_multiplier, "weight_decay": wd},
            {"params": cls, "lr": lr * self.lesion_cls_lr_multiplier, "weight_decay": wd},
        ])


def load_dual_net(ckpt_path, build_kwargs, device):
    """Rebuild CSGR5Lightning from known kwargs and load the Lightning state dict (no hparams round trip)."""
    lit = CSGR5Lightning(pretrained=False, **build_kwargs)
    sd = torch.load(str(ckpt_path), map_location="cpu")["state_dict"]
    missing, unexpected = lit.load_state_dict(sd, strict=False)
    bad = [k for k in missing if not k.startswith("cls_weight")]
    if bad or unexpected:
        raise RuntimeError("checkpoint mismatch: missing {} unexpected {}".format(bad[:5], unexpected[:5]))
    return lit.model.to(device).eval()


# ----------------------------------------------------------------------------------------------------------------------
# Single encoder with the adversary toolbox (P9; P10 uses the same head factory)
# ----------------------------------------------------------------------------------------------------------------------


def domain_head(in_dim, n_domains, depth=2, width_mult=1.0):
    """depth 1 = linear (the probe's family); depth 2 = the Paper B derm head (phase15_common.derm_domain_head);
    depth 3 = one more hidden block of the same form."""
    if depth == 1:
        return nn.Linear(in_dim, n_domains)
    h1 = max(int(in_dim * width_mult), 32)
    h2 = max(int(in_dim * width_mult) // 2, 16)
    layers = [nn.Linear(in_dim, h1), nn.BatchNorm1d(h1), nn.ReLU(inplace=True), nn.Dropout(p=0.5)]
    if depth >= 3:
        layers += [nn.Linear(h1, h1), nn.BatchNorm1d(h1), nn.ReLU(inplace=True), nn.Dropout(p=0.5)]
    layers += [nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.ReLU(inplace=True), nn.Dropout(p=0.5), nn.Linear(h2, n_domains)]
    return nn.Sequential(*layers)


class _GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lambd * g, None


class AdversaryBank(nn.Module):
    """K domain heads on GRL(z). head_specs: list of (depth, width_mult)."""

    def __init__(self, in_dim, n_domains, head_specs):
        super().__init__()
        self.specs = list(head_specs)
        self.in_dim, self.n_domains = in_dim, n_domains
        self.heads = nn.ModuleList([domain_head(in_dim, n_domains, d, w) for d, w in self.specs])

    def reinit(self, k):
        d, w = self.specs[k]
        fresh = domain_head(self.in_dim, self.n_domains, d, w).to(next(self.heads[k].parameters()).device)
        self.heads[k].load_state_dict(fresh.state_dict())

    def forward(self, z, coef):
        zr = _GRL.apply(z, float(coef))
        return [h(zr) for h in self.heads]


def grl_alpha_schedule(step, total, kind):
    """'current' = Paper B schedule (gamma 20, p^0.5, floor 0.2): α ≥ 0.2 from step 0.
    'ganin'   = Ganin & Lempitsky 2015: α = 2/(1+exp(-10 p)) - 1, starts at 0.
    'fixed'   = α = 1 from step 0 (no schedule; use with adv_warmup_epochs=0)."""
    p = max(0.0, min(1.0, float(step) / float(max(int(total) - 1, 1))))
    if kind == "fixed":
        return 1.0, p
    if kind == "ganin":
        return float(2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0), p
    a = 2.0 / (1.0 + math.exp(-20.0 * p ** 0.5)) - 1.0
    return float(max(0.2, a)), p


TOOLBOX_DEFAULT = {
    "grl_schedule": "current",
    "adv_warmup_epochs": 0,
    "head_specs": [[2, 1.0]],
    "adv_lr_multiplier": 30.0,
    "adv_inner_steps": 0,
    "reinit_every_steps": 0,
    "max_epochs": 40,
}


class SingleEncNet(nn.Module):
    def __init__(self, backbone, n_classes, n_domains, head_specs, pretrained=True):
        super().__init__()
        self.backbone, self.feat_dim = build_backbone(backbone, pretrained)
        self.feat_bn = nn.BatchNorm1d(self.feat_dim)
        self.classifier = nn.Linear(self.feat_dim, n_classes)
        self.adv = AdversaryBank(self.feat_dim, n_domains, head_specs)
        self.coef = 0.0

    def forward(self, x):
        z = self.feat_bn(self.backbone(x))
        return self.classifier(z), z


class SingleEncToolboxLit(pl.LightningModule):
    """Single-encoder DANN: L = CE_cls + λ · mean_k CE(head_k(GRL_α(z))) (+ w · penalty), the Phase 15.1 / item-5
    loss placement. Toolbox (all off in TOOLBOX_DEFAULT, which reproduces the Phase 15.1 / item-5 recipe):
      grl_schedule        current | ganin
      adv_warmup_epochs   first k epochs: heads train on z.detach(), the encoder gets no reversed gradient
      head_specs          K heads [(depth, width_mult), ...]; loss averaged over heads
      adv_lr_multiplier   head LR = lr × this
      adv_inner_steps     extra head-only AdamW steps per batch on z.detach() (a stronger, closer-to-optimal adversary)
      reinit_every_steps  every N steps one head (round robin) is re-initialised and its Adam state cleared
    Manual optimisation. Main step: one AdamW with the Phase 15.1 groups (encoder+BN lr, heads lr×adv_mult,
    classifier lr×cls_mult), so TOOLBOX_DEFAULT equals the old optimiser. Inner steps: a separate fp32 AdamW over the
    heads only, outside the AMP scaler. The step counter for α is self._step (one per batch)."""

    def __init__(self, backbone="resnet50", n_classes=8, n_domains=2, lr=1e-4, weight_decay=1e-4,
                 cls_lr_multiplier=0.2, lambda_adv=0.0, toolbox=None, cls_weight=None, penalty="none",
                 penalty_weight=0.0, total_steps=1, log_path=None, pretrained=True, backbone_lr_mult=1.0):
        super().__init__()
        self.automatic_optimization = False
        tb = dict(TOOLBOX_DEFAULT)
        tb.update(toolbox or {})
        self.tb = tb
        self.lr, self.wd, self.cls_mult = float(lr), float(weight_decay), float(cls_lr_multiplier)
        self.backbone_lr_mult = float(backbone_lr_mult)
        self.lambda_adv = float(lambda_adv)
        self.net = SingleEncNet(backbone, n_classes, n_domains, [tuple(h) for h in tb["head_specs"]], pretrained)
        self.register_buffer("cls_weight", torch.tensor(cls_weight, dtype=torch.float32) if cls_weight is not None
                             else torch.ones(n_classes))
        self.use_cls_weight = cls_weight is not None
        self.penalty, self.penalty_weight = penalty, float(penalty_weight)
        self.total_steps = int(total_steps)
        self.log_path = Path(log_path) if log_path else None
        self._step = 0
        self._ep = {"loss_adv": [], "acc_adv": [], "loss_cls": [], "penalty": [], "inner_ce": []}
        self._reinits = 0

    def forward(self, x):
        logits, z = self.net(x)
        return logits, z

    def configure_optimizers(self):
        adv_lr = self.lr * float(self.tb["adv_lr_multiplier"])
        bb_lr = self.lr * self.backbone_lr_mult
        return torch.optim.AdamW([
            {"params": list(self.net.backbone.parameters()) + list(self.net.feat_bn.parameters()), "lr": bb_lr,
             "weight_decay": self.wd},
            {"params": self.net.adv.parameters(), "lr": adv_lr, "weight_decay": self.wd},
            {"params": self.net.classifier.parameters(), "lr": self.lr * self.cls_mult, "weight_decay": self.wd},
        ])

    def on_train_start(self):
        self._inner = torch.optim.AdamW(self.net.adv.parameters(), lr=self.lr * float(self.tb["adv_lr_multiplier"]),
                                        weight_decay=self.wd)

    @staticmethod
    def _heads_ce(outs, d):
        return torch.stack([F.cross_entropy(o.float(), d) for o in outs]).mean()

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        x, _les, y, _ys, d_all = batch[:5]
        m = batch[5].bool() if len(batch) == 6 else torch.ones_like(d_all, dtype=torch.bool)
        alpha, _p = grl_alpha_schedule(self._step, self.total_steps, self.tb["grl_schedule"])
        warm = int(self.current_epoch) < int(self.tb["adv_warmup_epochs"])
        logits, z_all = self.net(x)
        loss_cls = F.cross_entropy(logits, y, weight=self.cls_weight if self.use_cls_weight else None, ignore_index=-1)
        z, d = z_all[m], d_all[m]
        z_adv = z.detach() if warm else z
        outs = self.net.adv(z_adv, alpha)
        loss_adv = self._heads_ce(outs, d)
        pen = z.new_tensor(0.0)
        if self.penalty != "none":
            zs, zt = z[d == 0].float(), z[d == 1].float()
            if len(zs) > 1 and len(zt) > 1:
                pen = penalty_fn(self.penalty)(zs, zt)
        lam = self.lambda_adv
        loss = loss_cls + lam * loss_adv + self.penalty_weight * pen
        opt.zero_grad(set_to_none=True)
        self.manual_backward(loss)
        opt.step()
        if lam > 0:
            zd = z.detach().float()
            with torch.autocast(device_type=zd.device.type, enabled=False):
                for _ in range(int(self.tb["adv_inner_steps"])):
                    self._inner.zero_grad(set_to_none=True)
                    l_in = self._heads_ce(self.net.adv(zd, 0.0), d)
                    (lam * l_in).backward()
                    self._inner.step()
                    self._ep["inner_ce"].append(float(l_in.detach()))
            n = int(self.tb["reinit_every_steps"])
            if n > 0 and self._step > 0 and self._step % n == 0:
                k = self._reinits % len(self.net.adv.heads)
                self.net.adv.reinit(k)
                main = opt.optimizer if hasattr(opt, "optimizer") else opt
                for p in self.net.adv.heads[k].parameters():
                    main.state.pop(p, None)
                    self._inner.state.pop(p, None)
                self._reinits += 1
        with torch.no_grad():
            acc_adv = torch.stack([(o.argmax(1) == d).float().mean() for o in outs]).mean()
        self._ep["loss_adv"].append(float(loss_adv.detach()))
        self._ep["acc_adv"].append(float(acc_adv))
        self._ep["loss_cls"].append(float(loss_cls.detach()))
        self._ep["penalty"].append(float(pen.detach()))
        self._step += 1
        self.log("train/loss_cls", loss_cls, on_step=False, on_epoch=True, batch_size=x.size(0))
        self.log("train/loss_adv", loss_adv, on_step=False, on_epoch=True, batch_size=x.size(0))

    def on_train_epoch_end(self):
        row = {"epoch": int(self.current_epoch) + 1, "grl_alpha": grl_alpha_schedule(
            self._step, self.total_steps, self.tb["grl_schedule"])[0], "lnK": LN2, "n_reinit": self._reinits,
            "warmup": int(self.current_epoch) < int(self.tb["adv_warmup_epochs"])}
        for k, v in self._ep.items():
            row[k] = float(np.mean(v)) if v else None
        self._ep = {k: [] for k in self._ep}
        if self.log_path:
            with self.log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")
        print("[epoch {:02d}] adv CE {} acc {} inner {} cls {} pen {}".format(
            row["epoch"], row["loss_adv"], row["acc_adv"], row["inner_ce"], row["loss_cls"], row["penalty"]), flush=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, _ = self.net(x)
        self.log("val/loss", F.cross_entropy(logits, y), on_epoch=True, prog_bar=True, batch_size=x.size(0))
        self.log("val/acc", (logits.argmax(1) == y).float().mean(), on_epoch=True, prog_bar=True, batch_size=x.size(0))


# ----------------------------------------------------------------------------------------------------------------------
# P11 penalties (inputs: source rows, target rows of the scoring space)
# ----------------------------------------------------------------------------------------------------------------------


def _sqdist(a, b):
    return (a.pow(2).sum(1, keepdim=True) + b.pow(2).sum(1)[None, :] - 2.0 * a @ b.t()).clamp_min(0.0)


def mk_mmd(zs, zt, mults=(0.25, 0.5, 1.0, 2.0, 4.0)):
    """Multi-kernel MMD² (Long et al. 2015, DAN): Gaussian kernels with bandwidths = median pairwise squared
    distance of the pooled batch (detached) × mults; unbiased estimator, kernels averaged. Scale-free in the feature
    dimension, unlike the Phase 15 fixed σ ∈ {1..16} kernels, whose values vanish on 1,536-d BN features."""
    xx, yy, xy = _sqdist(zs, zs), _sqdist(zt, zt), _sqdist(zs, zt)
    with torch.no_grad():
        pool = torch.cat([xx[~torch.eye(len(zs), dtype=torch.bool, device=zs.device)], xy.flatten()])
        med = pool.median().clamp_min(1e-6)
    n, m = zs.size(0), zt.size(0)
    out = zs.new_tensor(0.0)
    for c in mults:
        g = 1.0 / (float(c) * med)
        kxx, kyy, kxy = torch.exp(-g * xx), torch.exp(-g * yy), torch.exp(-g * xy)
        out = out + (kxx.sum() - kxx.diagonal().sum()) / (n * (n - 1)) \
            + (kyy.sum() - kyy.diagonal().sum()) / (m * (m - 1)) - 2.0 * kxy.mean()
    return out / len(mults)


def coral_scaled(zs, zt):
    """CORAL with the first moment and dimension-normalised scaling:
    (||μ_s − μ_t||² + ||C_s − C_t||_F²) / d. The Phase 15 term ||C_s − C_t||_F² / (4 d²) has no mean term (the
    15b anomaly: means drift apart while covariances match) and is ~1e-3 on 1,536-d features."""
    d = zs.size(1)

    def cov(z):
        c = z - z.mean(0, keepdim=True)
        return c.t() @ c / max(z.size(0) - 1, 1)

    return ((zs.mean(0) - zt.mean(0)).pow(2).sum() + (cov(zs) - cov(zt)).pow(2).sum()) / d


def mmd_legacy(zs, zt):
    from phase15_common import mmd_rbf_penalty

    return mmd_rbf_penalty(zs, zt)


def penalty_fn(name):
    return {"mkmmd": mk_mmd, "coral": coral_scaled, "mmd_legacy": mmd_legacy}[name]
