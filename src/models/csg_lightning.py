# src/models/csg_lightning.py
# PyTorch Lightning wrapper for CSG-lite (disentanglement losses).

import pytorch_lightning as pl
import math
import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassAccuracy

from src.losses.csg_losses import AdvLoss, ContextLoss
from src.models.csg_lite import get_csg_lite


class CSGLiteLightning(pl.LightningModule):
    def __init__(
        self,
        lesion_classes=8,
        domain_classes=2,
        lesion_latent_dim=16,
        context_latent_dim=64,
        learning_rate=2e-4,
        lambda_ctx=1.0,
        lambda_adv=5.0,
        lambda_orth=10.0,
        lambda_supcon=1.0,
        supcon_temperature=0.1,
        adv_lr_multiplier=30.0,
        lesion_cls_lr_multiplier=0.2,
        grl_lambda=1.0,
        grl_gamma=20.0,
        grl_progress_power=0.5,
        grl_alpha_min=0.2,
        print_every_n_steps=0,
        pretrained=True,
        backbone_variant="b3",
        weight_decay=1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lambda_ctx = lambda_ctx
        self.lambda_adv = lambda_adv
        self.lambda_orth = lambda_orth
        self.lambda_supcon = lambda_supcon
        self.supcon_temperature = float(supcon_temperature)
        self.adv_lr_multiplier = float(adv_lr_multiplier)
        self.lesion_cls_lr_multiplier = float(lesion_cls_lr_multiplier)
        self.grl_gamma = float(grl_gamma)
        self.grl_progress_power = float(grl_progress_power)
        self.grl_alpha_min = float(grl_alpha_min)
        self.print_every_n_steps = int(print_every_n_steps)
        self._last_alpha = 0.0

        self.model = get_csg_lite(
            lesion_classes=lesion_classes,
            domain_classes=domain_classes,
            lesion_latent_dim=lesion_latent_dim,
            context_latent_dim=context_latent_dim,
            grl_lambda=grl_lambda,
            pretrained=pretrained,
            backbone_variant=backbone_variant,
        )
        self.ctx_loss_mod = ContextLoss()
        self.adv_loss_mod = AdvLoss()
        self.val_acc = MulticlassAccuracy(num_classes=lesion_classes)
        self._val_debug_printed = False
        self._val_pred_counts = None
        self._val_true_counts = None
        self._printed_lesion_input_mode = False

    def forward(self, x, return_latents=False):
        return self.model(x, return_latents=return_latents)

    def _compute_grl_alpha(self):
        """
        Ganin & Lempitsky schedule:
          alpha = 2 / (1 + exp(-10 * p)) - 1
          p = current_step / total_steps
        """
        total_steps = 0
        try:
            trainer = self.trainer
        except RuntimeError:
            trainer = None
        if trainer is not None:
            total_steps = int(getattr(trainer, "estimated_stepping_batches", 0) or 0)
        denom = max(total_steps - 1, 1)
        progress = float(self.global_step) / float(denom)
        progress = max(0.0, min(1.0, progress))
        progress_shaped = progress ** self.grl_progress_power
        alpha = (2.0 / (1.0 + math.exp(-self.grl_gamma * progress_shaped))) - 1.0
        alpha = max(self.grl_alpha_min, alpha)
        return float(alpha), float(progress)

    def training_step(self, batch, batch_idx):
        if len(batch) == 5:
            images_ctx, images_lesion, y, y_supcon, domain = batch
        elif len(batch) == 4:
            images_ctx, images_lesion, y, domain = batch
            y_supcon = y.clone()
        else:
            images_ctx, y, domain = batch
            images_lesion = None
            y_supcon = y.clone()
        # Use the same lesion path in train/val: derive lesion input inside model from images_ctx.
        # This avoids train/val mismatch (pre-normalization grayscale vs post-normalization conversion).
        if not self._printed_lesion_input_mode:
            self._printed_lesion_input_mode = True
            print("[train-debug] lesion input mode: derive from images_ctx inside model (ignore external images_lesion).")
        images_lesion = None
        alpha, progress = self._compute_grl_alpha()
        self.model.set_grl_lambda(alpha)
        self._last_alpha = float(alpha)

        y_hat, d_ctx, d_adv, z_l, z_c = self.model(images_ctx, x_lesion=images_lesion, return_latents=True)

        # lesion CE only for ISIC rows (PAD has y=-1 from paired collate).
        loss_cls = F.cross_entropy(y_hat, y, ignore_index=-1)
        loss_ctx = self.ctx_loss_mod(d_ctx, domain)
        loss_adv = self.adv_loss_mod(d_adv, domain)
        z_c_proj = self.model.project_context_for_orth(z_c)
        loss_orth = torch.mean(F.cosine_similarity(z_l, z_c_proj, dim=1).pow(2))
        loss_supcon = self._supervised_contrastive_cross_domain_loss(
            z_l,
            y_supcon,
            domain,
            temperature=self.supcon_temperature,
        )

        with torch.no_grad():
            pred_ctx = torch.argmax(d_ctx, dim=1)
            pred_adv = torch.argmax(d_adv, dim=1)
            domain_acc_ctx = (pred_ctx == domain).float().mean()
            domain_acc_adv = (pred_adv == domain).float().mean()
            isic_mask = y >= 0
            if isic_mask.any():
                pred_cls = torch.argmax(y_hat[isic_mask], dim=1)
                cls_acc_isic = (pred_cls == y[isic_mask]).float().mean()
            else:
                cls_acc_isic = y_hat.new_tensor(0.0)

        # L = L_cls + lambda_ctx * L_ctx + lambda_adv * L_adv + lambda_orth * L_orth + lambda_supcon * L_supcon
        loss = (
            loss_cls
            + self.lambda_ctx * loss_ctx
            + self.lambda_adv * loss_adv
            + self.lambda_orth * loss_orth
            + self.lambda_supcon * loss_supcon
        )
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=images_ctx.size(0))
        self.log("train/loss_cls", loss_cls, on_step=False, on_epoch=True)
        self.log("train/loss_ctx", loss_ctx, on_step=False, on_epoch=True)
        self.log("train/loss_adv", loss_adv, on_step=False, on_epoch=True)
        self.log("train/loss_orth", loss_orth, on_step=False, on_epoch=True)
        self.log("train/loss_supcon", loss_supcon, on_step=False, on_epoch=True)
        self.log("train/acc_ctx", domain_acc_ctx, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc_adv", domain_acc_adv, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc_cls", cls_acc_isic, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/grl_a", alpha, on_step=False, on_epoch=True, prog_bar=False, batch_size=images_ctx.size(0))
        self.log("train/grl_progress", progress, on_step=True, on_epoch=False, prog_bar=False, batch_size=images_ctx.size(0))
        return loss

    def _get_metric_value(self, *keys, default=0.0):
        try:
            metrics = self.trainer.callback_metrics
        except RuntimeError:
            return float(default)
        for k in keys:
            if k in metrics:
                v = metrics[k]
                if isinstance(v, torch.Tensor):
                    return float(v.detach().cpu())
                return float(v)
        return float(default)

    def on_train_epoch_end(self):
        acc_adv = self._get_metric_value("train/acc_adv", "train/acc_adv_epoch", default=0.0)
        acc_ctx = self._get_metric_value("train/acc_ctx", "train/acc_ctx_epoch", default=0.0)
        acc_cls = self._get_metric_value("train/acc_cls", "train/acc_cls_epoch", default=0.0)
        loss_orth = self._get_metric_value("train/loss_orth", "train/loss_orth_epoch", default=0.0)
        loss_supcon = self._get_metric_value("train/loss_supcon", "train/loss_supcon_epoch", default=0.0)
        print(
            "[epoch {:02d}] acc_adv={:.4f} acc_ctx={:.4f} acc_cls={:.4f} grl_a={:.4f} orth={:.4f} supcon={:.4f}".format(
                int(self.current_epoch) + 1,
                acc_adv,
                acc_ctx,
                acc_cls,
                float(self._last_alpha),
                loss_orth,
                loss_supcon,
            )
        )

    def _supervised_contrastive_cross_domain_loss(self, z, labels, domains, temperature=0.1):
        """
        Supervised contrastive variant:
        - positive pairs: same class, different domain
        - negatives: all other samples (except itself)
        """
        if z.size(0) < 2:
            return z.new_tensor(0.0)

        z = F.normalize(z, p=2, dim=1)
        sim = torch.matmul(z, z.t()) / max(float(temperature), 1e-6)

        labels = labels.view(-1)
        domains = domains.view(-1)
        same_class = labels.unsqueeze(0).eq(labels.unsqueeze(1))
        diff_domain = ~domains.unsqueeze(0).eq(domains.unsqueeze(1))
        eye = torch.eye(z.size(0), device=z.device, dtype=torch.bool)
        pos_mask = same_class & diff_domain & (~eye)
        logits_mask = ~eye

        losses = []
        for i in range(z.size(0)):
            pos_i = pos_mask[i]
            if pos_i.sum() == 0:
                continue
            logits_i = sim[i]
            denom = torch.logsumexp(logits_i[logits_mask[i]], dim=0)
            pos_log_prob = logits_i[pos_i] - denom
            losses.append(-pos_log_prob.mean())

        if not losses:
            return z.new_tensor(0.0)
        return torch.stack(losses).mean()

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        y_hat, _, _ = self.model(images)
        loss = F.cross_entropy(y_hat, targets)
        preds = torch.argmax(y_hat, dim=1)
        self.val_acc.update(preds, targets)

        with torch.no_grad():
            pred_counts = torch.bincount(preds.detach().cpu(), minlength=self.hparams.lesion_classes)
            true_counts = torch.bincount(targets.detach().cpu(), minlength=self.hparams.lesion_classes)
            self._val_pred_counts += pred_counts
            self._val_true_counts += true_counts

            if not self._val_debug_printed and batch_idx == 0:
                self._val_debug_printed = True
                images_lesion = self.model._rgb_to_gray3(images)
                print(
                    "[val-debug] y_hat shape={} | targets range=[{}, {}]".format(
                        tuple(y_hat.shape),
                        int(targets.min().item()),
                        int(targets.max().item()),
                    )
                )
                print(
                    "[val-debug] images_lesion shape={} mean={:.4f} min={:.4f} max={:.4f}".format(
                        tuple(images_lesion.shape),
                        float(images_lesion.mean().item()),
                        float(images_lesion.min().item()),
                        float(images_lesion.max().item()),
                    )
                )
                print("[val-debug] first-batch pred argmax counts={}".format(pred_counts.tolist()))

                # Ensure lesion classifier receives lesion encoder features as expected.
                feats_les = self.model.extract_lesion_features(images_lesion)
                z_les = self.model.lesion_projector(feats_les)
                z_les = self.model.lesion_bn(z_les)
                y_hat_manual = self.model.lesion_classifier(z_les)
                max_abs_diff = float((y_hat - y_hat_manual).abs().max().item())
                print("[val-debug] lesion classifier path max|diff|={:.6f}".format(max_abs_diff))

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=images.size(0))
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_start(self):
        self._val_debug_printed = False
        self._val_pred_counts = torch.zeros(self.hparams.lesion_classes, dtype=torch.long)
        self._val_true_counts = torch.zeros(self.hparams.lesion_classes, dtype=torch.long)

    def on_validation_epoch_end(self):
        if self._val_pred_counts is not None and self._val_true_counts is not None:
            print("[val-debug] epoch pred argmax counts={}".format(self._val_pred_counts.tolist()))
            print("[val-debug] epoch true label counts={}".format(self._val_true_counts.tolist()))

    def configure_optimizers(self):
        adv_params = list(self.model.domain_classifier_adv.parameters())
        adv_param_ids = {id(p) for p in adv_params}
        cls_params = list(self.model.lesion_classifier.parameters())
        cls_param_ids = {id(p) for p in cls_params}
        main_params = [p for p in self.model.parameters() if id(p) not in adv_param_ids and id(p) not in cls_param_ids]

        if not adv_params:
            raise RuntimeError("Adversarial head has no parameters.")
        if not cls_params:
            raise RuntimeError("Lesion classifier has no parameters.")
        if not main_params:
            raise RuntimeError("Main parameter group is empty.")

        return torch.optim.AdamW(
            [
                {
                    "params": main_params,
                    "lr": self.learning_rate,
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": adv_params,
                    "lr": self.learning_rate * self.adv_lr_multiplier,
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": cls_params,
                    "lr": self.learning_rate * self.lesion_cls_lr_multiplier,
                    "weight_decay": self.weight_decay,
                },
            ]
        )
