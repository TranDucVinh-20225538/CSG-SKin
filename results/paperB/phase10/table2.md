| Representation | PCA k=1 AUROC (var) | Linear domain-head AUROC | supervision |
|---|---|---|---|
| ImageNet ResNet-50 (frozen) | 0.647 (14.6%) | 0.998 ± 0.0003 | ImageNet |
| ImageNet EffNet-B3 (frozen) | 0.545 (10.7%) | 0.994 ± 0.0013 | ImageNet |
| Trained R50 backbone_raw | 0.648 (32.5%) | 0.997 ± 0.0011 | class-supervised |
| Trained EffB3 backbone_raw | 0.698 (10.9%) | 0.991 ± 0.0015 | class-supervised |
| z_context | > 0.9999 (84.5%) | > 0.9999 | domain-supervised |
