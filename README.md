# CSG-Skin

Code and manuscript for a study of what **domain-adversarial invariance training**
does to out-of-distribution monitoring in skin lesion classification.

> **Finding.** When adversarial invariance training succeeds, out-of-distribution
> detection collapses from AUROC 0.863 to 0.508 at the smallest non-zero adversarial
> weight tested, then inverts below chance (0.413) — the model ranks images from an
> unseen clinic as *more* routine than its own test data. It becomes more confident on
> those images (0.489 → 0.930) than on in-distribution data, and its predictions
> collapse toward the benign majority class. None of this is visible to the metrics a
> developer monitors: in-distribution accuracy does not fall, in-distribution
> calibration does not change, and the domain leakage probe improves as intended.

Manuscript: [`paper/paper_b.tex`](paper/paper_b.tex) · [`paper/paper_b.pdf`](paper/paper_b.pdf)

---

## Status

**Manuscript in preparation.** The `\author` and repository-link fields are not yet
filled, figures are referenced but not committed, and the Results section is a
numeric skeleton rather than finished prose.

**This checkout does not run end to end.** `data/`, `results/`, `checkpoints/` and
`lightning_logs/` are all gitignored and absent here; the experiments were run on a
cluster. What is committed is the source, the configuration, the manuscript, and the
protocol documents. See [`docs/REPOSITORY_GUIDE.md`](docs/REPOSITORY_GUIDE.md) for a
file-by-file audit of exactly what is and is not present.

## Headline results

Seven-point sweep of the adversarial weight $\lambda_{\text{adv}}$, 3–5 seeds each
(27 runs), everything else held fixed. Leakage is balanced accuracy (chance 0.5);
cross-domain accuracy is balanced over six classes (chance 0.167).

| λ_adv | Leakage ↓ | ID bal. acc ↑ | ID ECE | OOD ECE | OOD AUROC | Cross-domain acc |
|------:|----------:|--------------:|-------:|--------:|----------:|-----------------:|
| 0     | 0.915 | 0.692 | 0.107 | 0.247 | **0.863** | 0.291 |
| 0.25  | 0.553 | 0.696 | 0.101 | 0.684 | **0.508** | 0.291 |
| 0.5   | 0.555 | 0.701 | 0.096 | 0.724 | 0.449 | 0.270 |
| 1     | 0.569 | 0.701 | 0.097 | 0.733 | 0.439 | 0.261 |
| 2     | 0.584 | 0.707 | 0.099 | 0.746 | **0.413** | 0.249 |
| 4     | 0.548 | 0.686 | 0.100 | 0.726 | 0.443 | 0.276 |
| 8     | 0.625 | 0.679 | 0.099 | 0.713 | 0.481 | 0.269 |
| *EffNet-B3 control* | 0.802 | 0.683 | 0.096 | — | 0.726 | 0.298 |

The first two columns move in the direction a practitioner wants. The last three do
not. Note that the non-adversarial control transfers better across domains (0.298)
than any adversarial setting.

**Six candidate explanations were tested and refuted** — image memorisation, class
composition, domain specificity, dimensional collapse, latent compression, and tail
asymmetry — along with a pre-registered synthetic reproduction that failed. The
mechanism remains open. Each test, its prediction and its outcome are recorded in
[`docs/work-orders/`](docs/work-orders/).

## Layout

```
src/
  datasets/    ISIC + PAD-UFES loaders, metadata preprocessing, splits
  models/      baseline (ResNet-50), csg_lite (dual encoder), effb3_single (control)
  losses/      adversarial, orthogonality, CSG composite losses
  utils/       metrics, OOD scoring, seeding, paths, visualisation
scripts/       training, leakage probing, OOD scoring, figures, benchmark driver
cbm_revision/  EffNet-B3 control, OOD benchmarks, statistical tests
configs/       baseline.yaml, csg.yaml
paper/         manuscript source, PDF, section outline
docs/          repository audit + phase work orders
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

CUDA is optional; training falls back to CPU (slowly). Experiments in the manuscript
used a single A100.

## Data

Not included. See [`data/README.md`](data/README.md) for sources, licences and the
expected layout. All three datasets are obtained from their original providers under
their own terms.

## Reproducing

```bash
# 1. build the lesion-cropped metadata table
python scripts/build_lesion_only_metadata.py --preset soft

# 2. sanity-check the data layout and labels
python scripts/verify_data_layout.py
python scripts/verify_isic_labels.py

# 3. train
python scripts/train_baseline.py --seed 42
python scripts/train_csg.py --lambda_adv 2.0 --lambda_orth 1.0 --seed 42

# 4. measure
python scripts/check_leakage.py          # domain probe
python scripts/eval_ood_scores.py        # MSP / Energy / cosine / Mahalanobis / kNN
python scripts/aggregate_results.py      # mean ± s.d. across seeds
```

Run `--help` on any script for its arguments. `scripts/run_full_benchmark.sh` drives
the whole sweep.

Two protocol points that matter for anyone re-running this:

- **Fit every OOD statistic on the ISIC training split only.** Letting PAD or the ISIC
  test split into the covariance or kNN bank invalidates the comparison.
- **PAD appears in training** via the context and adversarial branches, so it cannot
  also serve as the OOD set. Use the patient-level `pad_heldout` partition
  (716 images, 412 patients, disjoint by patient and by lesion).

## Documents

| File | What it is |
|---|---|
| [`paper/paper_b.tex`](paper/paper_b.tex) | manuscript source (compiles with `pdflatex`) |
| [`paper/PAPER_B_OUTLINE.md`](paper/PAPER_B_OUTLINE.md) | section plan, figure list, open items |
| [`docs/REPOSITORY_GUIDE.md`](docs/REPOSITORY_GUIDE.md) | file-by-file audit of this checkout |
| [`docs/work-orders/`](docs/work-orders/) | the experiment series, Phases 0–14, with pre-registered predictions and outcomes |

The work orders are worth reading if you want to know how the result was stress-tested:
each one states a prediction before the run, and several of them record the prediction
failing.

## Citing

See [`CITATION.cff`](CITATION.cff). The manuscript is unpublished; cite the repository
until it appears.

## Licence

Not yet chosen — code is under no explicit licence, which means default copyright
applies. Dataset terms are separate and are listed in [`data/README.md`](data/README.md).
