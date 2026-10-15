"""Paper B R5: training-recipe registry shared by Q1, P9, P10, Q3, P11 (--recipe).

`--recipe current`  the recipe of every published run (Phase 3 / R2 item 4): the values in CURRENT.
`--recipe p8`       the recipe P8 selects (scripts/train_r5_p8.py, stage selB), read from
                    results/paperB/r5/p8/SELECTED_RECIPE.json (flat dict in CURRENT keys, or a dict with key
                    "recipe"). If the file is missing the run stops.
`--recipe <file.json>` a JSON dict in either schema below (e.g. a P8 grid candidate before selection).

Hook contract with P8 (train_r5_p8.py / r6_common.py are owned by another agent and are not edited here). P8 writes
r6_common schema keys; they are translated here:
    latent       -> lesion_latent_dim
    cls_lr_mult  -> cls_lr_multiplier
    max_epochs   -> max_epochs
    balance      -> class_balance   (none -> none, loss -> weighted_ce, sampling -> sampler)
    name, grid_name, grid_id: provenance only (kept in the run's config.json, never change the run)
Any other key stops the run (a silently ignored key would mix recipes). Keys that do not apply to an architecture
(e.g. latent size for a single encoder) are listed in config.json under "recipe_not_applicable".

class_balance = sampler follows P8's three-stream design (r6_common.default_mode): the classifier stream is
class-balanced, the adversary sees a separate natural-mix labelled-domain stream, so resampling does not change the
label shift the adversary is exposed to (r5_common.ThreeStreamISIC; Q1 has its own CXR version).
"""

from __future__ import annotations

import json
from pathlib import Path

PAPERB = Path("/data2/cmdir/home/toandq/CSG-Skin-paperB")
P8_SELECTED = PAPERB / "results" / "paperB" / "r5" / "p8" / "SELECTED_RECIPE.json"  # written by train_r5_p8.py (HOOK)

CURRENT = {
    "lesion_latent_dim": 16,
    "context_latent_dim": 64,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "cls_lr_multiplier": 0.2,
    "adv_lr_multiplier": 30.0,
    "max_epochs": 40,
    "batch_size": 32,
    "lambda_orth": 1.0,
    "lambda_ctx": 1.0,
    "class_balance": "none",
}
CLASS_BALANCE = ("none", "weighted_ce", "sampler")
DUAL_ONLY = ("lesion_latent_dim", "context_latent_dim", "lambda_orth", "lambda_ctx")
P8_KEYS = {"latent": "lesion_latent_dim", "cls_lr_mult": "cls_lr_multiplier", "max_epochs": "max_epochs",
           "balance": "class_balance"}
P8_BALANCE = {"none": "none", "loss": "weighted_ce", "sampling": "sampler"}
PROVENANCE = ("name", "grid_name", "grid_id")


def _translate(over: dict) -> tuple[dict, dict]:
    out, prov = {}, {}
    for k, v in over.items():
        if str(k).startswith("_") or k in PROVENANCE:
            prov[k] = v
        elif k in P8_KEYS:
            out[P8_KEYS[k]] = P8_BALANCE.get(v, v) if k == "balance" else v
        else:
            out[k] = v
    return out, prov


P8_DECISION = PAPERB / "results" / "paperB" / "r5" / "p8" / "RECIPE_DECISION.json"  # written by the P8 report


def concrete(name: str) -> str:
    """'auto' -> 'p8' if the P8 report flagged the λ = 0 model competent, else 'current' (same rule as
    r6_common.resolve_recipe); resolved when the job runs, so a job queued before P8 finishes follows its decision."""
    if name != "auto":
        return name
    if not P8_DECISION.exists():
        raise SystemExit("--recipe auto: P8 recipe decision missing ({}); run the P8 report first".format(P8_DECISION))
    return "p8" if json.loads(P8_DECISION.read_text())["competent"] else "current"


def resolve(name: str) -> tuple[dict, str]:
    """Return (full recipe dict, provenance string)."""
    name = concrete(name)
    if name == "current":
        return dict(CURRENT), "current (Phase 3 / R2 item 4 values)"
    if name == "p8":
        if not P8_SELECTED.exists():
            raise SystemExit("--recipe p8: P8 has not selected a recipe yet ({} missing)".format(P8_SELECTED))
        raw, src = json.loads(P8_SELECTED.read_text()), str(P8_SELECTED)
        raw = raw.get("recipe", raw)
    else:
        p = Path(name)
        if not p.exists():
            raise SystemExit("--recipe must be 'current', 'p8' or a JSON file; got {}".format(name))
        raw, src = json.loads(p.read_text()), str(p)
        raw = raw.get("recipe", raw)
    over, prov = _translate(raw)
    unknown = sorted(set(over) - set(CURRENT))
    if unknown:
        raise SystemExit("recipe {} has keys outside r5_recipe.CURRENT: {}. Add them to the hook before running.".format(
            src, unknown))
    rec = dict(CURRENT)
    rec.update(over)
    if rec["class_balance"] not in CLASS_BALANCE:
        raise SystemExit("class_balance must be one of {}".format(CLASS_BALANCE))
    if prov:
        src = "{} {}".format(src, json.dumps(prov, sort_keys=True))
    return rec, src


def add_args(ap):
    ap.add_argument("--recipe", default="current", help="current | p8 | auto | path/to/recipe.json (see r5_recipe.py)")
    return ap


def tag(name: str) -> str:
    """Directory suffix: '' for current (keeps run names comparable with R2/R4), else '_rec<name>'."""
    name = concrete(name)
    if name == "current":
        return ""
    return "_rec" + ("p8" if name == "p8" else Path(name).stem)


def describe(rec: dict, src: str, not_applicable=()) -> dict:
    return {"recipe": rec, "recipe_source": src, "recipe_not_applicable": sorted(not_applicable)}
