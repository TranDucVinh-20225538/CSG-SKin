# Checking numbers in `paper_b.tex`

`scripts/verify_manuscript_numbers.py` compares every decimal it can parse in
`paper/midl/paper_b.tex` with `results/paperB/**/*.json` (and CSV). It does not
edit the manuscript.

```bash
python scripts/verify_manuscript_numbers.py \
  --manifest paper/number_manifest.json \
  --allowlist paper/number_allowlist.json \
  --out results/paperB/manuscript_number_check.md
```

Exit status is 0 unless you pass `--fail-on manifest` (any manifest entry
fails) or `--fail-on problem` (that, plus any free-scan mismatch).

Two reports are written into the markdown file:

- **Free scan.** A printed value is a match when some indexed source rounds,
  half up, to the printed precision; last-digit when the nearest source is at
  most one unit off in that last digit; otherwise a mismatch. With this many
  sources a three-digit value collides often, so a match here does not mean
  the number belongs to the dataset in the sentence. Small bare integers are
  omitted unless you pass `--ints`.
- **Manifest.** `paper/number_manifest.json` pins each load-bearing number to
  one file and key. The checker rounds that key to the printed precision.
  When the entry sets `label`, the number must be attached to that dataset
  (Fitzpatrick17k, MED-NODE, SD-198, SCIN, `pad_heldout`, BCN, HAM, …), not
  merely present somewhere. `context` is a regular expression that has to
  match a window around the occurrence, so a repeated value such as `0.361`
  is checked at the sentence that claims it. `near` lists dataset names that
  must appear in that window. Entries that set `context` are checked at every
  matching site. Older entries without `context` still pass when any site
  carries the declared label.

R11 (label space, monitor features, the frozen-head adversary, BCN→HAM, and
the semi-synthetic dose grid) is the block of ids that start with `r11_`.
They are rebuilt by:

```bash
python scripts/build_r11_manifest_entries.py
```

The script reads `results/paperB/r11/` and will not write an entry whose
source does not round to the printed value. It leaves every earlier manifest
entry in place.

`\pm` is treated as a separator, so `$0.613\pm0.043$` contributes both the
mean and the seed s.d. In the processed text the command is the placeholder
`@@@`; a `context` pattern does not need to mention it.

On this branch the manuscript is the R11 draft (label space through the
semi-synthetic grid). The check written to
`results/paperB/manuscript_number_check.md` reported, for
`paper/midl/paper_b.tex`: match 1942, last-digit 55, mismatch 76, allowlisted
0. The manifest has 1192 entries (1120 from before R11, plus 72 `r11_`
entries) and 0 failures. A free-scan mismatch is not by itself a wrong
sentence: the manifest is the check that a printed value matches one declared
key and the dataset named next to it.

Attribution tests, without walking the results tree:

```bash
python -m unittest tests/test_manuscript_attribution.py
```

## What this run does not pin

Protocol choices that are not result leaves: the melanoma confidence cutoff
0.9 and the 95% true-positive operating point (stated with the X2 clinical
flag shares), and the words “five seeds” / “three seeds”. Shift fractions
printed as 100% or 0% in the BCN→HAM paragraph are the experimental factor,
not a measured output. The main-text SCIN sentence (0.723 to 0.551) repeats
figures already covered by the earlier manifest.
