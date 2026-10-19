# Appendix B — LaTeX rows (R1)

Source: `r6/r1/cur/REPORT.md` dose + detector λ=0 baselines; `leakage_matched/regression.json`. Commit `0b20a70`.

## Full Mahalanobis table (shift × λ ∈ {0, 0.5, 2} × sets)

```latex
% Appendix B — R1 Mahalanobis (excl. degraded; primary rule A)
% columns: set & shift & $\lambda$ & mean $\pm$ sd & 95\% CI & $k/n$ (CI$<0.5$) & rule A \\
pad\_heldout & shift 100\% & 0 & 0.825\,$\pm$\,0.022 & [0.810, 0.839] & 0/5 & not met \\
pad\_heldout & shift 100\% & 0.5 & 0.361\,$\pm$\,0.029 & [0.338, 0.383] & 5/5 & met \\
pad\_heldout & shift 50\% & 0.5 & 0.490\,$\pm$\,0.058 & [0.466, 0.512] & 2/3 & not met \\
pad\_heldout & shift 0\% & 0.5 & 0.498\,$\pm$\,0.025 & [0.478, 0.517] & 0/5 & not met \\
pad\_heldout & shift 100\% & 2 & 0.368\,$\pm$\,0.017 & [0.344, 0.391] & 5/5 & met \\
pad\_heldout & shift 50\% & 2 & 0.454\,$\pm$\,0.009 & [0.432, 0.476] & 3/3 & met \\
pad\_heldout & shift 0\% & 2 & 0.458\,$\pm$\,0.003 & [0.438, 0.477] & 4/4 & met \\
fitzpatrick17k & shift 100\% & 0 & 0.712\,$\pm$\,0.013 & [0.698, 0.725] & 0/5 & not met \\
fitzpatrick17k & shift 100\% & 0.5 & 0.385\,$\pm$\,0.023 & [0.373, 0.398] & 5/5 & met \\
fitzpatrick17k & shift 50\% & 0.5 & 0.482\,$\pm$\,0.143 & [0.469, 0.494] & 2/3 & not met \\
fitzpatrick17k & shift 0\% & 0.5 & 0.672\,$\pm$\,0.046 & [0.661, 0.684] & 0/5 & not met \\
fitzpatrick17k & shift 100\% & 2 & 0.340\,$\pm$\,0.065 & [0.328, 0.352] & 5/5 & met \\
fitzpatrick17k & shift 50\% & 2 & 0.441\,$\pm$\,0.076 & [0.428, 0.454] & 2/3 & not met \\
fitzpatrick17k & shift 0\% & 2 & 0.691\,$\pm$\,0.031 & [0.680, 0.702] & 0/4 & not met \\
mednode & shift 100\% & 0 & 0.559\,$\pm$\,0.019 & [0.531, 0.587] & 0/5 & not met \\
mednode & shift 100\% & 0.5 & 0.310\,$\pm$\,0.055 & [0.284, 0.337] & 5/5 & met \\
mednode & shift 50\% & 0.5 & 0.303\,$\pm$\,0.054 & [0.272, 0.336] & 3/3 & met \\
mednode & shift 0\% & 0.5 & 0.525\,$\pm$\,0.050 & [0.497, 0.554] & 1/5 & not met \\
mednode & shift 100\% & 2 & 0.298\,$\pm$\,0.032 & [0.271, 0.326] & 5/5 & met \\
mednode & shift 50\% & 2 & 0.299\,$\pm$\,0.052 & [0.269, 0.331] & 3/3 & met \\
mednode & shift 0\% & 2 & 0.541\,$\pm$\,0.027 & [0.512, 0.572] & 0/4 & not met \\
sd198 & shift 100\% & 0 & 0.708\,$\pm$\,0.019 & [0.695, 0.722] & 0/5 & not met \\
sd198 & shift 100\% & 0.5 & 0.365\,$\pm$\,0.029 & [0.353, 0.377] & 5/5 & met \\
sd198 & shift 50\% & 0.5 & 0.451\,$\pm$\,0.133 & [0.439, 0.461] & 2/3 & not met \\
sd198 & shift 0\% & 0.5 & 0.664\,$\pm$\,0.058 & [0.653, 0.675] & 0/5 & not met \\
sd198 & shift 100\% & 2 & 0.325\,$\pm$\,0.069 & [0.315, 0.336] & 5/5 & met \\
sd198 & shift 50\% & 2 & 0.414\,$\pm$\,0.047 & [0.402, 0.425] & 3/3 & met \\
sd198 & shift 0\% & 2 & 0.694\,$\pm$\,0.031 & [0.683, 0.704] & 0/4 & not met \\
scin & shift 100\% & 0 & 0.723\,$\pm$\,0.018 & [0.710, 0.736] & 0/5 & not met \\
scin & shift 100\% & 0.5 & 0.534\,$\pm$\,0.015 & [0.521, 0.547] & 0/5 & not met \\
scin & shift 50\% & 0.5 & 0.586\,$\pm$\,0.101 & [0.575, 0.599] & 1/3 & not met \\
scin & shift 0\% & 0.5 & 0.718\,$\pm$\,0.036 & [0.708, 0.729] & 0/5 & not met \\
scin & shift 100\% & 2 & 0.519\,$\pm$\,0.073 & [0.507, 0.532] & 1/5 & not met \\
scin & shift 50\% & 2 & 0.566\,$\pm$\,0.051 & [0.553, 0.578] & 0/3 & not met \\
scin & shift 0\% & 2 & 0.732\,$\pm$\,0.035 & [0.722, 0.742] & 0/4 & not met \\
```

## Regression (a) with VIFs

```latex
% Appendix B — regression (a): Maha ~ leakage + shift_level + recipe
% set & n & shift coef & 95\% CI & excl.\ 0 & VIF$_{\mathrm{shift|leak}}$ & VIF$_{\mathrm{shift|recipe}}$ & identifiable \\
fitzpatrick17k & 75 & -0.166 & [-0.217, -0.115] & yes & 1.04 & 1.03 & yes \\
mednode & 75 & -0.107 & [-0.148, -0.065] & yes & 1.04 & 1.03 & yes \\
sd198 & 75 & -0.172 & [-0.221, -0.123] & yes & 1.04 & 1.03 & yes \\
pad\_heldout (not counted) & 75 & 0.041 & [-0.033, 0.110] & no & 1.04 & 1.03 & yes \\
```

