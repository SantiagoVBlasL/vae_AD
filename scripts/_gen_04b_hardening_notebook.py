#!/usr/bin/env python3
"""Generate the hardening audit notebook (04_b)."""
import json, nbformat as nbf, sys
from pathlib import Path

NB_VERSION = 4
nb = nbf.v4.new_notebook()
nb.metadata.update({"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})

def md(src): return nbf.v4.new_markdown_cell(src)
def code(src): return nbf.v4.new_code_cell(src)

# ───────────────────────── Title ────────────────────────────
nb.cells.append(md("""\
# 04-b  Fatigue Connectome Baselines — Hardening Audit

**Purpose:** Three adversarial analyses to challenge the raw baseline findings
before any fatigue-related connectomic signal can be claimed.

| # | Analysis | Target | Verdict |
|---|----------|--------|---------|
| 1 | Max-permutation correction (7 channels) | D + E | Does best channel survive channel selection? |
| 2 | Holm / BH corrections on main-run *p*-values | D + E | FWER vs FDR picture |
| 3 | Incremental value over metadata | D only | Does connectome beat Age+Sex? |
| 4 | Sex-matched subsampling (females) | D only | Does signal persist within-sex? |
"""))

# ───────────────────────── Setup ────────────────────────────
nb.cells.append(code("""\
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 200,
                      "font.size": 10, "axes.titlesize": 12})

OUT = Path("../results/fatigue_connectome_baselines")
FIG = OUT / "Figures"; FIG.mkdir(exist_ok=True)

def _load_json(name):
    p = OUT / "Tables" / name
    return json.loads(p.read_text()) if p.exists() else {}
"""))

# ══════════════════════ §1  Max-perm ════════════════════════
nb.cells.append(md("""\
---
## §1  Max-permutation correction across 7 channels

For each of 1 000 permuted label vectors, we train *all 7 channels*
(raw & residualised) and record the **maximum AUC** across channels.
The corrected *p*-value is the fraction of null maxima ≥ the observed best AUC.

> PCA was pre-computed globally for speed (valid for permutation tests
> because the same procedure is applied to both the observed and every
> permuted label vector).
"""))

nb.cells.append(code("""\
mp_d = _load_json("hardening_D_max_perm.json")
mp_e = _load_json("hardening_E_max_perm.json")

null_d = np.load(OUT / "Tables" / "hardening_D_null_distributions.npz")
null_e = np.load(OUT / "Tables" / "hardening_E_null_distributions.npz")

# Summary table
rows = []
for tgt, mp in [("D", mp_d), ("E", mp_e)]:
    for cond in ["raw", "resid"]:
        key_best = f"best_{cond}_ch"
        key_auc  = f"best_{cond}_auc"
        rows.append(dict(
            Target=tgt, Condition=cond,
            Best_Channel=mp.get(key_best, "—"),
            AUC_globalPCA=mp.get(key_auc, "—"),
            Nominal_p=mp["nominal_p"][str(mp[key_best])][cond] if key_best in mp else "—",
            Corrected_p=mp.get(f"corrected_p_{cond}", "—"),
        ))
df_mp = pd.DataFrame(rows)
print(df_mp.to_string(index=False))
"""))

nb.cells.append(code("""\
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

for ax, tgt, null, mp, label in [
    (axes[0], "D", null_d, mp_d, "Target D (COVID-only, n=53)"),
    (axes[1], "E", null_e, mp_e, "Target E (all groups, n=85)"),
]:
    for cond, color in [("raw", "#2196F3"), ("resid", "#FF5722")]:
        null_arr = null[f"null_max_{cond}"]
        obs = mp[f"best_{cond}_auc"]
        corr_p = mp[f"corrected_p_{cond}"]
        ax.hist(null_arr, bins=40, alpha=0.45, color=color,
                label=f"{cond}  obs={obs:.3f}  p_corr={corr_p:.3f}")
        ax.axvline(obs, color=color, ls="--", lw=2)
    ax.set_xlabel("Max AUC across 7 channels (null distribution)")
    ax.set_title(label)
    ax.legend(fontsize=8)
    ax.axvline(0.5, color="grey", ls=":", lw=0.8)
axes[0].set_ylabel("Frequency")
fig.tight_layout()
fig.savefig(FIG / "fig6_max_perm_null_distributions.png", bbox_inches="tight")
fig.savefig(FIG / "fig6_max_perm_null_distributions.pdf", bbox_inches="tight")
plt.show()
print("Fig 6 saved.")
"""))

# ══════════════════════ §2  Holm / BH ══════════════════════
nb.cells.append(md("""\
---
## §2  Holm–Bonferroni / Benjamini–Hochberg corrections

Applied to the **main-run within-fold permutation *p*-values** (1 000 perms,
proper within-fold PCA).  7 channels tested per target × condition.

- **Holm**: step-down FWER control (α = 0.05)
- **BH**: step-up FDR control (q = 0.05)
"""))

nb.cells.append(code("""\
corr = pd.read_csv(OUT / "Tables" / "hardening_holm_bh_corrections.csv")

def style_row(row):
    styles = [""] * len(row)
    if row["holm_p"] < 0.05:
        styles = ["font-weight: bold; background-color: #c8e6c9"] * len(row)
    elif row["bh_fdr_q"] < 0.05:
        styles = ["font-weight: bold; background-color: #fff9c4"] * len(row)
    return styles

# Show only resid rows for clarity (main interest)
resid = corr[corr["condition"] == "resid"].copy()
resid = resid.sort_values(["target", "nominal_p"])
print("=== Residualised channels — corrections ===")
print(resid.to_string(index=False))
print()
print("Key:  Holm < 0.05 → survives FWER.  BH < 0.05 → survives FDR.")
"""))

nb.cells.append(code("""\
# Paired bar chart: nominal vs Holm vs BH for Target D resid
d_resid = corr[(corr["target"] == "D") & (corr["condition"] == "resid")].sort_values("nominal_p")

fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(d_resid))
w = 0.25
ax.bar(x - w, d_resid["nominal_p"], w, label="Nominal", color="#42A5F5")
ax.bar(x,     d_resid["holm_p"],    w, label="Holm (FWER)", color="#FF7043")
ax.bar(x + w, d_resid["bh_fdr_q"],  w, label="BH (FDR)", color="#66BB6A")
ax.axhline(0.05, color="red", ls="--", lw=1, label="α / q = 0.05")
ax.set_xticks(x)
ax.set_xticklabels(d_resid["channel"], rotation=35, ha="right")
ax.set_ylabel("p-value / q-value")
ax.set_title("Target D — Residualised: Nominal vs Corrected p-values")
ax.legend(fontsize=8)
ax.set_ylim(0, min(1.0, d_resid["holm_p"].max() * 1.3))
fig.tight_layout()
fig.savefig(FIG / "fig7_holm_bh_corrections_D.png", bbox_inches="tight")
fig.savefig(FIG / "fig7_holm_bh_corrections_D.pdf", bbox_inches="tight")
plt.show()
print("Fig 7 saved.")
"""))

# ══════════════════════ §3  Incremental value ══════════════
nb.cells.append(md("""\
---
## §3  Incremental value over metadata  (Target D)

Does the connectome add predictive value beyond Age + Sex?

Three models (same 5-fold CV):
1. **Metadata only** (Age + Sex → StandardScaler → LogReg)
2. **Connectome only** (residualised → PCA(20) → LogReg)
3. **Combined** (resid PCA scores + Age + Sex → LogReg)

Paired bootstrap test on OOF predictions (2 000 resamples).
"""))

nb.cells.append(code("""\
iv_mi  = _load_json("hardening_D_incremental_value.json")
iv_dfc = _load_json("hardening_D_incremental_value_ch3.json")

print("=== Incremental Value — Target D ===")
print()
for lbl, iv in [("MI_KNN (ch2)", iv_mi), ("dFC_AbsDiffMean (ch3)", iv_dfc)]:
    if not iv:
        continue
    print(f"Channel: {lbl}")
    print(f"  Metadata-only AUC:   {iv['auc_metadata']:.4f}")
    print(f"  Connectome-only AUC: {iv['auc_connectome_resid']:.4f}  95%CI {iv['connectome_ci']}")
    print(f"  Combined AUC:        {iv['auc_combined']:.4f}")
    print(f"  ΔAUC (combined-meta): {iv['delta_comb_vs_meta']:.4f}  p={iv['delta_p']:.3f}  95%CI {iv['delta_ci']}")
    print()
"""))

nb.cells.append(code("""\
# Bar chart: metadata vs connectome vs combined
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

for ax, lbl, iv in [
    (axes[0], "MI_KNN (ch2)", iv_mi),
    (axes[1], "dFC_AbsDiffMean (ch3)", iv_dfc),
]:
    if not iv:
        continue
    aucs = [iv["auc_metadata"], iv["auc_connectome_resid"], iv["auc_combined"]]
    labels = ["Metadata\\n(Age+Sex)", f"Connectome\\n(resid {lbl.split()[0]})", "Combined"]
    colors = ["#78909C", "#42A5F5", "#66BB6A"]
    bars = ax.bar(labels, aucs, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0.5, color="grey", ls=":", lw=0.8)
    for b, v in zip(bars, aucs):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=9)
    ax.set_ylim(0.4, 0.85)
    ax.set_ylabel("OOF AUC")
    delta = iv["delta_comb_vs_meta"]
    p = iv["delta_p"]
    ax.set_title(f"{lbl}\\nΔAUC = {delta:+.3f}, p = {p:.3f}")
fig.suptitle("Target D — Incremental Value of Connectome over Metadata", y=1.02)
fig.tight_layout()
fig.savefig(FIG / "fig8_incremental_value.png", bbox_inches="tight")
fig.savefig(FIG / "fig8_incremental_value.pdf", bbox_inches="tight")
plt.show()
print("Fig 8 saved.")
"""))

# ══════════════════════ §4  Sex-matched subsampling ════════
nb.cells.append(md("""\
---
## §4  Sex-matched subsampling  (Target D, females)

Target D has a significant sex imbalance (23F/6M in FATIGA EXTREMA vs 11F/13M
in NO HAY FATIGA, χ² = 5.03, p = 0.025).  Does the signal persist when sex is
held constant?

**Procedure:** Restrict to females (23 pos / 11 neg).
Subsample 11 from 23 positives → balanced 11 vs 11.
LOOCV with PCA(10) + LogReg on each subsample.  200 repetitions.
"""))

nb.cells.append(code("""\
fem = pd.read_csv(OUT / "Tables" / "hardening_D_female_subsampling.csv")
male_path = OUT / "Tables" / "hardening_D_male_subsampling.csv"
male = pd.read_csv(male_path) if male_path.exists() else pd.DataFrame()

print("=== Female sex-matched subsampling (n=22 per sample) ===")
for ch_name in fem["channel_name"].unique():
    aucs = fem.loc[fem["channel_name"] == ch_name, "auc"].dropna()
    print(f"  {ch_name:20s}  median={aucs.median():.3f}  "
          f"IQR=[{aucs.quantile(0.25):.3f}, {aucs.quantile(0.75):.3f}]  "
          f">0.5: {100*(aucs>0.5).mean():.0f}%")

if len(male) > 0:
    print()
    print("=== Male descriptive (n=12 per sample, very noisy) ===")
    for ch_name in male["channel_name"].unique():
        aucs = male.loc[male["channel_name"] == ch_name, "auc"].dropna()
        if len(aucs) > 0:
            print(f"  {ch_name:20s}  median={aucs.median():.3f}  "
                  f"IQR=[{aucs.quantile(0.25):.3f}, {aucs.quantile(0.75):.3f}]  "
                  f">0.5: {100*(aucs>0.5).mean():.0f}%")
"""))

nb.cells.append(code("""\
channels = fem["channel_name"].unique()
fig, ax = plt.subplots(figsize=(7, 5))

positions = np.arange(len(channels))
data = [fem.loc[fem["channel_name"] == ch, "auc"].dropna().values for ch in channels]
vp = ax.violinplot(data, positions=positions, showmedians=True, showextrema=False)
for body in vp["bodies"]:
    body.set_facecolor("#E91E63")
    body.set_alpha(0.4)
vp["cmedians"].set_color("black")

# Overlay boxplot
bp = ax.boxplot(data, positions=positions, widths=0.15, patch_artist=True,
                showfliers=False)
for patch in bp["boxes"]:
    patch.set_facecolor("#F8BBD0")
    patch.set_edgecolor("black")

ax.axhline(0.5, color="grey", ls="--", lw=1, label="Chance (0.5)")
ax.set_xticks(positions)
ax.set_xticklabels(channels, rotation=20, ha="right")
ax.set_ylabel("LOOCV AUC (balanced 11 vs 11)")
ax.set_title("Target D — Female Sex-Matched Subsampling\\n(200 reps, LOOCV)")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(0.0, 1.0)
fig.tight_layout()
fig.savefig(FIG / "fig9_sex_matched_subsampling.png", bbox_inches="tight")
fig.savefig(FIG / "fig9_sex_matched_subsampling.pdf", bbox_inches="tight")
plt.show()
print("Fig 9 saved.")
"""))

# ══════════════════════ §5  Verdict ════════════════════════
nb.cells.append(md("""\
---
## §5  Hardened Verdict

### Target D  (COVID-only, FATIGA EXTREMA vs NO HAY FATIGA)

| Test | Result | Verdict |
|------|--------|---------|
| Holm–Bonferroni (FWER) | MI_KNN p=0.070, dFC_AbsDiffMean p=0.070 | **Borderline** — does NOT reach α=0.05 |
| Benjamini–Hochberg (FDR) | MI_KNN q=0.039, dFC_AbsDiffMean q=0.039 | **Survives** at q=0.05 |
| Max-perm (global PCA) | corrected p = 0.111 (raw), 0.369 (resid) | **Fails** |
| Incremental value over Age+Sex | ΔAUC ≈ +0.09, p ≈ 0.20, CI crosses 0 | **Not significant** |
| Female sex-matched | MI_KNN median=0.504, dFC_AbsDiffMean median=0.550 | **At/near chance** |

**Summary:**  The Target D signal is **fragile**.
- Survives BH-FDR but not Holm-FWER.
- Does not add significant value beyond age + sex alone.
- Collapses when sex imbalance is removed (female-only ≈ chance).
- dFC_AbsDiffMean holds direction slightly (median 0.55, 62% > 0.5) but far from convincing.

### Target E  (all groups, shortcut audit)

| Test | Result | Verdict |
|------|--------|---------|
| Max-perm corrected | raw p=0.001, resid p=0.005 | **Survives** |
| Holm (FWER) | DistanceCorr resid p=0.028 | **Survives** |
| Group confound | χ²=18.19, p<0.0001 | **Massive shortcut threat** |

**Summary:**  Target E shows a strong signal, but it is almost certainly
driven (at least in part) by the COVID vs CONTROL group composition imbalance.
Not interpretable at face value.

### Overall Recommendation

> **CAUTIOUS NO** for claiming robust fatigue-connectome signal at this stage.
>
> The most charitable reading: there is a *suggestion* of fatigue-related
> information in dFC_AbsDiffMean that survives FDR correction, but it is
> entangled with sex and does not survive the most stringent tests (FWER,
> incremental value, female-only).  Any publication should frame this as
> *exploratory / hypothesis-generating*, not confirmatory.
"""))

# ── Save notebook ──
out_path = Path(__file__).resolve().parents[1] / "notebooks" / "04_b_fatigue_hardening.ipynb"
nbf.write(nb, str(out_path))
print(f"Wrote {out_path}  ({len(nb.cells)} cells)")
