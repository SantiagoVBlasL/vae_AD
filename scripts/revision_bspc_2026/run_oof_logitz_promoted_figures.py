"""
Figure-ready package — promoted OOF-logitz candidate (recover035_latent384_beta3p75).
Read-only. No training. No model selection.

Outputs: results/revision_bspc_2026/oof_logitz_promoted_candidate_figures/
  fig1_roc_pooled.{pdf,png}
  fig2_pr_pooled.{pdf,png}
  fig3_fold_score_ranges.{pdf,png}
  fig4_score_distributions.{pdf,png}
  fig5_foldwise_auc_bar.{pdf,png}
  fig6_confusion_matrix.{pdf,png}
  fig7_subgroup_bars.{pdf,png}
  fig8_reliability_calibration.{pdf,png}
  command_log.json
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve, auc as sklearn_auc,
    precision_recall_curve, average_precision_score,
)

# ─── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
RES  = ROOT / "results" / "revision_bspc_2026"

ALL_COMP  = RES / "oof_logitz_all_candidates_comparison"
STAT_VAL  = RES / "oof_logitz_statistical_validation"
STAGE_B   = RES / "recover035_latent384_beta3p75_stageB_oof_score_calibration"
OUT_DIR   = RES / "oof_logitz_promoted_candidate_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── colour / style constants ─────────────────────────────────────────────────
C_LOCKED  = "#6B6B6B"   # dark grey  — locked_v5p1b raw
C_RAW     = "#4682B4"   # steel-blue — beta3p75 raw
C_LOGITZ  = "#C0392B"   # crimson    — beta3p75 OOF-logitz (promoted)
C_CN      = "#3B9AB2"   # teal-blue  — CN subjects
C_AD      = "#E86B2A"   # warm-orange — AD subjects

LABEL_LOCKED = "locked v5.1b  (raw)"
LABEL_RAW    = "beta3p75  (raw)"
LABEL_LOGITZ = "beta3p75  (OOF-logit-z)  ←promoted"

DPI  = 300
PAD  = 0.15   # tight_layout padding

def _style_ax(ax, despine=True):
    if despine:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)

def _save(fig, stem: str):
    fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{stem}.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {stem}.pdf / .png")

# ─── load data ────────────────────────────────────────────────────────────────
pred_all  = pd.read_csv(ALL_COMP / "calib_predictions_all.csv")
fw_metrics = pd.read_csv(ALL_COMP / "foldwise_metrics.csv")
sr_fold    = pd.read_csv(ALL_COMP / "score_range_by_fold.csv")
philips_fp = pd.read_csv(ALL_COMP / "philips_cn_fp.csv")
ge_fn      = pd.read_csv(ALL_COMP / "ge_ad_fn.csv")
brier_ece  = pd.read_csv(ALL_COMP / "brier_ece_comparison.csv")
stage_sr   = pd.read_csv(STAGE_B / "calib_score_range_by_fold.csv")

boot_ci = json.load(open(STAT_VAL / "bootstrap_ci.json"))
delong  = json.load(open(STAT_VAL / "delong_test.json"))

# convenience: pull the 3 prediction sets
def _preds(candidate, calib):
    return pred_all.loc[
        (pred_all["candidate"] == candidate) &
        (pred_all["calib_method"] == calib)
    ].copy()

pL  = _preds("locked_v5p1b",                      "raw")
pRR = _preds("recover035_latent384_beta3p75",      "raw")
pOF = _preds("recover035_latent384_beta3p75",      "oof_logitz")

# ─── Figure 1 — ROC curve (pooled) ───────────────────────────────────────────
print("[1] ROC pooled …")

def _roc(df):
    fpr, tpr, _ = roc_curve(df["y_true"], df["y_score"])
    return fpr, tpr, sklearn_auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(3.8, 3.6))

for df, col, label, ci_key, ls in [
    (pL,  C_LOCKED,  LABEL_LOCKED,  "locked_v5p1b_raw",                    "-"),
    (pRR, C_RAW,     LABEL_RAW,     "recover035_latent384_beta3p75_raw",    "--"),
    (pOF, C_LOGITZ,  LABEL_LOGITZ,  "recover035_latent384_beta3p75_oof_logitz", "-"),
]:
    fpr, tpr, auc_v = _roc(df)
    ci = boot_ci[ci_key]
    lbl = f"{label}\nAUC = {auc_v:.3f}  [{ci['auc_ci95_lower']:.3f}, {ci['auc_ci95_upper']:.3f}]"
    ax.plot(fpr, tpr, color=col, lw=1.8, ls=ls, label=lbl)

# diagonal
ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)

# DeLong annotation — upper left so it doesn't overlap the lower-right legend
dl_v = delong["beta3p75_logitz_vs_locked_raw"]
p_str = f"p = {dl_v['p_two_sided']:.3f} (n.s.)" if dl_v["p_two_sided"] >= 0.05 \
        else f"p = {dl_v['p_two_sided']:.3f} *"
ax.text(0.03, 0.97,
        f"DeLong (promoted vs locked):\nΔAUC = {dl_v['delta_auc']:+.3f},  {p_str}",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=7.5, color="#444444",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))

ax.set_xlabel("1 − Specificity (FPR)")
ax.set_ylabel("Sensitivity (TPR)")
ax.set_title("ROC Curve — pooled (outer 5-fold CV)", fontsize=9, pad=6)
ax.legend(fontsize=7, loc="lower right", framealpha=0.9, frameon=True)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
_style_ax(ax)
fig.tight_layout(pad=PAD)
_save(fig, "fig1_roc_pooled")

# ─── Figure 2 — PR curve (pooled) ────────────────────────────────────────────
print("[2] PR pooled …")

def _pr(df):
    prec, rec, _ = precision_recall_curve(df["y_true"], df["y_score"])
    ap = average_precision_score(df["y_true"], df["y_score"])
    return rec, prec, ap

fig, ax = plt.subplots(figsize=(3.8, 3.6))

for df, col, label, ci_key, ls in [
    (pL,  C_LOCKED,  LABEL_LOCKED,  "locked_v5p1b_raw",                    "-"),
    (pRR, C_RAW,     LABEL_RAW,     "recover035_latent384_beta3p75_raw",    "--"),
    (pOF, C_LOGITZ,  LABEL_LOGITZ,  "recover035_latent384_beta3p75_oof_logitz", "-"),
]:
    rec, prec, ap = _pr(df)
    ci = boot_ci[ci_key]
    lbl = (f"{label}\nPR-AUC = {ap:.3f}  "
           f"[{ci['prauc_ci95_lower']:.3f}, {ci['prauc_ci95_upper']:.3f}]")
    ax.plot(rec, prec, color=col, lw=1.8, ls=ls, label=lbl)

# prevalence baseline
prev = pOF["y_true"].mean()
ax.axhline(prev, color="k", ls=":", lw=0.8, alpha=0.4)
ax.text(1.01, prev, f"prev\n{prev:.2f}", va="center", ha="left", fontsize=7, color="#888")

ax.set_xlabel("Recall (Sensitivity)")
ax.set_ylabel("Precision")
ax.set_title("Precision–Recall Curve — pooled (outer 5-fold CV)", fontsize=9, pad=6)
ax.legend(fontsize=7, loc="upper right", framealpha=0.9, frameon=True)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.08)
_style_ax(ax)
fig.tight_layout(pad=PAD)
_save(fig, "fig2_pr_pooled")

# ─── Figure 3 — fold score ranges before/after OOF-logit-z ───────────────────
print("[3] Fold score ranges …")

beta_sr = sr_fold[sr_fold["candidate"] == "recover035_latent384_beta3p75"].copy()
oof_sr  = beta_sr[beta_sr["calib_method"] == "oof_logitz"].sort_values("fold")
raw_sr  = beta_sr[beta_sr["calib_method"] == "raw"].sort_values("fold")

folds = np.arange(1, 6)
x     = np.arange(5)
w     = 0.28

fig, ax = plt.subplots(figsize=(5.2, 3.4))

b1 = ax.bar(x - w, raw_sr["oof_range_raw"].values,  w, color="#AAAAAA", label="OOF train score range (raw)",    zorder=3)
b2 = ax.bar(x,     raw_sr["test_range_raw"].values,  w, color=C_RAW,    label="Test score range — raw",         zorder=3, alpha=0.85)
b3 = ax.bar(x + w, oof_sr["test_range_calib"].values,w, color=C_LOGITZ, label="Test score range — OOF-logit-z", zorder=3, alpha=0.85)

# annotate fold 1 raw test range
ax.annotate(f"Fold 1\nraw range\n= {raw_sr[raw_sr['fold']==1]['test_range_raw'].values[0]:.3f}",
            xy=(0, raw_sr[raw_sr['fold']==1]['test_range_raw'].values[0]),
            xytext=(0.5, 0.89), textcoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color=C_RAW, lw=1.2),
            fontsize=7.5, color=C_RAW, ha="center")

ax.set_xticks(x)
ax.set_xticklabels([f"Fold {f}" for f in folds])
ax.set_ylabel("Score range (max − min)")
ax.set_ylim(0, 1.12)
ax.set_title("Cross-fold score range: raw vs OOF-logit-z\n(beta3p75, logreg_l2, z_plus_age_sex)", fontsize=9, pad=6)
ax.legend(fontsize=7.5, framealpha=0.9)
ax.yaxis.grid(True, lw=0.5, alpha=0.4, zorder=0)
_style_ax(ax)
fig.tight_layout(pad=PAD)
_save(fig, "fig3_fold_score_ranges")

# ─── Figure 4 — score distributions by fold and diagnosis ────────────────────
print("[4] Score distributions …")

# Filter stageB score ranges: logreg_l2_original, z_plus_age_sex, raw & oof_logitz
stb = stage_sr[
    (stage_sr["model_name"] == "logreg_l2_original") &
    (stage_sr["feature_set"] == "z_plus_age_sex") &
    (stage_sr["calib_method"].isin(["raw", "oof_logitz"])) &
    (stage_sr["diagnosis"].isin(["CN", "AD"]))
].copy()

fig, axes = plt.subplots(2, 5, figsize=(9.5, 5.0), sharey=False, sharex=False)
# leave space on left for row-label text
fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.06, hspace=0.35, wspace=0.25)

ROW_LABELS = ["Raw scores", "OOF-logit-z scores"]

for row_i, calib in enumerate(["raw", "oof_logitz"]):
    for col_i, fold in enumerate(range(1, 6)):
        ax = axes[row_i, col_i]
        sub = stb[(stb["fold"] == fold) & (stb["calib_method"] == calib)]

        for diag, col in [("CN", C_CN), ("AD", C_AD)]:
            r = sub[sub["diagnosis"] == diag]
            if r.empty:
                continue
            r = r.iloc[0]
            xpos = 0.5 if diag == "CN" else 1.5
            # IQR box (p25–p75)
            ax.bar(xpos, r["p75"] - r["p25"], bottom=r["p25"],
                   width=0.55, color=col, alpha=0.65, zorder=3)
            # whiskers p10–p90
            ax.plot([xpos, xpos], [r["p10"], r["p90"]],
                    color=col, lw=1.5, zorder=4)
            # min/max ticks
            for yv in [r["min"], r["max"]]:
                ax.plot(xpos, yv, color=col, marker="_", ms=6, mew=1.5, zorder=4)
            # median line
            ax.plot(xpos, r["median"], "w|", mew=2.5, ms=7, zorder=5)

        ax.set_xticks([0.5, 1.5])
        ax.set_xticklabels(["CN", "AD"], fontsize=8)
        ax.set_xlim(0, 2)
        ax.set_ylim(-0.05, 1.10)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        if col_i == 0:
            ax.set_yticklabels(["0", ".25", ".5", ".75", "1"], fontsize=7)
        else:
            ax.set_yticklabels([])
        if row_i == 0:
            ax.set_title(f"Fold {fold}", fontsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, lw=0.4, alpha=0.35, zorder=0)

    # row label — centred vertically on each row using the leftmost axis bbox
    ax0 = axes[row_i, 0]
    fig.text(0.005, ax0.get_position().y0 + ax0.get_position().height / 2,
             ROW_LABELS[row_i], va="center", ha="left",
             fontsize=9, fontweight="bold", rotation=90)

# legend
cn_patch = mpatches.Patch(color=C_CN, alpha=0.75, label="CN")
ad_patch = mpatches.Patch(color=C_AD, alpha=0.75, label="AD")
fig.legend(handles=[cn_patch, ad_patch], loc="upper right",
           fontsize=8, framealpha=0.9, ncol=2,
           bbox_to_anchor=(0.97, 0.97))

fig.suptitle("Score distributions by fold and diagnosis — beta3p75 OOF-logit-z\n"
             "(bar = IQR, whiskers = p10–p90, line = median)",
             fontsize=9, y=0.97)
_save(fig, "fig4_score_distributions")

# ─── Figure 5 — foldwise AUC / PR-AUC barplot ────────────────────────────────
print("[5] Foldwise AUC …")

b3p75 = fw_metrics[fw_metrics["candidate"] == "recover035_latent384_beta3p75"].copy()
raw_fw = b3p75[b3p75["calib_method"] == "raw"].sort_values("fold")
lof_fw = b3p75[b3p75["calib_method"] == "oof_logitz"].sort_values("fold")

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))

metrics = [("auc", "AUC"), ("pr_auc", "PR-AUC")]
for ax, (col, label) in zip(axes, metrics):
    x  = np.arange(5)
    w  = 0.30
    ax.bar(x - w/2, raw_fw[col].values, w, color=C_RAW,    alpha=0.85,
           label=LABEL_RAW,    zorder=3)
    ax.bar(x + w/2, lof_fw[col].values, w, color=C_LOGITZ, alpha=0.85,
           label=LABEL_LOGITZ, zorder=3)
    # pooled line
    pooled_r = b3p75[b3p75["calib_method"] == "raw"][col].mean() if False else \
               raw_fw[col].values.mean()   # foldwise mean for reference
    pooled_c = lof_fw[col].values.mean()
    ax.axhline(pooled_r, color=C_RAW,    ls="--", lw=1.1, alpha=0.7,
               label=f"Foldwise mean (raw) = {pooled_r:.3f}")
    ax.axhline(pooled_c, color=C_LOGITZ, ls="--", lw=1.1, alpha=0.7,
               label=f"Foldwise mean (logitz) = {pooled_c:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i}" for i in range(1, 6)], fontsize=8)
    ax.set_ylabel(label, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_title(f"Foldwise {label}", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower right", framealpha=0.9)
    ax.yaxis.grid(True, lw=0.5, alpha=0.4, zorder=0)
    _style_ax(ax)

fig.suptitle("Foldwise AUC and PR-AUC — beta3p75 raw vs OOF-logit-z",
             fontsize=9, y=1.02)
fig.tight_layout(pad=PAD)
_save(fig, "fig5_foldwise_auc_bar")

# ─── Figure 6 — confusion matrix (promoted candidate, pooled) ────────────────
print("[6] Confusion matrix …")

# Read pooled metrics for promoted
pm = pd.read_csv(ALL_COMP / "pooled_metrics.csv")
row = pm[(pm["candidate"] == "recover035_latent384_beta3p75") &
         (pm["calib_method"] == "oof_logitz")].iloc[0]
tp, fp, fn, tn = int(row["tp"]), int(row["fp"]), int(row["fn"]), int(row["tn"])
cm = np.array([[tn, fp], [fn, tp]])
n_total = tp + fp + fn + tn

fig, ax = plt.subplots(figsize=(3.4, 3.6))
fig.subplots_adjust(top=0.88, bottom=0.28, left=0.18, right=0.95)
cmap = plt.cm.Blues
im = ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=0, vmax=n_total // 2)

classes = ["CN (pred −)", "AD (pred +)"]
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(classes, fontsize=8.5)
ax.set_yticklabels(["CN (true)", "AD (true)"], fontsize=8.5)
ax.set_xlabel("Predicted label", fontsize=9)
ax.set_ylabel("True label", fontsize=9)
ax.set_title("Confusion matrix — promoted candidate\n"
             "(beta3p75 OOF-logit-z, pooled, n=397)", fontsize=8.5, pad=6)

thresh = cm.max() / 2.0
for i in range(2):
    for j in range(2):
        pct = cm[i, j] / n_total * 100
        ax.text(j, i, f"{cm[i, j]}\n({pct:.1f}%)",
                ha="center", va="center", fontsize=9.5,
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold")

# metrics box — placed in figure space below the axes
sens = tp / (tp + fn); spec = tn / (tn + fp)
f1   = 2*tp / (2*tp + fp + fn)
ba   = (sens + spec) / 2
ann  = (f"Sens = {sens:.3f}   Spec = {spec:.3f}\n"
        f"BA = {ba:.3f}   F1 = {f1:.3f}\n"
        f"AUC = {row['auc']:.3f}   PR-AUC = {row['pr_auc']:.3f}")
fig.text(0.565, 0.13, ann, ha="center", va="top", fontsize=7.5,
         transform=fig.transFigure,
         bbox=dict(boxstyle="round,pad=0.35", fc="#f8f8f8", ec="#cccccc"))

_save(fig, "fig6_confusion_matrix")

# ─── Figure 7 — Philips CN FP and GE AD FN subgroup bars ─────────────────────
print("[7] Subgroup bars …")

# Extract locked and promoted logitz for philips and GE
keep = philips_fp[
    (philips_fp["candidate"].isin(["locked_v5p1b", "recover035_latent384_beta3p75"])) &
    (philips_fp["calib_method"].isin(["raw", "oof_logitz"]))
].copy()

# We show 3 rows: GE CN, Philips CN, Siemens CN  →  FPR
# and GE AD sensitivity
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))

# Panel A — CN FPR by manufacturer
ax = axes[0]
mfrs   = ["GE", "Philips", "SIEMENS"]
mfr_labels = ["GE", "Philips", "Siemens"]
arms = [
    ("locked_v5p1b",                  "raw",        C_LOCKED, LABEL_LOCKED),
    ("recover035_latent384_beta3p75",  "oof_logitz", C_LOGITZ, LABEL_LOGITZ),
]
x = np.arange(len(mfrs))
w = 0.30
for i, (cand, calib, col, lbl) in enumerate(arms):
    vals = []
    for mfr in mfrs:
        r = keep[(keep["candidate"] == cand) &
                 (keep["calib_method"] == calib) &
                 (keep["manufacturer"] == mfr)]
        vals.append(float(r["fpr_cn"].values[0]) if len(r) else np.nan)
    ax.bar(x + (i - 0.5) * w, vals, w, color=col, alpha=0.85, label=lbl, zorder=3)
    for xi, v in zip(x + (i - 0.5) * w, vals):
        if not np.isnan(v):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x)
ax.set_xticklabels(mfr_labels, fontsize=9)
ax.set_ylabel("CN False-Positive Rate", fontsize=9)
ax.set_ylim(0, 0.70)
ax.set_title("CN FPR by scanner manufacturer", fontsize=9, pad=6)
ax.legend(fontsize=7, framealpha=0.9, loc="upper right")
ax.yaxis.grid(True, lw=0.5, alpha=0.4, zorder=0)
_style_ax(ax)

# Panel B — GE AD sensitivity and overall comparison
ax = axes[1]
ge_sub = ge_fn[
    (ge_fn["candidate"].isin(["locked_v5p1b", "recover035_latent384_beta3p75"])) &
    (ge_fn["calib_method"].isin(["raw", "oof_logitz"]))
].copy()

# Show: GE AD sensitivity + overall sensitivity
pm_sub = pd.read_csv(ALL_COMP / "pooled_metrics.csv")
rows_compare = [
    ("locked_v5p1b",                  "raw",        C_LOCKED, "Locked v5.1b (raw)"),
    ("recover035_latent384_beta3p75",  "oof_logitz", C_LOGITZ, "beta3p75 (OOF-logit-z)"),
]
categories = ["Overall\nAD sensitivity", "GE AD\nsensitivity"]
x2 = np.arange(len(categories))
for i, (cand, calib, col, lbl) in enumerate(rows_compare):
    r_pm  = pm_sub[(pm_sub["candidate"] == cand) & (pm_sub["calib_method"] == calib)].iloc[0]
    r_ge  = ge_sub[(ge_sub["candidate"] == cand) & (ge_sub["calib_method"] == calib)].iloc[0]
    vals2 = [float(r_pm["sensitivity"]), float(r_ge["sensitivity_ge_ad"])]
    bars  = ax.bar(x2 + (i - 0.5) * w, vals2, w, color=col, alpha=0.85, label=lbl, zorder=3)
    for xi, v in zip(x2 + (i - 0.5) * w, vals2):
        ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5)

ax.set_xticks(x2)
ax.set_xticklabels(categories, fontsize=8.5)
ax.set_ylabel("Sensitivity", fontsize=9)
ax.set_ylim(0, 1.0)
ax.set_title("AD sensitivity: overall vs GE subgroup", fontsize=9, pad=6)
ax.legend(fontsize=7, framealpha=0.9, loc="upper right")
ax.yaxis.grid(True, lw=0.5, alpha=0.4, zorder=0)
_style_ax(ax)

fig.suptitle("Subgroup performance: scanner manufacturer", fontsize=9, y=1.01)
fig.tight_layout(pad=PAD)
_save(fig, "fig7_subgroup_bars")

# ─── Figure 8 — reliability / Brier / ECE ────────────────────────────────────
print("[8] Reliability / calibration diagnostic …")

def _reliability_curve(y_true, y_score, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_score, bins[1:-1])
    frac_pos, mean_pred = [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        frac_pos.append(y_true[mask].mean())
        mean_pred.append(y_score[mask].mean())
    return np.array(mean_pred), np.array(frac_pos)

def _brier(y_true, y_score):
    return float(np.mean((y_score - y_true) ** 2))

def _ece(y_true, y_score, n_bins=10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_score, bins[1:-1])
    n = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        ece += mask.sum() / n * abs(y_true[mask].mean() - y_score[mask].mean())
    return ece

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))

# Panel A — reliability diagram
ax = axes[0]
ax.plot([0, 1], [0, 1], "k--", lw=0.9, alpha=0.4, label="Perfect calibration")

for df, col, label in [
    (pL,  C_LOCKED, LABEL_LOCKED),
    (pRR, C_RAW,    LABEL_RAW),
    (pOF, C_LOGITZ, LABEL_LOGITZ),
]:
    yt = df["y_true"].values
    ys = df["y_score"].values
    mp, fp_ = _reliability_curve(yt, ys, n_bins=10)
    bri = _brier(yt, ys)
    ece = _ece(yt, ys, n_bins=10)
    lbl = f"{label}\nBrier={bri:.3f}, ECE={ece:.3f}"
    ax.plot(mp, fp_, "o-", color=col, lw=1.6, ms=4.5, label=lbl, alpha=0.9)

ax.set_xlabel("Mean predicted probability", fontsize=9)
ax.set_ylabel("Fraction of positives", fontsize=9)
ax.set_title("Reliability diagram", fontsize=9, pad=6)
ax.legend(fontsize=6.5, loc="upper left", framealpha=0.9)
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.05, 1.10)
ax.text(0.97, 0.04,
        "Note: OOF-logit-z optimises rank order (AUC),\nnot probability calibration.\n"
        "ECE/Brier are diagnostic only.",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=6.8, color="#555555",
        bbox=dict(boxstyle="round,pad=0.3", fc="#fff9e6", ec="#ddcc88", alpha=0.92))
_style_ax(ax)

# Panel B — Brier and ECE bar comparison
ax = axes[1]
rows = [
    ("locked_v5p1b",                  "raw",        C_LOCKED, "Locked\n(raw)"),
    ("recover035_latent384_beta3p75",  "raw",        C_RAW,    "beta3p75\n(raw)"),
    ("recover035_latent384_beta3p75",  "oof_logitz", C_LOGITZ, "beta3p75\n(OOF-logit-z)"),
]
metrics_be = ["brier", "ece_10bins"]
metric_labels = ["Brier score", "ECE (10-bin)"]
x3 = np.arange(len(rows))
w3 = 0.30
for mi, (mc, ml) in enumerate(zip(metrics_be, metric_labels)):
    vals = []
    for cand, calib, col, _ in rows:
        r = brier_ece[(brier_ece["candidate"] == cand) & (brier_ece["calib_method"] == calib)]
        vals.append(float(r[mc].values[0]))
    offset = (mi - 0.5) * w3
    bars = ax.bar(x3 + offset, vals, w3, label=ml,
                  color=[C_LOCKED, C_RAW, C_LOGITZ],
                  alpha=0.85 if mi == 0 else 0.55,
                  hatch="" if mi == 0 else "///",
                  zorder=3)
    for xi, v in zip(x3 + offset, vals):
        ax.text(xi, v + 0.003, f"{v:.3f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x3)
ax.set_xticklabels([r[3] for r in rows], fontsize=8)
ax.set_ylabel("Score (lower = better)", fontsize=9)
ax.set_ylim(0, 0.38)
ax.set_title("Brier score & ECE\n(probability calibration diagnostic)", fontsize=9, pad=6)
hatch_patch = mpatches.Patch(facecolor="#aaaaaa", hatch="///", alpha=0.55, label="ECE (10-bin)")
solid_patch  = mpatches.Patch(facecolor="#aaaaaa", alpha=0.85, label="Brier score")
ax.legend(handles=[solid_patch, hatch_patch], fontsize=7.5, loc="upper right", framealpha=0.9)
ax.yaxis.grid(True, lw=0.5, alpha=0.4, zorder=0)
_style_ax(ax)

fig.suptitle("Probability calibration diagnostic — not optimised by OOF-logit-z",
             fontsize=9, y=1.01)
fig.tight_layout(pad=PAD)
_save(fig, "fig8_reliability_calibration")

# ─── command log ─────────────────────────────────────────────────────────────
log = {
    "script": "scripts/revision_bspc_2026/run_oof_logitz_promoted_figures.py",
    "generated_utc": datetime.now(timezone.utc).isoformat(),
    "inputs": [
        str(ALL_COMP),
        str(STAT_VAL),
        str(STAGE_B),
    ],
    "output_dir": str(OUT_DIR),
    "figures": [
        "fig1_roc_pooled",
        "fig2_pr_pooled",
        "fig3_fold_score_ranges",
        "fig4_score_distributions",
        "fig5_foldwise_auc_bar",
        "fig6_confusion_matrix",
        "fig7_subgroup_bars",
        "fig8_reliability_calibration",
    ],
    "training_launched": False,
    "model_selection": False,
}
with open(OUT_DIR / "command_log.json", "w") as f:
    json.dump(log, f, indent=2)

print(f"\nDone. All figures written to {OUT_DIR}")
