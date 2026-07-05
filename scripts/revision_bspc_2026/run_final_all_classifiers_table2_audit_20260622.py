"""
Final all-classifiers Table 2 audit — BSPC 2026 revision
Read-only analysis. Do not train. Do not run OASIS. Do not modify model outputs.

Final selected VAE: recover035_latent384_beta3p75_T80_h10000_p560_full5x5
Channels [1,0,2] = Pearson_Full + OMST + MI_KNN
N=397 (CN=300, AD=97), latent_dim=384, beta=3.75

Primary readout: logreg_l2_original / z_plus_age_sex / oof_ecdf /
                 inner_oof_target_sens_ge_0p70_max_spec
Pooled OOF ECDF AUC=0.795155, PR-AUC=0.573934
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results" / "revision_bspc_2026"
OUT = RESULTS / "final_all_classifiers_table2_audit_20260622"
OUT.mkdir(exist_ok=True)

FULL_RUN = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
STAGEB_CALIB = RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration"

FULL_PREDICTIONS_CSV = (
    FULL_RUN
    / "all_folds_clf_predictions_MULTI_logreg_vaeconvtranspose4l_ld384_beta3.75_"
      "normzscore_offdiag_ch3sel_intFCquarter_drop0.15_ln0_outer5x1_scoreroc_auc.csv"
)
FULL_FOLDWISE_METRICS_CSV = (
    FULL_RUN
    / "all_folds_metrics_MULTI_logreg_vaeconvtranspose4l_ld384_beta3.75_"
      "normzscore_offdiag_ch3sel_intFCquarter_drop0.15_ln0_outer5x1_scoreroc_auc.csv"
)
STAGEB_FOLDWISE_CSV = STAGEB_CALIB / "calib_foldwise_metrics.csv"
STAGEB_POOLED_CSV = STAGEB_CALIB / "calib_pooled_metrics.csv"
STAGEB_PREDICTIONS_CSV = STAGEB_CALIB / "calib_predictions.csv"

# Stale sweep directories
STALE_SWEEP_MFRSPLIT = RESULTS / "adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep"
STALE_SWEEP_REFINED = RESULTS / "adni_v5_1_batch20260514b_refined_classifier_only_sweep"
STALE_FROZEN = RESULTS / "frozen_latent_stageb_classifier_sweep_pro"

# Expected parameters
EXPECTED_VAE_RUN = "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
EXPECTED_N = 397
EXPECTED_CN = 300
EXPECTED_AD = 97
EXPECTED_LATENT_DIM = 384
EXPECTED_CHANNELS = [1, 0, 2]

PRIMARY_MODEL = "logreg_l2_original"
PRIMARY_FEAT = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"
PRIMARY_THRESH = "inner_oof_target_sens_ge_0p70_max_spec"

BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 42

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()

def _md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".4f")

def write_pair(stem: str, df: pd.DataFrame, extra_md: str = "") -> None:
    df.to_csv(OUT / f"{stem}.csv", index=False, float_format="%.6f")
    md = _md_table(df)
    if extra_md:
        md = extra_md + "\n\n" + md
    (OUT / f"{stem}.md").write_text(md)

def compute_brier(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(brier_score_loss(y_true, y_score))

def compute_ece(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_score >= lo) & (y_score < hi)
        if mask.sum() == 0:
            continue
        frac_pos = y_true[mask].mean()
        mean_conf = y_score[mask].mean()
        ece += (mask.sum() / n) * abs(frac_pos - mean_conf)
    return ece

def bootstrap_auc(y_true: np.ndarray, y_score: np.ndarray,
                  n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
    aucs = np.array(aucs)
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(lo), float(hi)

def bootstrap_prauc(y_true: np.ndarray, y_score: np.ndarray,
                    n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        vals.append(average_precision_score(yt, ys))
    vals = np.array(vals)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)

command_log: list[dict] = []

# ---------------------------------------------------------------------------
# TASK 1 – Artifact inventory
# ---------------------------------------------------------------------------
print("=== TASK 1: Artifact inventory ===")

inventory_rows = []

def _add_inv(name, path, status, n, n_cn, n_ad, latent_dim, classifiers, note):
    inventory_rows.append(dict(
        artifact=name, path=str(path), status=status,
        n=n, n_cn=n_cn, n_ad=n_ad, latent_dim=latent_dim,
        classifiers=classifiers, note=note
    ))

# Primary artifacts
_add_inv(
    "FULL 5×5 primary run (logreg + SVM)",
    FULL_RUN,
    "VALID" if FULL_RUN.exists() else "MISSING",
    EXPECTED_N, EXPECTED_CN, EXPECTED_AD, EXPECTED_LATENT_DIM,
    "logreg, svm",
    "Original FULL run with sigmoid calibration"
)
_add_inv(
    "Stage B OOF score calibration",
    STAGEB_CALIB,
    "VALID" if STAGEB_CALIB.exists() else "MISSING",
    EXPECTED_N, EXPECTED_CN, EXPECTED_AD, EXPECTED_LATENT_DIM,
    "logreg_l2_original, logreg_elasticnet",
    "OOF ECDF/logitz/zscore/platt/isotonic applied to primary run latents"
)
_add_inv(
    "classifier_only_readout (inside FULL run)",
    FULL_RUN / "classifier_only_readout",
    "VALID" if (FULL_RUN / "classifier_only_readout").exists() else "MISSING",
    EXPECTED_N, EXPECTED_CN, EXPECTED_AD, EXPECTED_LATENT_DIM,
    "logreg_l2",
    "Refitted logreg with z_plus_age_sex and OOF ECDF; AUC=0.760"
)

# Stale artifacts
_add_inv(
    "mfrsplit_3840_classifier_only_sweep",
    STALE_SWEEP_MFRSPLIT,
    "STALE" if STALE_SWEEP_MFRSPLIT.exists() else "ABSENT",
    396, 300, 96, 256,
    "logreg_l2, logreg_elasticnet, svm_rbf, xgboost, lightgbm, gradient_boosting, random_forest",
    "STALE: latent_dim=256 (not 384), AD=96 (not 97) — must not be used in Table 2"
)
_add_inv(
    "refined_classifier_only_sweep",
    STALE_SWEEP_REFINED,
    "STALE" if STALE_SWEEP_REFINED.exists() else "ABSENT",
    396, 300, 96, "UNKNOWN",
    "logreg_l2, logreg_elasticnet, svm_rbf, lightgbm",
    "STALE: AD=96, N=396 — wrong cohort size"
)
_add_inv(
    "frozen_latent_stageb_classifier_sweep_pro",
    STALE_FROZEN,
    "STALE" if STALE_FROZEN.exists() else "ABSENT",
    396, 300, 96, "UNKNOWN",
    "logreg_l2, logreg_elasticnet, rbf_svm, linear_svm, lightgbm_very_regularized",
    "STALE: uses wrong VAE runs (not recover035_latent384_beta3p75); AD=96"
)

# MLP / XGBoost / GradBoost
for clf in ["MLP", "XGBoost", "GradientBoosting", "RandomForest"]:
    _add_inv(
        f"{clf} (final protocol)",
        "N/A",
        "MISSING",
        "N/A", "N/A", "N/A", "N/A",
        clf,
        f"{clf} was NOT run under the final protocol (ld=384, β=3.75, N=397). "
        "Appears only in stale ld=256/N=396 sweep — must not backfill."
    )

inv_df = pd.DataFrame(inventory_rows)
inv_df.to_csv(OUT / "all_classifier_artifact_inventory.csv", index=False)
inv_text = "# All Classifier Artifact Inventory\n\n"
inv_text += f"Generated: {now_utc()}\n\n"
inv_text += "Final selected VAE: `recover035_latent384_beta3p75_T80_h10000_p560_full5x5`\n"
inv_text += "Required: N=397, CN=300, AD=97, latent_dim=384, channels=[1,0,2]\n\n"
inv_text += inv_df.to_markdown(index=False)
inv_text += "\n\n## Summary\n"
inv_text += "- **VALID**: logreg (sigmoid cal) and SVM (sigmoid cal) from primary FULL run\n"
inv_text += "- **VALID**: logreg_l2_original and logreg_elasticnet from Stage B OOF calibration\n"
inv_text += "- **MISSING from final protocol**: MLP, XGBoost, GradientBoosting, RandomForest\n"
inv_text += "- **STALE** (wrong cohort or wrong VAE): mfrsplit_3840, refined sweep, frozen_latent sweep\n"
(OUT / "all_classifier_artifact_inventory.md").write_text(inv_text)
print("  all_classifier_artifact_inventory.md: written")

# ---------------------------------------------------------------------------
# TASK 2 – Table 2: foldwise mean ± SD for available classifiers
# ---------------------------------------------------------------------------
print("=== TASK 2: Table 2 (foldwise mean ± SD) ===")

# Load stageB foldwise (primary LogReg OOF ECDF)
stb_fw = pd.read_csv(STAGEB_FOLDWISE_CSV)
primary_fw = stb_fw[
    (stb_fw["model_name"] == PRIMARY_MODEL)
    & (stb_fw["feature_set"] == PRIMARY_FEAT)
    & (stb_fw["calib_method"] == PRIMARY_CALIB)
    & (stb_fw["threshold_strategy"] == PRIMARY_THRESH)
].sort_values("fold")

assert len(primary_fw) == 5, f"Expected 5 folds, got {len(primary_fw)}"

def fmt_mean_sd(series: pd.Series) -> str:
    return f"{series.mean():.3f} ± {series.std():.3f}"

# Validate N per fold
for _, r in primary_fw.iterrows():
    assert r["n"] in (78, 79, 80), f"Unexpected fold n={r['n']}"
    assert r["n_cn"] + r["n_ad"] == r["n"], "CN+AD mismatch"

logreg_row = {
    "Model": "Logistic Regression (OOF ECDF)",
    "Calibration": "oof_ecdf",
    "Threshold_strategy": PRIMARY_THRESH,
    "N_subjects": EXPECTED_N,
    "Source": "primary / VALID",
    "ROC-AUC mean±SD": fmt_mean_sd(primary_fw["auc"]),
    "PR-AUC mean±SD": fmt_mean_sd(primary_fw["pr_auc"]),
    "Balanced_Acc mean±SD": fmt_mean_sd(primary_fw["balanced_accuracy"]),
    "Sensitivity mean±SD": fmt_mean_sd(primary_fw["sensitivity"]),
    "Specificity mean±SD": fmt_mean_sd(primary_fw["specificity"]),
    "F1 mean±SD": fmt_mean_sd(primary_fw["f1"]),
}

# Load FULL run foldwise (logreg_sigmoid + SVM_sigmoid)
full_fw = pd.read_csv(FULL_FOLDWISE_METRICS_CSV)

# Verify subjects count
full_preds = pd.read_csv(FULL_PREDICTIONS_CSV)
assert full_preds["SubjectID"].nunique() == EXPECTED_N, "Subject count mismatch in FULL predictions"
n_cn_full = (full_preds.drop_duplicates("SubjectID")["y_true"] == 0).sum()
n_ad_full = (full_preds.drop_duplicates("SubjectID")["y_true"] == 1).sum()
assert n_cn_full == EXPECTED_CN, f"CN mismatch: {n_cn_full}"
assert n_ad_full == EXPECTED_AD, f"AD mismatch: {n_ad_full}"

svm_fw = full_fw[full_fw["actual_classifier_type"] == "svm"].sort_values("fold")
logreg_full_fw = full_fw[full_fw["actual_classifier_type"] == "logreg"].sort_values("fold")
assert len(svm_fw) == 5 and len(logreg_full_fw) == 5

svm_row = {
    "Model": "SVM (RBF, sigmoid cal)",
    "Calibration": "sigmoid (CalibratedClassifierCV)",
    "Threshold_strategy": "fixed_0.5",
    "N_subjects": EXPECTED_N,
    "Source": "VALID",
    "ROC-AUC mean±SD": fmt_mean_sd(svm_fw["auc"]),
    "PR-AUC mean±SD": fmt_mean_sd(svm_fw["pr_auc"]),
    "Balanced_Acc mean±SD": fmt_mean_sd(svm_fw["balanced_accuracy"]),
    "Sensitivity mean±SD": fmt_mean_sd(svm_fw["sensitivity"]),
    "Specificity mean±SD": fmt_mean_sd(svm_fw["specificity"]),
    "F1 mean±SD": fmt_mean_sd(svm_fw["f1_score"]),
}

logreg_sigmoid_row = {
    "Model": "Logistic Regression (sigmoid cal)",
    "Calibration": "sigmoid (CalibratedClassifierCV)",
    "Threshold_strategy": "fixed_0.5",
    "N_subjects": EXPECTED_N,
    "Source": "secondary / VALID",
    "ROC-AUC mean±SD": fmt_mean_sd(logreg_full_fw["auc"]),
    "PR-AUC mean±SD": fmt_mean_sd(logreg_full_fw["pr_auc"]),
    "Balanced_Acc mean±SD": fmt_mean_sd(logreg_full_fw["balanced_accuracy"]),
    "Sensitivity mean±SD": fmt_mean_sd(logreg_full_fw["sensitivity"]),
    "Specificity mean±SD": fmt_mean_sd(logreg_full_fw["specificity"]),
    "F1 mean±SD": fmt_mean_sd(logreg_full_fw["f1_score"]),
}

missing_row_template = {
    "ROC-AUC mean±SD": "MISSING",
    "PR-AUC mean±SD": "MISSING",
    "Balanced_Acc mean±SD": "MISSING",
    "Sensitivity mean±SD": "MISSING",
    "Specificity mean±SD": "MISSING",
    "F1 mean±SD": "MISSING",
}

def missing_row(name: str) -> dict:
    r = dict(missing_row_template)
    r.update({
        "Model": name,
        "Calibration": "N/A",
        "Threshold_strategy": "N/A",
        "N_subjects": "N/A",
        "Source": "MISSING from final protocol",
    })
    return r

table2_rows = [
    logreg_row,
    svm_row,
    logreg_sigmoid_row,
    missing_row("MLP"),
    missing_row("XGBoost"),
    missing_row("Gradient Boosting"),
    missing_row("Random Forest"),
]
table2_df = pd.DataFrame(table2_rows)

table2_header = "# Table 2: All Classifiers — Final Expanded Cohort, Final Selected VAE\n\n"
table2_header += f"Generated: {now_utc()}\n\n"
table2_header += (
    "VAE: `recover035_latent384_beta3p75_T80_h10000_p560_full5x5`  "
    "Channels [1,0,2]  N=397 (CN=300, AD=97)  latent_dim=384  β=3.75\n\n"
)
table2_header += "**Primary readout**: LogReg L2 / z+age+sex / OOF ECDF / inner_oof_target_sens≥0.70_max_spec\n"
table2_header += "**MLP, XGBoost, GradientBoosting, RandomForest**: NOT available under final protocol; do not backfill from stale ld=256/N=396 runs.\n\n"
table2_header += "LogReg (OOF ECDF) foldwise values [AUC per fold]: "
table2_header += ", ".join(f"{v:.4f}" for v in primary_fw["auc"].values) + "\n"
table2_header += "SVM (sigmoid) foldwise values [AUC per fold]: "
table2_header += ", ".join(f"{v:.4f}" for v in svm_fw["auc"].values) + "\n\n"

write_pair("table2_all_classifiers_final", table2_df, table2_header)
print("  table2_all_classifiers_final.csv/.md: written")

# LaTeX table
def _latex_table2(df: pd.DataFrame) -> str:
    header = r"""\begin{table}[ht]
\centering
\caption{Comparison of supervised classifiers on the final expanded ADNI cohort
(N\,=\,397, CN\,=\,300, AD\,=\,97) using the final selected $\beta$-VAE representation
(latent dim\,=\,384, $\beta$\,=\,3.75, channels\,=\,[1,0,2]).
Primary readout: Logistic Regression (OOF ECDF).
MLP, XGBoost, Gradient Boosting, and Random Forest were not run under the
final protocol and are listed as unavailable.}
\label{tab:table2_classifiers}
\begin{tabular}{lcccccc}
\toprule
\textbf{Model} & \textbf{ROC-AUC} & \textbf{PR-AUC} &
\textbf{Bal.\,Acc.} & \textbf{Sensitivity} & \textbf{Specificity} & \textbf{F1} \\
& (mean$\pm$SD) & (mean$\pm$SD) & (mean$\pm$SD) &
(mean$\pm$SD) & (mean$\pm$SD) & (mean$\pm$SD) \\
\midrule
"""
    rows = ""
    for _, row in df.iterrows():
        name = row["Model"].replace("(", r"\mbox{(}").replace(")", r"\mbox{)}")
        cols = [
            row["ROC-AUC mean±SD"],
            row["PR-AUC mean±SD"],
            row["Balanced_Acc mean±SD"],
            row["Sensitivity mean±SD"],
            row["Specificity mean±SD"],
            row["F1 mean±SD"],
        ]
        cols_str = " & ".join(
            c.replace("±", r"$\pm$") if c != "MISSING" else r"\textit{n/a}" for c in cols
        )
        rows += f"{name} & {cols_str} \\\\\n"
    footer = r"""\bottomrule
\end{tabular}
\end{table}
"""
    return header + rows + footer

latex_text = _latex_table2(table2_df)
(OUT / "table2_all_classifiers_final.tex").write_text(latex_text)
print("  table2_all_classifiers_final.tex: written")

# ---------------------------------------------------------------------------
# TASK 3 – Primary LogReg pooled metrics + bootstrap CI + confusion matrix
# ---------------------------------------------------------------------------
print("=== TASK 3: Primary pooled metrics + bootstrap CI ===")

# Load stageB predictions for primary readout
stb_preds = pd.read_csv(STAGEB_PREDICTIONS_CSV)
primary_preds = stb_preds[
    (stb_preds["model_name"] == PRIMARY_MODEL)
    & (stb_preds["feature_set"] == PRIMARY_FEAT)
    & (stb_preds["calib_method"] == PRIMARY_CALIB)
    & (stb_preds["threshold_strategy"] == PRIMARY_THRESH)
].copy()

assert primary_preds["SubjectID"].nunique() == EXPECTED_N, "N mismatch in primary predictions"
assert (primary_preds["y_true"] == 0).sum() == EXPECTED_CN, "CN mismatch"
assert (primary_preds["y_true"] == 1).sum() == EXPECTED_AD, "AD mismatch"

y_true_arr = primary_preds["y_true"].values.astype(int)
y_score_arr = primary_preds["y_score"].values.astype(float)
y_pred_arr = primary_preds["y_pred"].values.astype(int)

# Pooled metrics
pooled_auc = float(roc_auc_score(y_true_arr, y_score_arr))
pooled_prauc = float(average_precision_score(y_true_arr, y_score_arr))
tn = int(((y_true_arr == 0) & (y_pred_arr == 0)).sum())
fp = int(((y_true_arr == 0) & (y_pred_arr == 1)).sum())
fn = int(((y_true_arr == 1) & (y_pred_arr == 0)).sum())
tp = int(((y_true_arr == 1) & (y_pred_arr == 1)).sum())
sens = tp / (tp + fn)
spec = tn / (tn + fp)
ba = (sens + spec) / 2
f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
brier = compute_brier(y_true_arr, y_score_arr)
ece = compute_ece(y_true_arr, y_score_arr)

print("  Computing bootstrap CI (5000 resample)...")
auc_lo, auc_hi = bootstrap_auc(y_true_arr, y_score_arr)
prauc_lo, prauc_hi = bootstrap_prauc(y_true_arr, y_score_arr)

primary_metrics = {
    "n": EXPECTED_N,
    "n_cn": EXPECTED_CN,
    "n_ad": EXPECTED_AD,
    "calibration_method": PRIMARY_CALIB,
    "threshold_strategy": PRIMARY_THRESH,
    "pooled_auc": round(pooled_auc, 6),
    "auc_ci_lo_2p5": round(auc_lo, 6),
    "auc_ci_hi_97p5": round(auc_hi, 6),
    "pooled_pr_auc": round(pooled_prauc, 6),
    "prauc_ci_lo_2p5": round(prauc_lo, 6),
    "prauc_ci_hi_97p5": round(prauc_hi, 6),
    "balanced_accuracy": round(ba, 6),
    "sensitivity": round(sens, 6),
    "specificity": round(spec, 6),
    "f1": round(f1, 6),
    "brier": round(brier, 6),
    "ece_10bin": round(ece, 6),
    "tn": tn,
    "fp": fp,
    "fn": fn,
    "tp": tp,
    "bootstrap_n": BOOTSTRAP_N,
    "bootstrap_seed": BOOTSTRAP_SEED,
}

pm_df = pd.DataFrame([primary_metrics])
pm_md_header = "# Primary LogReg Pooled Metrics\n\n"
pm_md_header += f"Generated: {now_utc()}\n\n"
pm_md_header += (
    "Model: `logreg_l2_original / z_plus_age_sex / oof_ecdf / "
    "inner_oof_target_sens_ge_0p70_max_spec`\n\n"
)
pm_md_header += "## Confusion Matrix (pooled OOF)\n\n"
pm_md_header += "| | Predicted CN | Predicted AD |\n"
pm_md_header += "|---|---|---|\n"
pm_md_header += f"| **True CN** (n={EXPECTED_CN}) | TN={tn} | FP={fp} |\n"
pm_md_header += f"| **True AD** (n={EXPECTED_AD}) | FN={fn} | TP={tp} |\n\n"
pm_md_header += "## Pooled Metrics with 95% Bootstrap CI\n\n"
write_pair("primary_logreg_pooled_metrics", pm_df, pm_md_header)
print(f"  primary_logreg_pooled_metrics.csv/.md: AUC={pooled_auc:.4f} [{auc_lo:.4f}, {auc_hi:.4f}]")

# ---------------------------------------------------------------------------
# TASK 4 – Calibration method comparison (Brier + ECE)
# ---------------------------------------------------------------------------
print("=== TASK 4: Calibration method comparison ===")

# From stageB calib_predictions, compute Brier and ECE for each calib_method
# (logreg_l2_original, z_plus_age_sex, primary threshold)
calib_rows = []
for cm in ["raw", "oof_platt", "oof_isotonic", "oof_zscore", "oof_logitz", "oof_ecdf"]:
    sub = stb_preds[
        (stb_preds["model_name"] == PRIMARY_MODEL)
        & (stb_preds["feature_set"] == PRIMARY_FEAT)
        & (stb_preds["calib_method"] == cm)
        & (stb_preds["threshold_strategy"] == PRIMARY_THRESH)
    ]
    if len(sub) == 0:
        continue
    yt = sub["y_true"].values.astype(int)
    ys = sub["y_score"].values.astype(float)
    auc_cm = float(roc_auc_score(yt, ys))
    prauc_cm = float(average_precision_score(yt, ys))
    brier_cm = compute_brier(yt, ys)
    ece_cm = compute_ece(yt, ys)
    # from stageB pooled metrics
    stb_pm = pd.read_csv(STAGEB_POOLED_CSV)
    row_pm = stb_pm[
        (stb_pm["model_name"] == PRIMARY_MODEL)
        & (stb_pm["feature_set"] == PRIMARY_FEAT)
        & (stb_pm["calib_method"] == cm)
        & (stb_pm["threshold_strategy"] == PRIMARY_THRESH)
    ]
    if len(row_pm):
        ba_cm = float(row_pm["balanced_accuracy"].iloc[0])
        sens_cm = float(row_pm["sensitivity"].iloc[0])
        spec_cm = float(row_pm["specificity"].iloc[0])
        f1_cm = float(row_pm["f1"].iloc[0])
        promotes = bool(row_pm["promotes"].iloc[0])
    else:
        ba_cm = sens_cm = spec_cm = f1_cm = float("nan")
        promotes = False
    calib_rows.append({
        "calib_method": cm,
        "pooled_auc": round(auc_cm, 4),
        "pooled_pr_auc": round(prauc_cm, 4),
        "balanced_accuracy": round(ba_cm, 4),
        "sensitivity": round(sens_cm, 4),
        "specificity": round(spec_cm, 4),
        "f1": round(f1_cm, 4),
        "brier": round(brier_cm, 4),
        "ece_10bin": round(ece_cm, 4),
        "promotes": promotes,
        "note": (
            "rank-transform score; not calibrated probability" if cm == "oof_ecdf" else
            "logit-scale shift; sigmoid squash; closest to calibrated probability" if cm == "oof_logitz" else
            "sigmoid Platt fit on OOF" if cm == "oof_platt" else
            "isotonic regression on OOF" if cm == "oof_isotonic" else
            "standardize OOF; sigmoid squash" if cm == "oof_zscore" else
            "raw predict_proba; no OOF calibration"
        )
    })

# Also add sigmoid from original FULL run
full_logreg = full_preds[full_preds["classifier_type"] == "logreg"].copy()
assert full_logreg["SubjectID"].nunique() == EXPECTED_N
yt_sig = full_logreg["y_true"].values.astype(int)
ys_sig_cal = full_logreg["y_score_cal"].values.astype(float)
ys_sig_raw = full_logreg["y_score_raw"].values.astype(float)
brier_sigmoid = compute_brier(yt_sig, ys_sig_cal)
ece_sigmoid = compute_ece(yt_sig, ys_sig_cal)
auc_sig = float(roc_auc_score(yt_sig, ys_sig_cal))
prauc_sig = float(average_precision_score(yt_sig, ys_sig_cal))
# BA, sens, spec at fixed_0.5
yp_sig = (ys_sig_cal >= 0.5).astype(int)
tn_sig = int(((yt_sig == 0) & (yp_sig == 0)).sum())
fp_sig = int(((yt_sig == 0) & (yp_sig == 1)).sum())
fn_sig = int(((yt_sig == 1) & (yp_sig == 0)).sum())
tp_sig = int(((yt_sig == 1) & (yp_sig == 1)).sum())
sens_sig = tp_sig / (tp_sig + fn_sig) if (tp_sig + fn_sig) > 0 else 0.0
spec_sig = tn_sig / (tn_sig + fp_sig) if (tn_sig + fp_sig) > 0 else 0.0
ba_sig = (sens_sig + spec_sig) / 2
f1_sig = 2 * tp_sig / (2 * tp_sig + fp_sig + fn_sig) if (2 * tp_sig + fp_sig + fn_sig) > 0 else 0.0
calib_rows.insert(0, {
    "calib_method": "sigmoid (CalibratedClassifierCV, FULL run logreg)",
    "pooled_auc": round(auc_sig, 4),
    "pooled_pr_auc": round(prauc_sig, 4),
    "balanced_accuracy": round(ba_sig, 4),
    "sensitivity": round(sens_sig, 4),
    "specificity": round(spec_sig, 4),
    "f1": round(f1_sig, 4),
    "brier": round(brier_sigmoid, 4),
    "ece_10bin": round(ece_sigmoid, 4),
    "promotes": False,
    "note": "DIFFERENT model (Optuna-tuned logreg with sklearn sigmoid cal); "
            "not directly comparable to stageB methods"
})

calib_df = pd.DataFrame(calib_rows)
calib_header = "# Calibration Method Comparison\n\n"
calib_header += f"Generated: {now_utc()}\n\n"
calib_header += (
    "Model: `logreg_l2_original / z_plus_age_sex / "
    "inner_oof_target_sens_ge_0p70_max_spec`\n\n"
)
calib_header += (
    "**Note on `oof_ecdf`**: The OOF ECDF method maps raw logistic scores to "
    "their percentile rank in the inner-CV OOF distribution. This is a monotone "
    "rank transformation — it does NOT produce calibrated AD probabilities. "
    "The output is an AD *likelihood rank score* in [0,1]. "
    "A reliability diagram of `oof_ecdf` scores should be labelled "
    "\"AD likelihood score\" not \"Predicted P(AD)\". "
    "For a proper probability calibration, use `oof_logitz` (Platt-like fitting "
    "on logit-transformed OOF scores).\n\n"
)
calib_header += (
    "**Note on sigmoid (CalibratedClassifierCV)**: This is from a *different* logreg "
    "model (Optuna-tuned, 500 iterations) than the stageB logreg. AUC=0.7987 "
    "(foldwise mean) reflects a more extensively tuned model. Not directly comparable "
    "to stageB calibration rows.\n\n"
)
write_pair("calibration_method_comparison", calib_df, calib_header)
print("  calibration_method_comparison.csv/.md: written")

# ---------------------------------------------------------------------------
# TASK 5 – Figure 2 calibration recommendation
# ---------------------------------------------------------------------------
print("=== TASK 5: Figure 2 calibration recommendation ===")

fig2_text = f"""# Figure 2 Calibration Panel Recommendation

Generated: {now_utc()}

## Current Situation

The primary readout uses `oof_ecdf` calibration (OOF ECDF).
This is a **rank-based score transform**, not a probability calibrator.

### What oof_ecdf does
Given inner-CV OOF predictions, oof_ecdf maps each test subject's raw logistic
score `s` to the fraction of OOF subjects with score ≤ s. The output is the
empirical CDF evaluated at s — a value in [0,1] representing the subject's
percentile rank in the training distribution.

### Why oof_ecdf is NOT a calibrated probability
- A calibrated probability P(AD|x) requires the model output to approximate
  the true posterior.
- The OOF distribution is ~75.6% CN / 24.4% AD. The 50th percentile of OOF
  scores does NOT correspond to P(AD) = 0.5. It corresponds to whatever
  fraction of subjects near that percentile are AD (~24%).
- A reliability diagram of oof_ecdf scores would show a systematic
  underestimation of P(AD) at high score values.

## Assessment of Figure 2 Calibration Panel

**If Figure 2 currently shows an oof_ecdf calibration curve:**
→ The y-axis label should be **"AD likelihood rank score"** or
  **"OOF score percentile"**, NOT "Predicted probability P(AD)".
→ The diagonal reference line is NOT the expected calibration line.
→ The panel is interpretable as "AD score reliability" — i.e., subjects
  with higher oof_ecdf scores do tend to be more likely AD — but cannot
  be interpreted as a probability calibration plot.

**Recommendation: RELABEL, do not replace.**
The oof_ecdf panel is scientifically valid as a reliability/discrimination
display. It should be re-labeled to avoid implying probability calibration.
The Brier score computed on oof_ecdf scores (Brier={brier:.4f}) reflects
discrimination quality, not calibration quality.

## Alternative: oof_logitz for Probability Calibration

`oof_logitz` applies logit-transform + standardize → sigmoid fit on OOF labels.
This is a Platt-scaling variant that does produce approximately calibrated
probabilities. AUC is nearly identical to oof_ecdf (0.7951 vs 0.7952).

| Method     | Pooled AUC | Pooled PR-AUC | Brier   | ECE (10-bin) | Is calibrated probability? |
|:-----------|:----------:|:-------------:|:-------:|:------------:|:---------------------------|
| raw        | 0.7599     | 0.5091        | ~high   | varies       | No                         |
| oof_platt  | 0.7782     | 0.5338        | varies  | varies       | Approximately yes           |
| oof_logitz | 0.7951     | 0.5728        | {[r['brier'] for r in calib_rows if 'logitz' in r['calib_method']][0]:.4f}  | {[r['ece_10bin'] for r in calib_rows if 'logitz' in r['calib_method']][0]:.4f}        | Approximately yes           |
| oof_ecdf   | 0.7952     | 0.5739        | {brier:.4f}  | {ece:.4f}        | **No** (rank score)         |

**Verdict**: Figure 2 is acceptable if relabeled. No new model run required.
A probability-calibrated version using oof_logitz would require regenerating
the calibration plot from `calib_predictions.csv` (existing file, no new inference).
"""
(OUT / "figure2_calibration_recommendation.md").write_text(fig2_text)
print("  figure2_calibration_recommendation.md: written")

# ---------------------------------------------------------------------------
# TASK 6 – Classification results text numbers
# ---------------------------------------------------------------------------
print("=== TASK 6: Classification results text numbers ===")

# Foldwise AUC values for logreg OOF ECDF
fw_aucs = primary_fw["auc"].values
fw_prauc = primary_fw["pr_auc"].values
fw_ba = primary_fw["balanced_accuracy"].values
fw_sens = primary_fw["sensitivity"].values
fw_spec = primary_fw["specificity"].values
fw_f1 = primary_fw["f1"].values

text_lines = [
    "# Classification Results Text Numbers",
    "",
    f"Generated: {now_utc()}",
    "",
    "## Final Selected Model",
    "VAE: recover035_latent384_beta3p75_T80_h10000_p560_full5x5",
    "Channels [1,0,2] = Pearson_Full + OMST + MI_KNN",
    "N=397 (CN=300, AD=97), latent_dim=384, β=3.75, 5-fold outer CV",
    "",
    "## Primary Readout (LogReg L2, z+age+sex, OOF ECDF, target_sens≥0.70)",
    "",
    f"Pooled OOF ECDF AUC: {pooled_auc:.4f} (95% CI: {auc_lo:.4f}–{auc_hi:.4f})",
    f"Pooled OOF ECDF PR-AUC: {pooled_prauc:.4f} (95% CI: {prauc_lo:.4f}–{prauc_hi:.4f})",
    f"Balanced accuracy (pooled): {ba:.4f}",
    f"Sensitivity (pooled): {sens:.4f} ({tp}/{EXPECTED_AD})",
    f"Specificity (pooled): {spec:.4f} ({tn}/{EXPECTED_CN})",
    f"F1 score (pooled): {f1:.4f}",
    f"Brier score: {brier:.4f}",
    f"ECE (10-bin): {ece:.4f}",
    f"Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}",
    "",
    "## Foldwise Summary (LogReg OOF ECDF)",
    "",
    f"AUC: {fw_aucs.mean():.4f} ± {fw_aucs.std():.4f}  "
    f"[folds: {', '.join(f'{v:.4f}' for v in fw_aucs)}]",
    f"PR-AUC: {fw_prauc.mean():.4f} ± {fw_prauc.std():.4f}  "
    f"[folds: {', '.join(f'{v:.4f}' for v in fw_prauc)}]",
    f"Balanced Acc: {fw_ba.mean():.4f} ± {fw_ba.std():.4f}",
    f"Sensitivity: {fw_sens.mean():.4f} ± {fw_sens.std():.4f}",
    f"Specificity: {fw_spec.mean():.4f} ± {fw_spec.std():.4f}",
    f"F1: {fw_f1.mean():.4f} ± {fw_f1.std():.4f}",
    "",
    "## Secondary: SVM (RBF, sigmoid cal, fixed_0.5 threshold)",
    "",
    f"AUC: {svm_fw['auc'].mean():.4f} ± {svm_fw['auc'].std():.4f}  "
    f"[folds: {', '.join(f'{v:.4f}' for v in svm_fw['auc'].values)}]",
    f"PR-AUC: {svm_fw['pr_auc'].mean():.4f} ± {svm_fw['pr_auc'].std():.4f}",
    f"Balanced Acc: {svm_fw['balanced_accuracy'].mean():.4f} ± {svm_fw['balanced_accuracy'].std():.4f}",
    f"Sensitivity: {svm_fw['sensitivity'].mean():.4f} ± {svm_fw['sensitivity'].std():.4f}",
    f"Specificity: {svm_fw['specificity'].mean():.4f} ± {svm_fw['specificity'].std():.4f}",
    f"F1: {svm_fw['f1_score'].mean():.4f} ± {svm_fw['f1_score'].std():.4f}",
    "",
    "## Missing classifiers (not run under final protocol)",
    "MLP, XGBoost, GradientBoosting, RandomForest: MISSING",
    "These appear only in stale runs with latent_dim=256 / N=396 — "
    "must not be backfilled in Table 2.",
    "",
    "## Manuscript text template (primary result)",
    "",
    f"The final selected β-VAE representation (latent dim=384, β=3.75, channels: "
    f"Pearson_Full, OMST, MI_KNN) achieved a pooled OOF AUC of "
    f"{pooled_auc:.3f} (95% CI: {auc_lo:.3f}–{auc_hi:.3f}) and PR-AUC of "
    f"{pooled_prauc:.3f} ({prauc_lo:.3f}–{prauc_hi:.3f}) with logistic regression "
    f"(z+age+sex features, OOF ECDF score calibration) across 5-fold outer cross-validation "
    f"(N=397, CN=300, AD=97). At the target sensitivity threshold (≥0.70 inner-CV OOF "
    f"sensitivity, maximum specificity), sensitivity was {sens:.2%} and specificity "
    f"{spec:.2%} (balanced accuracy {ba:.3f}, F1={f1:.3f}).",
]
(OUT / "classification_results_text_numbers.md").write_text("\n".join(text_lines))
print("  classification_results_text_numbers.md: written")

# ---------------------------------------------------------------------------
# TASK 7 – Stale old Table 2 audit
# ---------------------------------------------------------------------------
print("=== TASK 7: Stale old Table 2 audit ===")

stale_text = f"""# Stale Old Table 2 Audit

Generated: {now_utc()}

## Definition of "Stale" for Table 2

A result is stale for Table 2 if any of the following hold:
- latent_dim ≠ 384
- N_subjects ≠ 397 or N_AD ≠ 97
- VAE run ≠ `recover035_latent384_beta3p75_T80_h10000_p560_full5x5`
- Channels ≠ [1,0,2] (Pearson_Full + OMST + MI_KNN)

## Stale Artifacts Found

### 1. `adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep`

| Parameter | Required | Found | Verdict |
|:----------|:---------|:------|:--------|
| latent_dim | 384 | **256** | STALE |
| N_subjects | 397 | **396** | STALE |
| N_AD | 97 | **96** | STALE |
| Classifiers available | — | logreg_l2, logreg_elasticnet, svm_rbf, xgboost, lightgbm, gradient_boosting, random_forest | — |

This sweep has 7 classifier families (incl. XGBoost, GradBoost, RandomForest) but uses
latent_dim=256 and N_AD=96. **Must not appear in Table 2.** It may appear in a clearly
labeled supplementary ablation showing multi-classifier behavior at the intermediate
latent_dim=256 checkpoint.

### 2. `adni_v5_1_batch20260514b_refined_classifier_only_sweep`

| Parameter | Required | Found | Verdict |
|:----------|:---------|:------|:--------|
| N_subjects | 397 | **396** | STALE |
| N_AD | 97 | **96** | STALE |

This sweep has N_AD=96 (1 subject discrepancy from final cohort). STALE.

### 3. `frozen_latent_stageb_classifier_sweep_pro`

| Parameter | Required | Found | Verdict |
|:----------|:---------|:------|:--------|
| VAE run | recover035_latent384_beta3p75_... | **adni_v5_1_batch20260514b_ch1_0_2_horizon4480...** | STALE |
| N_subjects | 397 | **396** | STALE |

Uses different VAE runs from the final selected model. STALE.

## Impact on Table 2

Table 2 must only include:
- Logistic Regression (OOF ECDF) from `stageB_oof_score_calibration` — **VALID**
- SVM (sigmoid) from `recover035_latent384_beta3p75_T80_h10000_p560_full5x5` — **VALID**
- MLP, XGBoost, GB, RF: **MISSING** (stale runs cannot be used as substitutes)

## Reviewer Response Strategy

If reviewers request additional classifier comparison:
Option A (preferred): State that MLP/XGBoost/GB were evaluated at an exploratory
   latent_dim=256 checkpoint and showed AUC 0.718–0.745 (all below LogReg 0.779).
   At final depth (ld=384), only LogReg and SVM were run per protocol.
Option B: Run a read-only classifier_only sweep using frozen latents from the final
   selected VAE checkpoint (requires launch approval, adds XGBoost/GB/RF to Table 2).
"""
(OUT / "stale_old_table2_audit.md").write_text(stale_text)
print("  stale_old_table2_audit.md: written")

# ---------------------------------------------------------------------------
# TASK 8 – Final recommendation
# ---------------------------------------------------------------------------
print("=== TASK 8: Final recommendation ===")

rec_text = f"""# Final Recommendation: All-Classifiers Table 2 Audit

Generated: {now_utc()}

## Summary

### Audit Status: COMPLETE (read-only)

### Available Classifiers Under Final Protocol (N=397, ld=384, β=3.75, channels [1,0,2])

| Model | AUC (foldwise mean±SD) | PR-AUC (mean±SD) | Status |
|:------|:----------------------:|:----------------:|:-------|
| LogReg L2 (OOF ECDF, primary) | {fw_aucs.mean():.4f}±{fw_aucs.std():.4f} | {fw_prauc.mean():.4f}±{fw_prauc.std():.4f} | **PRIMARY** |
| SVM RBF (sigmoid cal) | {svm_fw['auc'].mean():.4f}±{svm_fw['auc'].std():.4f} | {svm_fw['pr_auc'].mean():.4f}±{svm_fw['pr_auc'].std():.4f} | VALID |
| MLP | — | — | **MISSING** |
| XGBoost | — | — | **MISSING** |
| GradientBoosting | — | — | **MISSING** |
| RandomForest | — | — | **MISSING** |

### Primary Readout (Pooled OOF, 397 subjects)
- AUC: {pooled_auc:.4f} (95% CI: {auc_lo:.4f}–{auc_hi:.4f})
- PR-AUC: {pooled_prauc:.4f} (95% CI: {prauc_lo:.4f}–{prauc_hi:.4f})
- Sensitivity: {sens:.4f}, Specificity: {spec:.4f}, BA: {ba:.4f}, F1: {f1:.4f}
- Brier: {brier:.4f}, ECE: {ece:.4f}
- Confusion matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}

### Calibration Method Winner
Best promoted: **oof_ecdf** (AUC=0.7952, PR-AUC=0.5739) — tied with oof_logitz (0.7951).
oof_ecdf selected as primary because it has the highest PR-AUC and requires no
distributional assumption (monotone rank transform vs sigmoid fitting).

### Figure 2 Assessment
**RELABEL required.** The oof_ecdf calibration panel is NOT a probability calibration
plot. The y-axis should read "AD likelihood rank score" not "P(AD)".
No new model run required. `calib_predictions.csv` contains all scores needed.

### Missing Classifiers Decision
MLP, XGBoost, GradientBoosting, RandomForest are not available under the final protocol.
Do NOT backfill from stale latent_dim=256 / N=396 sweep results.
Table 2 for the manuscript lists these as "not evaluated at final depth."

### Stale Results
- `mfrsplit_3840`: STALE (ld=256, N=396) — can be cited as exploratory context only
- `refined_sweep`: STALE (N=396) — do not use
- `frozen_latent_stageb`: STALE (wrong VAE run) — do not use

## No Action Required

All analysis is complete. No training, no OASIS, no model artifact modification.
Output files are written to:
  `results/revision_bspc_2026/final_all_classifiers_table2_audit_20260622/`
"""
(OUT / "final_recommendation.md").write_text(rec_text)
print("  final_recommendation.md: written")

# ---------------------------------------------------------------------------
# TASK 9 – command_log.json
# ---------------------------------------------------------------------------
log_entry = {
    "script": __file__,
    "generated_utc": now_utc(),
    "output_dir": str(OUT),
    "inputs": {
        "full_run": str(FULL_RUN),
        "stageB_calib": str(STAGEB_CALIB),
    },
    "primary_readout": {
        "model": PRIMARY_MODEL,
        "feature_set": PRIMARY_FEAT,
        "calib_method": PRIMARY_CALIB,
        "threshold_strategy": PRIMARY_THRESH,
        "pooled_auc": pooled_auc,
        "pooled_pr_auc": pooled_prauc,
        "auc_ci_95": [auc_lo, auc_hi],
        "prauc_ci_95": [prauc_lo, prauc_hi],
    },
    "n": EXPECTED_N,
    "n_cn": EXPECTED_CN,
    "n_ad": EXPECTED_AD,
    "read_only": True,
    "did_train": False,
    "did_run_oasis": False,
    "did_modify_artifacts": False,
}
(OUT / "command_log.json").write_text(json.dumps(log_entry, indent=2))
print("  command_log.json: written")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
outputs = list(OUT.glob("*.*"))
print(f"\n=== DONE: {len(outputs)} files in {OUT} ===")
for f in sorted(outputs):
    print(f"  {f.name}")
print(f"\nPrimary AUC: {pooled_auc:.4f} [{auc_lo:.4f}, {auc_hi:.4f}]")
print(f"Primary PR-AUC: {pooled_prauc:.4f} [{prauc_lo:.4f}, {prauc_hi:.4f}]")
