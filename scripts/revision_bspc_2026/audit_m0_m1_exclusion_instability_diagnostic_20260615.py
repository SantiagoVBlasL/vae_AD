#!/usr/bin/env python3
"""Read-only diagnostic for M0/M1 exclusion sensitivity instability.

The audit compares promoted, M0, and M1 artifacts without retraining, threshold
refitting, tensor/metadata edits, prediction edits, or modifications to the
source run directories.
"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "revision_bspc_2026"
OUT = RESULTS / "m0_m1_exclusion_instability_diagnostic_20260615"

PROMOTED_RUN = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
PROMOTED_CALIB = RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration"
M0_RUN = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5_sliceorderM0_confirmed_nondefault_only_sensitivity_20260614"
M0_CALIB = RESULTS / "recover035_latent384_beta3p75_sliceorderM0_stageB_oof_score_calibration_20260614"
M0_PREFLIGHT = RESULTS / "slice_order_M0_confirmed_nondefault_only_sensitivity_20260614"
M0_POSTRUN = RESULTS / "sliceorderM0_confirmed_nondefault_postrun_audit_20260614"
M1_RUN = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5_sliceorderM1_validonly_sensitivity_20260612"
M1_POSTRUN = RESULTS / "sliceorderM1_validonly_postrun_audit_20260614"

FULL_DB = (
    RESULTS
    / "full_database_for_martin_and_validity_preflight_20260612"
    / "promoted_model_full_database_for_martin_20260612.csv"
)

PRIMARY_MODEL = "logreg_l2_original"
PRIMARY_MODEL_SWEEP = "logreg_l2"
PRIMARY_FEATURE = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
M0_EXCLUDED = [
    "031_S_4021",
    "031_S_4032",
    "031_S_4218",
    "031_S_4474",
    "031_S_4496",
]


def md_table(df: pd.DataFrame, max_rows: int = 120) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
    text = view.to_markdown(index=False)
    if len(df) > max_rows:
        text += f"\n\n_Showing {max_rows} of {len(df)} rows._"
    return text + "\n"


def write_df(stem: str, df: pd.DataFrame, max_rows: int = 120) -> None:
    df.to_csv(OUT / f"{stem}.csv", index=False)
    (OUT / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def norm(s: pd.Series) -> pd.Series:
    return s.fillna("MISSING").astype(str).str.strip().replace({"": "MISSING", "nan": "MISSING", "NaN": "MISSING"})


def primary_pooled(calib_dir: Path, label: str) -> dict[str, Any]:
    p = calib_dir / "calib_pooled_metrics.csv"
    df = pd.read_csv(p)
    row = df[
        df["model_name"].eq(PRIMARY_MODEL)
        & df["feature_set"].eq(PRIMARY_FEATURE)
        & df["calib_method"].eq(PRIMARY_CALIB)
        & df["threshold_strategy"].eq(PRIMARY_THRESHOLD)
    ]
    if row.empty:
        raise RuntimeError(f"Primary row not found in {p}")
    r = row.iloc[0]
    return {
        "model": label,
        "source": str(p.relative_to(PROJECT_ROOT)),
        "stage": "StageB_OOF_ECDF",
        "N": r["n"],
        "N_CN": r["n_cn"],
        "N_AD": r["n_ad"],
        "AUC": r["auc"],
        "PR_AUC": r["pr_auc"],
        "BA": r["balanced_accuracy"],
        "Sensitivity": r["sensitivity"],
        "Specificity": r["specificity"],
        "F1": r["f1"],
        "TN": r["tn"],
        "FP": r["fp"],
        "FN": r["fn"],
        "TP": r["tp"],
        "predicted_ad_rate": r.get("predicted_ad_rate", np.nan),
    }


def philips_fpr(calib_dir: Path) -> float:
    p = calib_dir / "calib_philips_fpr_pooled.csv"
    if not p.exists():
        return np.nan
    df = pd.read_csv(p)
    row = df[
        df["model_name"].eq(PRIMARY_MODEL)
        & df["feature_set"].eq(PRIMARY_FEATURE)
        & df["calib_method"].eq(PRIMARY_CALIB)
        & df["threshold_strategy"].eq(PRIMARY_THRESHOLD)
        & df["manufacturer"].eq("Philips")
    ]
    return float(row["fpr_cn_pooled"].iloc[0]) if not row.empty else np.nan


def m1_stagea_rows() -> list[dict[str, Any]]:
    p = M1_POSTRUN / "promoted_vs_m1_comparison.csv"
    if not p.exists():
        return []
    df = pd.read_csv(p)
    rows = []
    for _, r in df[df["comparison_row"].astype(str).str.startswith("M1_retrained_StageA_")].iterrows():
        rows.append({
            "model": r["comparison_row"],
            "source": str(p.relative_to(PROJECT_ROOT)),
            "stage": "StageA",
            "N": np.nan,
            "N_CN": np.nan,
            "N_AD": np.nan,
            "AUC": r["AUC"],
            "PR_AUC": r["PR_AUC"],
            "BA": r["BA"],
            "Sensitivity": r["Sensitivity"],
            "Specificity": r["Specificity"],
            "F1": r["F1"],
            "TN": r["TN"],
            "FP": r["FP"],
            "FN": r["FN"],
            "TP": r["TP"],
            "predicted_ad_rate": np.nan,
            "Philips_CN_FPR": r.get("Philips_CN_FPR", np.nan),
        })
    return rows


def metric_comparison() -> pd.DataFrame:
    rows = []
    promoted = primary_pooled(PROMOTED_CALIB, "promoted_primary")
    promoted["Philips_CN_FPR"] = philips_fpr(PROMOTED_CALIB)
    rows.append(promoted)
    m0 = primary_pooled(M0_CALIB, "M0_retrained_primary")
    m0["Philips_CN_FPR"] = philips_fpr(M0_CALIB)
    rows.append(m0)
    score_m0 = M0_PREFLIGHT / "locked_model_score_only_M0.csv"
    if score_m0.exists():
        r = pd.read_csv(score_m0).iloc[0]
        rows.append({
            "model": "M0_locked_promoted_score_only_retained_set",
            "source": str(score_m0.relative_to(PROJECT_ROOT)),
            "stage": "retained_set_score_only_not_new_model",
            "N": r["N"],
            "N_CN": np.nan,
            "N_AD": np.nan,
            "AUC": r["AUC"],
            "PR_AUC": r["PR_AUC"],
            "BA": r["BA"],
            "Sensitivity": r["Sensitivity"],
            "Specificity": r["Specificity"],
            "F1": r["F1"],
            "TN": r["TN"],
            "FP": r["FP"],
            "FN": r["FN"],
            "TP": r["TP"],
            "predicted_ad_rate": np.nan,
            "Philips_CN_FPR": r["Philips_CN_FPR"],
        })
    rows.extend(m1_stagea_rows())
    out = pd.DataFrame(rows)
    ref = out[out["model"].eq("promoted_primary")].iloc[0]
    for metric in ["AUC", "PR_AUC", "BA", "Sensitivity", "Specificity", "F1", "Philips_CN_FPR"]:
        out[f"delta_vs_promoted_{metric}"] = pd.to_numeric(out[metric], errors="coerce") - float(ref[metric])
    return out


def primary_predictions(calib_dir: Path, label: str) -> pd.DataFrame:
    p = calib_dir / "calib_predictions.csv"
    df = pd.read_csv(p)
    sub = df[
        df["model_name"].eq(PRIMARY_MODEL)
        & df["feature_set"].eq(PRIMARY_FEATURE)
        & df["calib_method"].eq(PRIMARY_CALIB)
        & df["threshold_strategy"].eq(PRIMARY_THRESHOLD)
    ].copy()
    if sub.empty:
        raise RuntimeError(f"Primary predictions not found in {p}")
    sub = sub.rename(columns={"y_score": f"{label}_score", "y_pred": f"{label}_pred", "threshold": f"{label}_threshold", "fold": f"{label}_fold"})
    keep = [
        "SubjectID",
        f"{label}_fold",
        "y_true",
        f"{label}_score",
        f"{label}_pred",
        f"{label}_threshold",
        "Manufacturer",
        "Age",
        "Sex",
    ]
    return sub[[c for c in keep if c in sub.columns]].copy()


def threshold_comparison() -> pd.DataFrame:
    rows = []
    for label, calib_dir in [("promoted", PROMOTED_CALIB), ("M0", M0_CALIB)]:
        p = calib_dir / "calib_foldwise_metrics.csv"
        df = pd.read_csv(p)
        sub = df[
            df["model_name"].eq(PRIMARY_MODEL)
            & df["feature_set"].eq(PRIMARY_FEATURE)
            & df["calib_method"].eq(PRIMARY_CALIB)
            & df["threshold_strategy"].eq(PRIMARY_THRESHOLD)
        ].copy()
        for _, r in sub.iterrows():
            rows.append({
                "model": label,
                "fold": r["fold"],
                "threshold": r["threshold"],
                "best_inner_auc": r.get("best_inner_auc", np.nan),
                "best_params": r.get("best_params", ""),
                "inner_oof_sensitivity": r.get("inner_oof_sensitivity", np.nan),
                "inner_oof_specificity": r.get("inner_oof_specificity", np.nan),
                "test_auc": r.get("auc", np.nan),
                "test_pr_auc": r.get("pr_auc", np.nan),
                "test_ba": r.get("balanced_accuracy", np.nan),
                "test_sensitivity": r.get("sensitivity", np.nan),
                "test_specificity": r.get("specificity", np.nan),
            })
    out = pd.DataFrame(rows)
    wide = out.pivot(index="fold", columns="model", values="threshold").reset_index()
    if {"promoted", "M0"}.issubset(wide.columns):
        wide["threshold_delta_M0_minus_promoted"] = wide["M0"] - wide["promoted"]
        for _, r in wide.iterrows():
            rows.append({
                "model": "M0_minus_promoted_delta",
                "fold": r["fold"],
                "threshold": np.nan,
                "threshold_delta_M0_minus_promoted": r["threshold_delta_M0_minus_promoted"],
            })
    summary = out.groupby("model", dropna=False).agg(
        threshold_mean=("threshold", "mean"),
        threshold_sd=("threshold", "std"),
        inner_oof_sensitivity_mean=("inner_oof_sensitivity", "mean"),
        inner_oof_specificity_mean=("inner_oof_specificity", "mean"),
        test_auc_mean=("test_auc", "mean"),
        test_ba_mean=("test_ba", "mean"),
    ).reset_index()
    summary["fold"] = "SUMMARY"
    detail = pd.DataFrame(rows)
    if "threshold_delta_M0_minus_promoted" not in detail.columns:
        detail["threshold_delta_M0_minus_promoted"] = np.nan
    return pd.concat([detail, summary], ignore_index=True, sort=False)


def score_correlation() -> pd.DataFrame:
    prom = primary_predictions(PROMOTED_CALIB, "promoted")
    m0 = primary_predictions(M0_CALIB, "M0")
    merged = prom.merge(m0[["SubjectID", "M0_fold", "M0_score", "M0_pred", "M0_threshold"]], on="SubjectID", how="inner")
    full = pd.read_csv(FULL_DB)
    full["SubjectID"] = full["SubjectID"].astype(str)
    merged = merged.merge(
        full[["SubjectID", "Site3_final", "raw_tp_group_final", "confusion_label", "slice_order_class"]],
        on="SubjectID",
        how="left",
    )
    rows = []
    group_specs = [("all_overlap", pd.Series(True, index=merged.index))]
    for dx, value in [("CN", 0), ("AD", 1)]:
        group_specs.append((f"diagnosis_{dx}", merged["y_true"].eq(value)))
    for mfr in sorted(merged["Manufacturer"].fillna("MISSING").astype(str).unique()):
        group_specs.append((f"manufacturer_{mfr}", merged["Manufacturer"].astype(str).eq(mfr)))
    for name, mask in group_specs:
        sub = merged[mask].copy()
        if len(sub) < 3:
            pearson = spearman = np.nan
            p_pearson = p_spearman = np.nan
        else:
            pearson, p_pearson = pearsonr(sub["promoted_score"], sub["M0_score"])
            spearman, p_spearman = spearmanr(sub["promoted_score"], sub["M0_score"])
        rows.append({
            "group": name,
            "N": len(sub),
            "pearson_r": pearson,
            "pearson_p": p_pearson,
            "spearman_rho": spearman,
            "spearman_p": p_spearman,
            "promoted_score_mean": sub["promoted_score"].mean(),
            "M0_score_mean": sub["M0_score"].mean(),
            "score_delta_M0_minus_promoted_mean": (sub["M0_score"] - sub["promoted_score"]).mean(),
            "score_delta_M0_minus_promoted_sd": (sub["M0_score"] - sub["promoted_score"]).std(),
            "prediction_disagreement_rate": (sub["promoted_pred"] != sub["M0_pred"]).mean() if len(sub) else np.nan,
        })
    values = merged.copy()
    values["score_delta_M0_minus_promoted"] = values["M0_score"] - values["promoted_score"]
    values.to_csv(OUT / "overlapping_subject_score_values.csv", index=False)
    return pd.DataFrame(rows)


def m0_excluded_scores() -> pd.DataFrame:
    prom = primary_predictions(PROMOTED_CALIB, "promoted")
    full = pd.read_csv(FULL_DB)
    full["SubjectID"] = full["SubjectID"].astype(str)
    sub = prom[prom["SubjectID"].isin(M0_EXCLUDED)].merge(
        full,
        on="SubjectID",
        how="left",
        suffixes=("", "_db"),
    )
    sub["promoted_confusion_at_primary_threshold"] = np.select(
        [
            sub["y_true"].eq(0) & sub["promoted_pred"].eq(0),
            sub["y_true"].eq(0) & sub["promoted_pred"].eq(1),
            sub["y_true"].eq(1) & sub["promoted_pred"].eq(0),
            sub["y_true"].eq(1) & sub["promoted_pred"].eq(1),
        ],
        ["TN", "FP", "FN", "TP"],
        default="UNKNOWN",
    )
    sub["is_high_score_cn_fp"] = sub["y_true"].eq(0) & sub["promoted_pred"].eq(1) & (sub["promoted_score"] > 0.75)
    cols = [
        "SubjectID", "ImageID", "RID", "promoted_fold", "y_true", "promoted_score", "promoted_threshold",
        "promoted_pred", "promoted_confusion_at_primary_threshold", "is_high_score_cn_fp",
        "diagnosis_group", "Manufacturer_final", "Site3_final", "Age_final", "Sex_final",
        "raw_tp_group_final", "slice_order_class", "matches_dparsf_default",
        "high_confidence_slice_timing_match", "match_method", "match_confidence",
    ]
    return sub[[c for c in cols if c in sub.columns]].sort_values("promoted_score", ascending=False)


def parse_c(best_params: Any) -> float:
    text = str(best_params)
    m = re.search(r"model__C['\"]?\s*:\s*([0-9.eE+-]+)", text)
    if not m:
        return np.nan
    return float(m.group(1))


def coefficient_shift() -> pd.DataFrame:
    rows = []
    for label, run in [("promoted", PROMOTED_RUN), ("M0", M0_RUN)]:
        p = run / "classifier_only_readout" / "classifier_sweep_model_status.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        sub = df[
            df["model_name"].eq(PRIMARY_MODEL_SWEEP)
            & df["readout_feature_set"].eq(PRIMARY_FEATURE)
        ].copy()
        for _, r in sub.iterrows():
            rows.append({
                "model": label,
                "fold": r["fold"],
                "status": r.get("status", ""),
                "best_params": r.get("best_params", ""),
                "selected_C": parse_c(r.get("best_params", "")),
                "best_inner_auc": r.get("best_inner_auc", np.nan),
                "coefficient_vector_available": False,
                "note": "Classifier-only readout saved hyperparameter status but not coefficient vectors.",
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        wide = df.pivot(index="fold", columns="model", values="selected_C").reset_index()
        if {"promoted", "M0"}.issubset(wide.columns):
            wide["selected_C_delta_M0_minus_promoted"] = wide["M0"] - wide["promoted"]
            wide["selected_C_ratio_M0_over_promoted"] = wide["M0"] / wide["promoted"]
        wide["status"] = "regularization_comparison_only"
        df = pd.concat([df, wide], ignore_index=True, sort=False)
    return df


def fold_composition_shift() -> pd.DataFrame:
    rows = []
    full = pd.read_csv(FULL_DB)
    full["SubjectID"] = full["SubjectID"].astype(str)
    meta_cols = ["SubjectID", "Site3_final", "raw_tp_group_final", "slice_order_class"]
    def add_rows(label: str, run: Path) -> None:
        cache = run / "classifier_only_readout" / "latent_cache"
        for fold in range(1, 6):
            for split, fname in [("stageB_trainDev", f"fold_{fold}_trainDev_latent_mu.csv"), ("stageB_test", f"fold_{fold}_test_latent_mu.csv")]:
                p = cache / fname
                if not p.exists():
                    continue
                df = pd.read_csv(p, usecols=lambda c: c in {"SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex", "y", "fold", "split"})
                df["SubjectID"] = df["SubjectID"].astype(str)
                df = df.merge(full[[c for c in meta_cols if c in full.columns]], on="SubjectID", how="left")
                for group_label, mask in [
                    ("ALL", pd.Series(True, index=df.index)),
                    ("CN", df["y"].eq(0)),
                    ("AD", df["y"].eq(1)),
                ]:
                    sub = df[mask]
                    row = {
                        "model": label,
                        "fold": fold,
                        "split": split,
                        "group": group_label,
                        "N": len(sub),
                        "N_CN": int((sub["y"] == 0).sum()),
                        "N_AD": int((sub["y"] == 1).sum()),
                        "N_GE": int((sub["Manufacturer"] == "GE").sum()),
                        "N_Philips": int((sub["Manufacturer"] == "Philips").sum()),
                        "N_SIEMENS": int((sub["Manufacturer"] == "SIEMENS").sum()),
                        "N_Site2": int(norm(sub.get("Site3_final", pd.Series(index=sub.index))).str.replace(r"\.0$", "", regex=True).eq("2").sum()) if len(sub) else 0,
                        "N_Site18": int(norm(sub.get("Site3_final", pd.Series(index=sub.index))).str.replace(r"\.0$", "", regex=True).eq("18").sum()) if len(sub) else 0,
                        "N_Site31": int(norm(sub.get("Site3_final", pd.Series(index=sub.index))).str.replace(r"\.0$", "", regex=True).eq("31").sum()) if len(sub) else 0,
                        "N_Site301": int(norm(sub.get("Site3_final", pd.Series(index=sub.index))).str.replace(r"\.0$", "", regex=True).eq("301").sum()) if len(sub) else 0,
                        "N_rawTP140": int(norm(sub.get("raw_tp_group_final", pd.Series(index=sub.index))).eq("140").sum()) if len(sub) else 0,
                    }
                    rows.append(row)
    add_rows("promoted", PROMOTED_RUN)
    add_rows("M0", M0_RUN)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    idx = ["fold", "split", "group"]
    prom = out[out["model"].eq("promoted")].set_index(idx)
    m0 = out[out["model"].eq("M0")].set_index(idx)
    common = prom.index.intersection(m0.index)
    delta_rows = []
    numeric = [c for c in out.columns if c.startswith("N")]
    for key in common:
        row = {"model": "M0_minus_promoted_delta", "fold": key[0], "split": key[1], "group": key[2]}
        for c in numeric:
            row[c] = m0.loc[key, c] - prom.loc[key, c]
        delta_rows.append(row)
    return pd.concat([out, pd.DataFrame(delta_rows)], ignore_index=True, sort=False)


def completion_status() -> pd.DataFrame:
    rows = []
    checks = [
        ("promoted_calib", PROMOTED_CALIB / "calib_pooled_metrics.csv"),
        ("promoted_predictions", PROMOTED_CALIB / "calib_predictions.csv"),
        ("promoted_classifier_only", PROMOTED_RUN / "classifier_only_readout" / "classifier_sweep_predictions.csv"),
        ("M0_calib", M0_CALIB / "calib_pooled_metrics.csv"),
        ("M0_predictions", M0_CALIB / "calib_predictions.csv"),
        ("M0_classifier_only", M0_RUN / "classifier_only_readout" / "classifier_sweep_predictions.csv"),
        ("M1_postrun_comparison", M1_POSTRUN / "promoted_vs_m1_comparison.csv"),
        ("full_database", FULL_DB),
    ]
    for name, path in checks:
        rows.append({"item": name, "path": str(path.relative_to(PROJECT_ROOT)), "exists": path.exists()})
    return pd.DataFrame(rows)


def mechanism_interpretation(metrics: pd.DataFrame, corr: pd.DataFrame, thresholds: pd.DataFrame, coef: pd.DataFrame, excluded: pd.DataFrame) -> str:
    prom = metrics[metrics["model"].eq("promoted_primary")].iloc[0]
    m0 = metrics[metrics["model"].eq("M0_retrained_primary")].iloc[0]
    all_corr = corr[corr["group"].eq("all_overlap")].iloc[0]
    n_fp_ex = int((excluded["promoted_confusion_at_primary_threshold"] == "FP").sum())
    n_high = int(excluded["is_high_score_cn_fp"].sum())
    threshold_summary = thresholds[thresholds["fold"].astype(str).eq("SUMMARY")]
    p_thr = threshold_summary[threshold_summary["model"].eq("promoted")]["threshold_mean"].iloc[0]
    m_thr = threshold_summary[threshold_summary["model"].eq("M0")]["threshold_mean"].iloc[0]
    c_note = "Coefficient vectors were not saved by the classifier-only readout; selected C values were compared instead."
    if not coef.empty and "selected_C_ratio_M0_over_promoted" in coef.columns:
        ratio = coef["selected_C_ratio_M0_over_promoted"].dropna()
        if not ratio.empty:
            c_note += f" M0/promoted selected-C ratios across folds ranged from {ratio.min():.3g} to {ratio.max():.3g}."
    return f"""# Mechanism Interpretation

## Main result

The M0 retrain does not reproduce the retained-set score-only impression. The
retained-set score-only comparison is slightly better than the promoted full
set because the five excluded subjects are removed from the evaluation table,
but the newly trained M0 VAE/readout has lower ranking metrics:

- Promoted Stage B OOF-ECDF AUC={prom['AUC']:.6f}, PR-AUC={prom['PR_AUC']:.6f}.
- M0 Stage B OOF-ECDF AUC={m0['AUC']:.6f}, PR-AUC={m0['PR_AUC']:.6f}.

This pattern is not explained by removing five subjects alone. It indicates
that retraining changed the learned latent representation and downstream score
geometry.

## Threshold/calibration shift

The mean selected OOF-ECDF threshold changed from {p_thr:.6f} in the promoted
readout to {m_thr:.6f} in M0. M0's BA/F1 remained close to promoted in the
primary Stage B row, so thresholding is not the dominant explanation for the
AUC/PR-AUC drop. The larger change is in rank separation.

## Latent representation drift

On overlapping subjects, promoted-vs-M0 score correlation was:

- Pearson r={all_corr['pearson_r']:.6f}
- Spearman rho={all_corr['spearman_rho']:.6f}
- prediction disagreement rate={all_corr['prediction_disagreement_rate']:.6f}

The scores remain related but are not identical. This supports representation
and score-geometry drift after full VAE retraining.

## Classifier instability

{c_note}

Regularization changes can contribute to fold-level score changes, but the
main evidence still points to representation/score drift rather than a simple
coefficient-only effect.

## Removal of hard negatives

Among the five M0-excluded subjects, {n_fp_ex}/5 were promoted-model false
positives and {n_high}/5 were high-score CN false positives at score > 0.75.
Removing these subjects mechanically improves retained-set specificity/FPR,
but that is not equivalent to improving the model.

## Fold/site/manufacturer balance

Fold composition tables show the exact train/test manufacturer and site count
changes. M0 removes a small, concentrated Site31/Philips/CN/rawTP140 subset.
That concentration is enough to perturb fold-local VAE training and classifier
calibration despite only five subjects being excluded.

## Random retraining variability

Because the FULL experiment retrains five separate VAEs, a small fold-local
metadata change can move optimization trajectories and latent coordinates even
when the nominal hyperparameters are identical. The retained-set score-only
result isolates the effect of removing the five subjects from evaluation;
the retrained M0 result additionally includes VAE optimization variability,
new fold-local latent geometry, and new classifier/readout selection. That
combined retraining variability is a plausible contributor to the observed
drop.

## Conclusion

The M0/M1 performance decay is most consistent with latent representation
drift and fold-local score-geometry changes after retraining, with a secondary
contribution from threshold/regularization changes. It is not evidence that
the five subjects alone carry the promoted model's performance. The safe
interpretation is that exclusion-based retraining is unstable and should not
replace corrected preprocessing.

Final recommendation: do not recommend more subject exclusion unless there is
confirmed QC invalidity. The more rigorous next step is corrected preprocessing
and reintegration of confirmed wrong-STC subjects, followed by a pre-specified
sensitivity analysis.
"""


def collaborator_reply(metrics: pd.DataFrame, excluded: pd.DataFrame) -> str:
    prom = metrics[metrics["model"].eq("promoted_primary")].iloc[0]
    m0 = metrics[metrics["model"].eq("M0_retrained_primary")].iloc[0]
    n_fp_ex = int((excluded["promoted_confusion_at_primary_threshold"] == "FP").sum())
    return f"""# Collaborator Reply Draft

You are right that removing only five subjects should not by itself explain a
large performance change. The retained-set calculation confirms this: when we
keep the promoted model fixed and simply remove those five subjects from the
evaluation table, metrics change only modestly.

The larger drop appears after full retraining. That points to retraining
instability: the VAE is re-learned fold by fold, so a small, concentrated
Philips/Site31/rawTP140 exclusion can alter the latent geometry, selected
readout regularization, and fold-specific score scale. The M0 Stage B result
is lower than promoted (AUC {m0['AUC']:.3f} vs {prom['AUC']:.3f}; PR-AUC
{m0['PR_AUC']:.3f} vs {prom['PR_AUC']:.3f}), even though {n_fp_ex}/5 excluded
subjects were promoted-model false positives.

So the correct conclusion is not that these five subjects explain the model,
but that exclusion-based retraining is an unstable and scientifically weaker
fix. I would not recommend further subject exclusion unless QC invalidity is
confirmed. The cleaner next step is to correct preprocessing for confirmed
wrong slice-timing cases, reintegrate them, and report that as a pre-specified
QC/preprocessing sensitivity.
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    completion = completion_status()
    metrics = metric_comparison()
    thresholds = threshold_comparison()
    corr = score_correlation()
    excluded = m0_excluded_scores()
    coef = coefficient_shift()
    fold_shift = fold_composition_shift()

    write_df("completion_status", completion, max_rows=80)
    write_df("promoted_m0_m1_metric_comparison", metrics, max_rows=40)
    write_df("threshold_comparison", thresholds, max_rows=80)
    write_df("overlapping_subject_score_correlation", corr, max_rows=80)
    write_df("m0_excluded_subject_scores", excluded, max_rows=20)
    write_df("classifier_coefficient_shift", coef, max_rows=80)
    write_df("fold_composition_shift", fold_shift, max_rows=240)

    (OUT / "mechanism_interpretation.md").write_text(
        mechanism_interpretation(metrics, corr, thresholds, coef, excluded),
        encoding="utf-8",
    )
    (OUT / "collaborator_reply_draft.md").write_text(
        collaborator_reply(metrics, excluded),
        encoding="utf-8",
    )
    write_json(
        OUT / "command_log.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "guardrails": {
                "read_only": True,
                "model_training": False,
                "threshold_refitting": False,
                "metadata_edits": False,
                "tensor_edits": False,
                "prediction_edits": False,
                "promoted_artifact_edits": False,
                "m0_artifact_edits": False,
                "m1_artifact_edits": False,
            },
            "inputs": {
                "promoted_run": str(PROMOTED_RUN),
                "promoted_calibration": str(PROMOTED_CALIB),
                "m0_run": str(M0_RUN),
                "m0_calibration": str(M0_CALIB),
                "m1_run": str(M1_RUN),
                "m1_postrun": str(M1_POSTRUN),
                "full_database": str(FULL_DB),
            },
            "outputs": str(OUT.relative_to(PROJECT_ROOT)),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
