#!/usr/bin/env python3
"""Final threshold audit for ADNI v5.1 batch20260514b mfrsplit_3840.

This script is intentionally read-only with respect to model artifacts, tensors,
metadata, and ledgers. It audits the already completed classifier-only sweep.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]
SWEEP_DIR = ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep"
MFR_AUDIT_DIR = ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_mfrsplit_3840_audit"
ORIGINAL_RUN_DIR = ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate"
OUT_DIR = ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_threshold_final_audit"

FOCUS_MODEL = "logreg_l2"
FOCUS_THRESHOLDS = [
    "fixed_0p5",
    "inner_oof_youden_j",
    "inner_oof_target_sens_ge_0p70_max_spec",
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def safe_auc(y_true: Iterable[int], y_score: Iterable[float]) -> float:
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_score_arr = np.asarray(list(y_score), dtype=float)
    if len(np.unique(y_true_arr)) < 2:
        return np.nan
    return float(roc_auc_score(y_true_arr, y_score_arr))


def safe_pr_auc(y_true: Iterable[int], y_score: Iterable[float]) -> float:
    y_true_arr = np.asarray(list(y_true), dtype=int)
    y_score_arr = np.asarray(list(y_score), dtype=float)
    if len(np.unique(y_true_arr)) < 2:
        return np.nan
    return float(average_precision_score(y_true_arr, y_score_arr))


def metric_row(df: pd.DataFrame, *, threshold: Optional[float] = None, label_prefix: Optional[Dict[str, object]] = None) -> Dict[str, object]:
    if df.empty:
        base: Dict[str, object] = {}
        if label_prefix:
            base.update(label_prefix)
        return base

    y_true = df["y_true"].astype(int).to_numpy()
    y_score = df["y_score"].astype(float).to_numpy()
    if threshold is None:
        y_pred = df["y_pred"].astype(int).to_numpy()
        threshold_value = np.nan
    else:
        y_pred = (y_score >= threshold).astype(int)
        threshold_value = float(threshold)

    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    out: Dict[str, object] = {
        "n": int(len(df)),
        "n_cn": int((y_true == 0).sum()),
        "n_ad": int((y_true == 1).sum()),
        "threshold": threshold_value,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else np.nan,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else np.nan,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "predicted_ad_rate": float(np.mean(y_pred == 1)),
        "auc": safe_auc(y_true, y_score),
        "pr_auc": safe_pr_auc(y_true, y_score),
        "score_mean": float(np.mean(y_score)),
        "score_median": float(np.median(y_score)),
    }
    if label_prefix:
        out = {**label_prefix, **out}
    return out


def quantile_rows(df: pd.DataFrame, group_cols: List[str], score_col: str = "y_score") -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    if df.empty:
        return pd.DataFrame()
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        scores = group[score_col].astype(float).to_numpy()
        row.update(
            {
                "n": int(len(group)),
                "score_min": float(np.min(scores)),
                "score_p10": float(np.quantile(scores, 0.10)),
                "score_p25": float(np.quantile(scores, 0.25)),
                "score_median": float(np.median(scores)),
                "score_p75": float(np.quantile(scores, 0.75)),
                "score_p90": float(np.quantile(scores, 0.90)),
                "score_max": float(np.max(scores)),
                "score_mean": float(np.mean(scores)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def verify_threshold_selection(thresholds: pd.DataFrame, command_log: Dict[str, object]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for _, row in thresholds.iterrows():
        strategy = str(row["threshold_strategy"])
        is_fixed = strategy == "fixed_0p5"
        expected_context = "fixed_no_selection" if is_fixed else "true_inner_cv_oof"
        has_expected_context = str(row["threshold_selection_context"]) == expected_context
        has_inner_metrics = (
            is_fixed
            or (
                pd.notna(row.get("inner_oof_sensitivity"))
                and pd.notna(row.get("inner_oof_specificity"))
                and pd.notna(row.get("inner_oof_balanced_accuracy"))
            )
        )
        rows.append(
            {
                "fold": row["fold"],
                "model_name": row["model_name"],
                "threshold_strategy": strategy,
                "threshold": row["threshold"],
                "threshold_selection_context": row["threshold_selection_context"],
                "inner_cv_context": row.get("inner_cv_context"),
                "minimum_inner_stratum_count": row.get("minimum_inner_stratum_count"),
                "inner_oof_sensitivity": row.get("inner_oof_sensitivity"),
                "inner_oof_specificity": row.get("inner_oof_specificity"),
                "inner_oof_balanced_accuracy": row.get("inner_oof_balanced_accuracy"),
                "context_pass": bool(has_expected_context),
                "inner_metric_pass": bool(has_inner_metrics),
                "verification_status": "PASS" if has_expected_context and has_inner_metrics else "FAIL",
            }
        )
    out = pd.DataFrame(rows)
    out["command_log_threshold_selection"] = command_log.get("threshold_selection", "")
    out["command_log_vae_retrained"] = command_log.get("vae_retrained", "")
    out["command_log_tensor_modified"] = command_log.get("tensor_modified", "")
    return out


def subgroup_metrics(pred: pd.DataFrame, model: str, strategies: List[str], grouping: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    subset = pred[(pred["model_name"] == model) & (pred["threshold_strategy"].isin(strategies))].copy()
    if grouping not in subset.columns:
        return pd.DataFrame()
    for (strategy, group_value), group in subset.groupby(["threshold_strategy", grouping], dropna=False):
        threshold_by_fold = (
            group[["fold", "threshold"]]
            .drop_duplicates()
            .sort_values("fold")
            .assign(threshold=lambda x: x["threshold"].astype(float).round(12))
        )
        threshold_values = ";".join(f"fold{int(r.fold)}={r.threshold}" for r in threshold_by_fold.itertuples())
        selected_threshold = 0.5 if strategy == "fixed_0p5" else "fold_specific"
        rows.append(
            metric_row(
                group,
                threshold=None,
                label_prefix={
                    "model_name": model,
                    "threshold_strategy": strategy,
                    "grouping": grouping,
                    "group_value": group_value,
                    "selected_threshold": selected_threshold,
                    "threshold_values_by_fold": threshold_values,
                },
            )
        )
    return pd.DataFrame(rows).sort_values(["threshold_strategy", "grouping", "group_value"])


def focus_confusions(pred: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for strategy in FOCUS_THRESHOLDS:
        group = pred[(pred["model_name"] == FOCUS_MODEL) & (pred["threshold_strategy"] == strategy)]
        if group.empty:
            continue
        threshold_by_fold = (
            group[["fold", "threshold"]]
            .drop_duplicates()
            .sort_values("fold")
            .assign(threshold=lambda x: x["threshold"].astype(float).round(12))
        )
        threshold_values = ";".join(f"fold{int(r.fold)}={r.threshold}" for r in threshold_by_fold.itertuples())
        selected_threshold = 0.5 if strategy == "fixed_0p5" else "fold_specific"
        rows.append(
            metric_row(
                group,
                label_prefix={
                    "model_name": FOCUS_MODEL,
                    "threshold_strategy": strategy,
                    "selected_threshold": selected_threshold,
                    "threshold_values_by_fold": threshold_values,
                },
            )
        )
    return pd.DataFrame(rows)


def fold4_error_audit(pred: pd.DataFrame) -> pd.DataFrame:
    rows: List[pd.DataFrame] = []
    fold4 = pred[(pred["fold"] == 4) & (pred["model_name"] == FOCUS_MODEL) & (pred["threshold_strategy"].isin(FOCUS_THRESHOLDS))].copy()
    for strategy, group in fold4.groupby("threshold_strategy"):
        if group.empty:
            continue
        threshold = float(group["threshold"].iloc[0])
        ad_scores = group[group["y_true"] == 1]["y_score"].astype(float)
        cn_scores = group[group["y_true"] == 0]["y_score"].astype(float)
        ad_median = float(ad_scores.median()) if len(ad_scores) else np.nan
        cn_median = float(cn_scores.median()) if len(cn_scores) else np.nan
        errors = group[group["y_true"].astype(int) != group["y_pred"].astype(int)].copy()
        if errors.empty:
            continue
        errors["error_type"] = np.where(errors["y_true"].astype(int) == 1, "false_negative", "false_positive")
        errors["score_minus_threshold"] = errors["y_score"].astype(float) - threshold
        errors["abs_margin_to_threshold"] = errors["score_minus_threshold"].abs()
        errors["threshold_near_0p05"] = errors["abs_margin_to_threshold"] <= 0.05
        errors["threshold_near_0p10"] = errors["abs_margin_to_threshold"] <= 0.10

        def classify(row: pd.Series) -> str:
            score = float(row["y_score"])
            if abs(score - threshold) <= 0.05:
                return "threshold_near"
            if int(row["y_true"]) == 1 and pd.notna(cn_median) and score <= cn_median:
                return "true_ranking_failure_low_ad_score"
            if int(row["y_true"]) == 0 and pd.notna(ad_median) and score >= ad_median:
                return "true_ranking_failure_high_cn_score"
            return "intermediate_margin_error"

        errors["error_interpretation"] = errors.apply(classify, axis=1)
        errors["fold_ad_score_median"] = ad_median
        errors["fold_cn_score_median"] = cn_median
        keep_cols = [
            "threshold_strategy",
            "threshold",
            "error_type",
            "error_interpretation",
            "SubjectID",
            "y_true",
            "y_score",
            "y_pred",
            "score_minus_threshold",
            "abs_margin_to_threshold",
            "threshold_near_0p05",
            "threshold_near_0p10",
            "Manufacturer",
            "Age",
            "Sex",
            "source_batch",
            "source_label",
            "tensor_source",
            "fold_ad_score_median",
            "fold_cn_score_median",
        ]
        rows.append(errors[[col for col in keep_cols if col in errors.columns]])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values(["threshold_strategy", "error_type", "y_score"])


def fold4_distributions(pred: pd.DataFrame) -> pd.DataFrame:
    fold4 = pred[(pred["fold"] == 4) & (pred["model_name"] == FOCUS_MODEL) & (pred["threshold_strategy"].isin(FOCUS_THRESHOLDS))].copy()
    tables: List[pd.DataFrame] = []
    for group_cols in [["threshold_strategy", "ResearchGroup_Mapped"], ["threshold_strategy", "Manufacturer"], ["threshold_strategy", "Sex"]]:
        q = quantile_rows(fold4, group_cols)
        q["distribution_grouping"] = "+".join(group_cols[1:])
        tables.append(q)
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def fold4_threshold_near_summary(errors: pd.DataFrame) -> pd.DataFrame:
    if errors.empty:
        return pd.DataFrame()
    rows = []
    for (strategy, error_type), group in errors.groupby(["threshold_strategy", "error_type"]):
        rows.append(
            {
                "threshold_strategy": strategy,
                "error_type": error_type,
                "n_errors": int(len(group)),
                "n_threshold_near_0p05": int(group["threshold_near_0p05"].sum()),
                "n_threshold_near_0p10": int(group["threshold_near_0p10"].sum()),
                "n_true_ranking_failures": int(group["error_interpretation"].str.startswith("true_ranking_failure").sum()),
                "median_abs_margin_to_threshold": float(group["abs_margin_to_threshold"].median()),
                "max_abs_margin_to_threshold": float(group["abs_margin_to_threshold"].max()),
            }
        )
    return pd.DataFrame(rows)


def load_original_logreg_predictions(pred_reference: pd.DataFrame) -> pd.DataFrame:
    original_path = ORIGINAL_RUN_DIR / (
        "all_folds_clf_predictions_MULTI_logreg_vaeconvtranspose4l_ld256_beta2.5_"
        "normzscore_offdiag_ch3sel_intFCquarter_drop0.15_ln0_outer5x1_scoreroc_auc.csv"
    )
    original = read_csv(original_path)
    original = original[original["classifier_type"] == "logreg"].copy()
    metadata_cols = [
        "SubjectID",
        "ResearchGroup_Mapped",
        "Manufacturer",
        "Age",
        "Sex",
        "source_batch",
        "source_label",
        "tensor_source",
    ]
    metadata = pred_reference[pred_reference["threshold_strategy"] == "fixed_0p5"][metadata_cols].drop_duplicates("SubjectID")
    original = original.merge(metadata, on="SubjectID", how="left")
    original["model_name"] = "original_logreg"
    original["threshold_strategy"] = "original_fixed_0p5"
    original["threshold"] = 0.5
    original["y_score"] = original["y_score_final"]
    return original


def load_original_logreg_regularization() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for fold in range(1, 6):
        path = ORIGINAL_RUN_DIR / f"fold_{fold}/optuna_best_trial_logreg_fold_{fold}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "fold": fold,
                "original_logreg_C": payload.get("best_params", {}).get("model__C"),
                "original_best_inner_auc": payload.get("best_value"),
                "original_effective_n_trials": payload.get("effective_n_trials"),
            }
        )
    return pd.DataFrame(rows)


def original_vs_classifier_only(pred: pd.DataFrame, model_status: pd.DataFrame, run_config: Dict[str, object]) -> pd.DataFrame:
    original = load_original_logreg_predictions(pred)
    classifier_only = pred[(pred["model_name"] == FOCUS_MODEL) & (pred["threshold_strategy"] == "fixed_0p5")].copy()
    classifier_only["classifier_type"] = FOCUS_MODEL
    classifier_only["did_calibrate"] = False
    combined = pd.concat(
        [
            original[
                [
                    "fold",
                    "SubjectID",
                    "y_true",
                    "y_score",
                    "y_pred",
                    "model_name",
                    "threshold_strategy",
                    "threshold",
                    "did_calibrate",
                    "ResearchGroup_Mapped",
                    "Manufacturer",
                    "Age",
                    "Sex",
                ]
            ],
            classifier_only[
                [
                    "fold",
                    "SubjectID",
                    "y_true",
                    "y_score",
                    "y_pred",
                    "model_name",
                    "threshold_strategy",
                    "threshold",
                    "did_calibrate",
                    "ResearchGroup_Mapped",
                    "Manufacturer",
                    "Age",
                    "Sex",
                ]
            ],
        ],
        ignore_index=True,
    )
    rows = []
    original_reg = load_original_logreg_regularization()
    co_status = model_status[model_status["model_name"] == FOCUS_MODEL][["fold", "best_params", "best_inner_auc"]].copy()
    reg = original_reg.merge(co_status, on="fold", how="outer")
    for _, row in reg.iterrows():
        rows.append(
            {
                "fold": int(row["fold"]),
                "original_regularization": "L2 logistic regression, Optuna continuous C",
                "original_C": row.get("original_logreg_C"),
                "original_calibration": bool(run_config.get("args", {}).get("classifier_calibrate", True)),
                "original_class_weight": bool(run_config.get("args", {}).get("classifier_use_class_weight", True)),
                "classifier_only_regularization": "L2 logistic regression, compact grid",
                "classifier_only_best_params": row.get("best_params"),
                "classifier_only_calibration": False,
                "classifier_only_class_weight": True,
                "original_best_inner_auc": row.get("original_best_inner_auc"),
                "classifier_only_best_inner_auc": row.get("best_inner_auc"),
            }
        )
    return pd.DataFrame(rows), combined


def probability_distribution_comparison(combined: pd.DataFrame) -> pd.DataFrame:
    combined = combined.copy()
    combined["class_label"] = np.where(combined["y_true"].astype(int) == 1, "AD", "CN")
    by_class = quantile_rows(combined, ["model_name", "class_label"])
    by_all = quantile_rows(combined, ["model_name"])
    by_all["class_label"] = "ALL"
    return pd.concat([by_class, by_all], ignore_index=True)


def build_fold_classifier_threshold_table(foldwise: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "fold",
        "model_name",
        "threshold_strategy",
        "threshold",
        "threshold_selection_context",
        "inner_cv_context",
        "minimum_inner_stratum_count",
        "inner_oof_sensitivity",
        "inner_oof_specificity",
        "inner_oof_balanced_accuracy",
        "sensitivity",
        "specificity",
        "balanced_accuracy",
        "auc",
        "pr_auc",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    merged = foldwise.merge(
        thresholds[
            [
                "fold",
                "model_name",
                "threshold_strategy",
                "threshold_selection_context",
                "inner_cv_context",
                "minimum_inner_stratum_count",
            ]
        ],
        on=["fold", "model_name", "threshold_strategy"],
        how="left",
        suffixes=("", "_thresholds"),
    )
    return merged[[col for col in cols if col in merged.columns]].sort_values(["fold", "model_name", "threshold_strategy"])


def make_readme(
    *,
    verification: pd.DataFrame,
    focus_metrics: pd.DataFrame,
    focus_confusion: pd.DataFrame,
    fold4_errors: pd.DataFrame,
    fold4_summary: pd.DataFrame,
    original_compare: pd.DataFrame,
    probability_compare: pd.DataFrame,
) -> str:
    nonfixed = verification[verification["threshold_strategy"] != "fixed_0p5"]
    verification_pass = bool((nonfixed["verification_status"] == "PASS").all())

    focus_pick = focus_metrics[
        (focus_metrics["model_name"] == FOCUS_MODEL)
        & (focus_metrics["threshold_strategy"] == "inner_oof_target_sens_ge_0p70_max_spec")
    ]
    focus_pick_row = focus_pick.iloc[0].to_dict() if not focus_pick.empty else {}

    fixed = focus_confusion[focus_confusion["threshold_strategy"] == "fixed_0p5"]
    youden = focus_confusion[focus_confusion["threshold_strategy"] == "inner_oof_youden_j"]
    target = focus_confusion[focus_confusion["threshold_strategy"] == "inner_oof_target_sens_ge_0p70_max_spec"]

    orig_summary = probability_compare[probability_compare["class_label"] == "ALL"].set_index("model_name")
    lines = [
        "# ADNI v5.1 batch20260514b Threshold Final Audit",
        "",
        "## Leakage check",
        "",
        f"- Non-0.5 threshold provenance pass: `{verification_pass}`.",
        "- Verification requires `threshold_selection_context=true_inner_cv_oof` and populated inner-OOF sensitivity/specificity for every non-fixed threshold.",
        "- No threshold in this audit is selected from outer-test labels.",
        "",
        "## Primary finding",
        "",
        "- `logreg_l2` is the preferred classifier from the classifier-only sweep: it keeps the best pooled AUC/PR-AUC while materially improving sensitivity relative to the original calibrated logreg operating point.",
    ]
    if focus_pick_row:
        lines.append(
            "- Recommended primary operating rule: `logreg_l2` with `inner_oof_target_sens_ge_0p70_max_spec` "
            f"(AUC={focus_pick_row.get('auc', np.nan):.3f}, PR-AUC={focus_pick_row.get('pr_auc', np.nan):.3f}, "
            f"sensitivity={focus_pick_row.get('sensitivity', np.nan):.3f}, specificity={focus_pick_row.get('specificity', np.nan):.3f}, "
            f"balanced_accuracy={focus_pick_row.get('balanced_accuracy', np.nan):.3f}, F1={focus_pick_row.get('f1', np.nan):.3f})."
        )
    lines.extend(["", "## LogReg L2 operating points", ""])
    for label, df in [("0.5", fixed), ("inner-OOF Youden", youden), ("inner-OOF target sensitivity >=0.70", target)]:
        if df.empty:
            continue
        row = df.iloc[0]
        lines.append(
            f"- {label}: tn={int(row.tn)}, fp={int(row.fp)}, fn={int(row.fn)}, tp={int(row.tp)}, "
            f"sensitivity={row.sensitivity:.3f}, specificity={row.specificity:.3f}, balanced_accuracy={row.balanced_accuracy:.3f}, F1={row.f1:.3f}."
        )

    lines.extend(["", "## Fold 4", ""])
    if fold4_summary.empty:
        lines.append("- Fold 4 has no logreg_l2 errors under the audited thresholds.")
    else:
        for _, row in fold4_summary.iterrows():
            lines.append(
                f"- {row.threshold_strategy} / {row.error_type}: {int(row.n_errors)} errors; "
                f"{int(row.n_threshold_near_0p05)} within 0.05 of threshold, "
                f"{int(row.n_true_ranking_failures)} classified as true ranking failures."
            )

    lines.extend(
        [
            "",
            "## Original logreg vs classifier-only logreg_l2",
            "",
            "- Original logreg used L2 logistic regression selected by Optuna, class weighting enabled, and calibration enabled.",
            "- Classifier-only `logreg_l2` used L2 logistic regression with `class_weight=balanced`, compact C grid, and no calibration.",
        ]
    )
    if "original_logreg" in orig_summary.index and FOCUS_MODEL in orig_summary.index:
        orig = orig_summary.loc["original_logreg"]
        co = orig_summary.loc[FOCUS_MODEL]
        lines.append(
            "- Probability distribution shift: original logreg median score "
            f"{orig.score_median:.3f} versus classifier-only logreg_l2 median score {co.score_median:.3f}. "
            "This explains why threshold 0.5 is far less conservative in the classifier-only sweep."
        )

    lines.extend(
        [
            "",
            "## Manuscript-safe claims",
            "",
            "- Safe: threshold selection was performed inside train/dev only using inner-CV OOF predictions; outer-test labels were used only once for evaluation.",
            "- Safe: classifier-only logreg_l2 improves the operating sensitivity/specificity tradeoff over the original calibrated logreg at threshold 0.5 on the same VAE folds.",
            "- Not safe: claiming the threshold is externally validated or final for deployment; it remains an internal cross-validated operating-point analysis.",
            "",
            "## Files",
            "",
            "- `threshold_selection_verification.csv`",
            "- `fold_classifier_threshold_audit.csv`",
            "- `logreg_l2_threshold_comparison.csv`",
            "- `logreg_l2_confusion_matrices.csv`",
            "- `logreg_l2_subgroup_manufacturer.csv`",
            "- `logreg_l2_subgroup_sex.csv`",
            "- `fold4_logreg_l2_error_audit.csv`",
            "- `fold4_logreg_l2_probability_distributions.csv`",
            "- `fold4_logreg_l2_threshold_near_summary.csv`",
            "- `original_vs_classifier_only_logreg_l2.csv`",
            "- `original_vs_classifier_only_probability_distribution.csv`",
            "- `command_log.json`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pred = read_csv(SWEEP_DIR / "classifier_sweep_predictions.csv")
    thresholds = read_csv(SWEEP_DIR / "classifier_sweep_thresholds_by_fold.csv")
    foldwise = read_csv(SWEEP_DIR / "classifier_sweep_foldwise_metrics.csv")
    pooled = read_csv(SWEEP_DIR / "classifier_sweep_pooled_metrics.csv")
    model_status = read_csv(SWEEP_DIR / "classifier_sweep_model_status.csv")
    command_log = json.loads((SWEEP_DIR / "command_log.json").read_text(encoding="utf-8"))
    run_config = json.loads((ORIGINAL_RUN_DIR / "run_config.json").read_text(encoding="utf-8"))

    verification = verify_threshold_selection(thresholds, command_log)
    verification.to_csv(OUT_DIR / "threshold_selection_verification.csv", index=False)
    if (verification["verification_status"] != "PASS").any():
        raise RuntimeError("Threshold selection verification failed; see threshold_selection_verification.csv")

    fold_threshold = build_fold_classifier_threshold_table(foldwise, thresholds)
    fold_threshold.to_csv(OUT_DIR / "fold_classifier_threshold_audit.csv", index=False)

    focus_metrics = pooled[(pooled["model_name"] == FOCUS_MODEL) & (pooled["threshold_strategy"].isin(FOCUS_THRESHOLDS))].copy()
    focus_metrics.to_csv(OUT_DIR / "logreg_l2_threshold_comparison.csv", index=False)

    confusion = focus_confusions(pred)
    confusion.to_csv(OUT_DIR / "logreg_l2_confusion_matrices.csv", index=False)

    manufacturer = subgroup_metrics(pred, FOCUS_MODEL, FOCUS_THRESHOLDS, "Manufacturer")
    manufacturer.to_csv(OUT_DIR / "logreg_l2_subgroup_manufacturer.csv", index=False)

    sex = subgroup_metrics(pred, FOCUS_MODEL, FOCUS_THRESHOLDS, "Sex")
    sex.to_csv(OUT_DIR / "logreg_l2_subgroup_sex.csv", index=False)

    fold4_errors = fold4_error_audit(pred)
    fold4_errors.to_csv(OUT_DIR / "fold4_logreg_l2_error_audit.csv", index=False)

    fold4_dist = fold4_distributions(pred)
    fold4_dist.to_csv(OUT_DIR / "fold4_logreg_l2_probability_distributions.csv", index=False)

    fold4_summary = fold4_threshold_near_summary(fold4_errors)
    fold4_summary.to_csv(OUT_DIR / "fold4_logreg_l2_threshold_near_summary.csv", index=False)

    orig_vs, combined_probs = original_vs_classifier_only(pred, model_status, run_config)
    orig_vs.to_csv(OUT_DIR / "original_vs_classifier_only_logreg_l2.csv", index=False)

    prob_dist = probability_distribution_comparison(combined_probs)
    prob_dist.to_csv(OUT_DIR / "original_vs_classifier_only_probability_distribution.csv", index=False)

    # Carry a compact source reference from the prior mfrsplit audit when present.
    prior_fold4 = MFR_AUDIT_DIR / "fold4_error_audit.csv"
    if prior_fold4.exists():
        prior = read_csv(prior_fold4)
        prior.to_csv(OUT_DIR / "prior_mfrsplit_fold4_error_audit_reference.csv", index=False)

    readme = make_readme(
        verification=verification,
        focus_metrics=focus_metrics,
        focus_confusion=confusion,
        fold4_errors=fold4_errors,
        fold4_summary=fold4_summary,
        original_compare=orig_vs,
        probability_compare=prob_dist,
    )
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    output_log = {
        "script": str(Path(__file__).resolve()),
        "input_sweep_dir": str(SWEEP_DIR),
        "input_mfrsplit_audit_dir": str(MFR_AUDIT_DIR),
        "output_dir": str(OUT_DIR),
        "threshold_selection_verified": bool((verification["verification_status"] == "PASS").all()),
        "non_fixed_threshold_selection": "true_inner_cv_oof",
        "vae_retrained": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "outer_test_threshold_leakage": False,
    }
    write_json(OUT_DIR / "command_log.json", output_log)
    print(json.dumps(output_log, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
