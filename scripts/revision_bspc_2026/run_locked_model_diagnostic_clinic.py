#!/usr/bin/env python3
"""Read-only diagnostic clinic for the locked ADNI model and OASIS pilot transfer.

This script aggregates existing prediction/QC artifacts only. It does not train,
score new data, refit thresholds, or modify tensors/metadata/model outputs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results" / "revision_bspc_2026"
OUTPUT_DIR = RESULTS_ROOT / "locked_model_diagnostic_clinic"

ADNI_RUN = RESULTS_ROOT / "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5"
ADNI_READOUT = ADNI_RUN / "classifier_only_readout"
ADNI_PRED = ADNI_READOUT / "classifier_sweep_predictions.csv"
ADNI_THRESHOLDS = ADNI_READOUT / "classifier_sweep_thresholds_by_fold.csv"

OASIS_PRIMARY = RESULTS_ROOT / "oasis_tanda_2026_05_25_external_scoring"
OASIS_SECONDARY = RESULTS_ROOT / "oasis_tanda_2026_05_25_external_scoring_secondary_models"
OASIS_140TR = RESULTS_ROOT / "oasis_tanda_2026_05_25_140TR_sensitivity"

PRIMARY_MODEL_ALIASES = {
    "primary_v5_1b_ch1_0_2_horizon4480",
    "locked_v5_1b_ch1_0_2_horizon4480",
}
PRIMARY_MODEL_CANONICAL = "locked_v5_1b_ch1_0_2_horizon4480"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def clean(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null", "."} else text


def md_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    view = df if max_rows is None else df.head(max_rows)
    if view.empty:
        return "_No rows._\n"
    header = "| " + " | ".join(str(c) for c in view.columns) + " |"
    sep = "| " + " | ".join("---" for _ in view.columns) + " |"
    rows = ["| " + " | ".join(clean(v) for v in row.tolist()) + " |" for _, row in view.iterrows()]
    extra = ""
    if max_rows is not None and len(df) > max_rows:
        extra = f"\n\n_Showing first {max_rows} of {len(df)} rows._\n"
    return "\n".join([header, sep] + rows) + extra


def write_table(df: pd.DataFrame, stem: str, max_md_rows: int | None = None) -> None:
    df.to_csv(OUTPUT_DIR / f"{stem}.csv", index=False)
    (OUTPUT_DIR / f"{stem}.md").write_text(md_table(df, max_rows=max_md_rows), encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def label_error_type(y_true: int, y_pred: int) -> str:
    if y_true == 0 and y_pred == 0:
        return "TN"
    if y_true == 0 and y_pred == 1:
        return "FP"
    if y_true == 1 and y_pred == 0:
        return "FN"
    if y_true == 1 and y_pred == 1:
        return "TP"
    return "unknown"


def add_sitecode(subject_id: Any) -> str:
    text = clean(subject_id)
    if len(text) >= 3 and text[:3].isdigit():
        return text[:3]
    return ""


def add_age_bin(age: Any) -> str:
    try:
        value = float(age)
    except (TypeError, ValueError):
        return "missing"
    bins = [0, 60, 65, 70, 75, 80, 200]
    labels = ["<60", "60-64", "65-69", "70-74", "75-79", "80+"]
    return str(pd.cut([value], bins=bins, labels=labels, right=False)[0])


def confusion_counts(df: pd.DataFrame) -> dict[str, Any]:
    y_true = df["y_true"].astype(int)
    y_pred = df["y_pred"].astype(int)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    return {
        "n": int(len(df)),
        "n_cn": int((y_true == 0).sum()),
        "n_ad": int((y_true == 1).sum()),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": np.nanmean([sens, spec]),
        "f1": f1,
        "fp_rate_cn": fp / (tn + fp) if tn + fp else np.nan,
        "fn_rate_ad": fn / (tp + fn) if tp + fn else np.nan,
    }


def group_error_summary(df: pd.DataFrame, columns: Iterable[str], dataset: str) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        for value, sub in df.groupby(col, dropna=False):
            row = {
                "dataset": dataset,
                "grouping": col,
                "group_value": clean(value) or "missing",
                **confusion_counts(sub),
                "score_mean": float(sub["y_score"].mean()),
                "score_median": float(sub["y_score"].median()),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def brier_score(y_true: pd.Series, score: pd.Series) -> float:
    y = y_true.astype(float).to_numpy()
    p = score.astype(float).to_numpy()
    return float(np.mean((p - y) ** 2))


def calibration_bins(df: pd.DataFrame, dataset: str, model: str, build: str, n_bins: int = 10) -> pd.DataFrame:
    rows = []
    y = df["y_true"].astype(float)
    p = df["y_score"].astype(float)
    brier = brier_score(y, p)
    ece = 0.0
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        sub = df[mask]
        if len(sub):
            mean_pred = float(sub["y_score"].mean())
            observed = float(sub["y_true"].mean())
            ece += len(sub) / len(df) * abs(mean_pred - observed)
        else:
            mean_pred = np.nan
            observed = np.nan
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "build_candidate": build,
                "bin": i + 1,
                "bin_low": lo,
                "bin_high": hi,
                "n": int(len(sub)),
                "mean_predicted_probability": mean_pred,
                "observed_ad_fraction": observed,
                "brier": brier,
                "ece_10bin": ece if i == n_bins - 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def score_distribution(df: pd.DataFrame, dataset: str, model: str, build: str) -> pd.DataFrame:
    rows = []
    for diagnosis_value, sub in df.groupby("y_true", dropna=False):
        label = "AD" if int(diagnosis_value) == 1 else "CN"
        threshold = float(sub["threshold"].mean()) if "threshold" in sub else float(sub["adni_threshold"].mean())
        scores = sub["y_score"].astype(float)
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "build_candidate": build,
                "class": label,
                "n": int(len(sub)),
                "score_mean": float(scores.mean()),
                "score_sd": float(scores.std(ddof=1)) if len(sub) > 1 else 0.0,
                "score_median": float(scores.median()),
                "score_q25": float(scores.quantile(0.25)),
                "score_q75": float(scores.quantile(0.75)),
                "score_min": float(scores.min()),
                "score_max": float(scores.max()),
                "threshold_mean": threshold,
                "fraction_above_threshold": float((scores >= threshold).mean()),
                "threshold_percentile_within_class": float((scores < threshold).mean()),
            }
        )
    return pd.DataFrame(rows)


def load_adni_primary() -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = read_csv(ADNI_PRED)
    primary = pred[
        (pred["model_name"] == "logreg_l2")
        & (pred["threshold_strategy"] == PRIMARY_THRESHOLD)
    ].copy()
    if primary.empty:
        raise RuntimeError("No ADNI primary Stage B predictions found.")
    primary["SiteCode"] = primary["SubjectID"].map(add_sitecode)
    primary["AgeBin"] = primary["Age"].map(add_age_bin)
    primary["error_type"] = [
        label_error_type(int(y), int(p)) for y, p in zip(primary["y_true"], primary["y_pred"])
    ]
    primary["margin_to_threshold"] = primary["y_score"] - primary["threshold"]
    primary["abs_margin_to_threshold"] = primary["margin_to_threshold"].abs()
    thresholds = read_csv(ADNI_THRESHOLDS)
    return primary, thresholds


def load_oasis_ensemble_predictions() -> pd.DataFrame:
    frames = []
    secondary = read_csv(OASIS_SECONDARY / "predictions.csv")
    frames.append(secondary)
    oasis140 = read_csv(OASIS_140TR / "predictions.csv")
    frames.append(oasis140)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined["prediction_level"] == "ensemble_mean_score_majority_vote"].copy()
    combined["adni_model"] = combined["adni_model"].replace(
        {"primary_v5_1b_ch1_0_2_horizon4480": PRIMARY_MODEL_CANONICAL}
    )
    combined["model_role"] = combined.get("model_role", "").fillna("")
    combined["error_type"] = [
        label_error_type(int(y), int(p)) for y, p in zip(combined["y_true"], combined["y_pred"])
    ]
    combined["margin_to_threshold"] = combined["y_score"] - combined["adni_threshold"]
    combined["abs_margin_to_threshold"] = combined["margin_to_threshold"].abs()
    return combined


def oasis_primary_secondary_consistency() -> pd.DataFrame:
    primary = read_csv(OASIS_PRIMARY / "predictions.csv")
    secondary = read_csv(OASIS_SECONDARY / "predictions.csv")
    primary = primary[primary["prediction_level"] == "ensemble_mean_score_majority_vote"].copy()
    secondary = secondary[
        (secondary["prediction_level"] == "ensemble_mean_score_majority_vote")
        & (secondary["adni_model"].replace({"primary_v5_1b_ch1_0_2_horizon4480": PRIMARY_MODEL_CANONICAL})
           == PRIMARY_MODEL_CANONICAL)
    ].copy()
    primary["adni_model_norm"] = primary["adni_model"].replace(
        {"primary_v5_1b_ch1_0_2_horizon4480": PRIMARY_MODEL_CANONICAL}
    )
    secondary["adni_model_norm"] = secondary["adni_model"].replace(
        {"primary_v5_1b_ch1_0_2_horizon4480": PRIMARY_MODEL_CANONICAL}
    )
    key = ["SubjectID", "build_candidate", "prediction_level", "threshold_strategy", "adni_model_norm"]
    merged = primary.merge(
        secondary,
        on=key,
        how="outer",
        suffixes=("_primary_file", "_secondary_file"),
        indicator=True,
    )
    rows = []
    for build, sub in merged.groupby("build_candidate", dropna=False):
        score_diff = (
            (sub["y_score_primary_file"] - sub["y_score_secondary_file"]).abs()
            if {"y_score_primary_file", "y_score_secondary_file"}.issubset(sub.columns)
            else pd.Series(dtype=float)
        )
        pred_diff = (
            sub["y_pred_primary_file"].ne(sub["y_pred_secondary_file"])
            if {"y_pred_primary_file", "y_pred_secondary_file"}.issubset(sub.columns)
            else pd.Series(dtype=bool)
        )
        rows.append(
            {
                "build_candidate": build,
                "primary_rows": int((sub["_merge"] != "right_only").sum()),
                "secondary_rows": int((sub["_merge"] != "left_only").sum()),
                "matched_rows": int((sub["_merge"] == "both").sum()),
                "primary_only_rows": int((sub["_merge"] == "left_only").sum()),
                "secondary_only_rows": int((sub["_merge"] == "right_only").sum()),
                "max_abs_score_diff": float(score_diff.max()) if len(score_diff.dropna()) else np.nan,
                "n_prediction_disagreements": int(pred_diff.fillna(False).sum()) if len(pred_diff) else 0,
            }
        )
    return pd.DataFrame(rows)


def oasis_stability(oasis: pd.DataFrame) -> pd.DataFrame:
    key_cols = ["adni_model", "SubjectID", "diagnosis", "y_true"]
    rows = []
    for key, sub in oasis.groupby(key_cols, dropna=False):
        model, subject, diagnosis, y_true = key
        preds = dict(zip(sub["build_candidate"], sub["error_type"]))
        scores = dict(zip(sub["build_candidate"], sub["y_score"]))
        error_types = [e for e in preds.values() if e in {"FP", "FN"}]
        correct_types = [e for e in preds.values() if e in {"TN", "TP"}]
        if len(error_types) == 0:
            stability = "stable_correct"
        elif len(correct_types) == 0:
            stability = "stable_error"
        else:
            stability = "unstable_error"
        rows.append(
            {
                "adni_model": model,
                "SubjectID": subject,
                "diagnosis": diagnosis,
                "y_true": int(y_true),
                "n_builds": int(sub["build_candidate"].nunique()),
                "error_stability": stability,
                "error_count_across_builds": len(error_types),
                "concatenated_error_type": preds.get("concatenated_timeseries", ""),
                "runwise164_error_type": preds.get("runwise_connectome_average", ""),
                "runwise140TR_error_type": preds.get("runwise_140TR_connectome_average", ""),
                "concatenated_score": scores.get("concatenated_timeseries", np.nan),
                "runwise164_score": scores.get("runwise_connectome_average", np.nan),
                "runwise140TR_score": scores.get("runwise_140TR_connectome_average", np.nan),
                "score_range_across_builds": float(sub["y_score"].max() - sub["y_score"].min()),
                "age_at_MR": sub["age_at_MR"].iloc[0],
                "sex": sub["sex"].iloc[0],
                "selected_qc_runs": sub["selected_qc_runs"].iloc[0],
                "selected_run_ids": sub["selected_run_ids"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def collect_latent_qc() -> pd.DataFrame:
    rows = []
    for fold_dir in sorted(ADNI_RUN.glob("fold_*")):
        if not fold_dir.is_dir():
            continue
        fold = clean(fold_dir.name).replace("fold_", "")
        info_path = fold_dir / f"fold_{fold}_test_latent_info_summary.csv"
        leak_path = fold_dir / f"fold_{fold}_test_scanner_leakage_summary.csv"
        if info_path.exists():
            info = pd.read_csv(info_path)
            for _, row in info.iterrows():
                rows.append(
                    {
                        "dataset": "ADNI_test_latent",
                        "fold": int(fold),
                        "qc_type": "latent_information",
                        "variable": row.get("variable", ""),
                        "mi_sum_nats": row.get("mi_sum_nats", np.nan),
                        "mi_mean_nats": row.get("mi_mean_nats", np.nan),
                        "n_active": row.get("n_active", np.nan),
                        "frac_active": row.get("frac_active", np.nan),
                        "total_correlation_nats": row.get("total_correlation_nats", np.nan),
                        "acc_site_raw": np.nan,
                        "acc_site_latent": np.nan,
                        "latent_minus_raw": np.nan,
                    }
                )
        if leak_path.exists():
            leak = pd.read_csv(leak_path)
            for _, row in leak.iterrows():
                raw = row.get("acc_site_raw", np.nan)
                latent = row.get("acc_site_latent", np.nan)
                rows.append(
                    {
                        "dataset": "ADNI_test_latent",
                        "fold": int(fold),
                        "qc_type": "scanner_leakage",
                        "variable": row.get("site_col", ""),
                        "mi_sum_nats": np.nan,
                        "mi_mean_nats": np.nan,
                        "n_active": np.nan,
                        "frac_active": np.nan,
                        "total_correlation_nats": np.nan,
                        "acc_site_raw": raw,
                        "acc_site_latent": latent,
                        "latent_minus_raw": latent - raw if pd.notna(raw) and pd.notna(latent) else np.nan,
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        oas_row = {
            "dataset": "OASIS_latent",
            "fold": "",
            "qc_type": "cohort_separability",
            "variable": "ADNI_vs_OASIS",
            "mi_sum_nats": np.nan,
            "mi_mean_nats": np.nan,
            "n_active": np.nan,
            "frac_active": np.nan,
            "total_correlation_nats": np.nan,
            "acc_site_raw": np.nan,
            "acc_site_latent": np.nan,
            "latent_minus_raw": np.nan,
            "note": "OASIS latent cache files were not present in the external scoring outputs; cohort separability was not recomputed.",
        }
        out["note"] = ""
        out = pd.concat([out, pd.DataFrame([oas_row])], ignore_index=True)
    return out


def write_readme(
    adni: pd.DataFrame,
    oasis_primary: pd.DataFrame,
    stable: pd.DataFrame,
    adni_cal: pd.DataFrame,
    oasis_cal: pd.DataFrame,
    latent_qc: pd.DataFrame,
    t0: str,
) -> None:
    adni_counts = confusion_counts(adni)
    locked = oasis_primary[oasis_primary["adni_model"] == PRIMARY_MODEL_CANONICAL]
    stable_locked = stable[stable["adni_model"] == PRIMARY_MODEL_CANONICAL]
    stable_counts = stable_locked["error_stability"].value_counts().to_dict()
    adni_brier = adni_cal["brier"].dropna().iloc[0] if not adni_cal.empty else np.nan
    adni_ece = adni_cal["ece_10bin"].dropna().iloc[0] if not adni_cal.empty else np.nan
    oasis_briers = (
        oasis_cal.groupby(["model", "build_candidate"])["brier"].first().reset_index()
        if not oasis_cal.empty
        else pd.DataFrame()
    )

    scanner_rows = latent_qc[
        (latent_qc["qc_type"] == "scanner_leakage") & (latent_qc["variable"] == "Manufacturer")
    ]
    mean_latent_leak = scanner_rows["acc_site_latent"].mean() if not scanner_rows.empty else np.nan
    mean_raw_leak = scanner_rows["acc_site_raw"].mean() if not scanner_rows.empty else np.nan

    readme = f"""# Locked Model Diagnostic Clinic

Generated: `{t0}`

## Scope

This is a read-only diagnostic aggregation for the locked ADNI model, the OASIS
pilot external scoring outputs, the OASIS 140TR sensitivity, and the secondary
model scoring outputs. No training, threshold fitting, model selection, or new
scoring was performed.

## ADNI internal behavior

The locked ADNI Stage B readout at the primary threshold has:

- TN `{adni_counts['tn']}`, FP `{adni_counts['fp']}`, FN `{adni_counts['fn']}`, TP `{adni_counts['tp']}`
- Sensitivity `{adni_counts['sensitivity']:.3f}`
- Specificity `{adni_counts['specificity']:.3f}`
- Balanced accuracy `{adni_counts['balanced_accuracy']:.3f}`
- F1 `{adni_counts['f1']:.3f}`

The main ADNI failure mode is not CN collapse. The sensitivity-constrained
operating point intentionally keeps AD sensitivity high and leaves a visible CN
false-positive burden.

## OASIS transfer behavior

For the locked model, OASIS errors are split into stable and run-handling-sensitive
cases across concatenated, runwise164, and runwise140TR scoring:

{md_table(pd.DataFrame([stable_counts]))}

The ADNI threshold transfers conservatively to OASIS: specificity is generally
high, while sensitivity is low because many OASIS AD_DEMENTIA scores fall below
the ADNI-derived threshold. This supports calibration as the next external step,
not additional ADNI model optimization.

## Calibration

- ADNI Brier: `{adni_brier:.4f}`
- ADNI 10-bin ECE: `{adni_ece:.4f}`

OASIS calibration estimates in this package are descriptive only. They quantify
transfer shift but are not used to fit thresholds or choose models.

## Latent QC

ADNI latent QC is summarized from saved per-fold artifacts. Mean Manufacturer
balanced accuracy from raw inputs was `{mean_raw_leak:.3f}` and from latent
features was `{mean_latent_leak:.3f}` across available test-fold leakage audits.
OASIS latent caches were not present in the external scoring outputs, so
ADNI-vs-OASIS latent cohort separability was not recomputed.

## Reviewer-ready interpretation

The model ranks AD above CN substantially better than chance in ADNI and shows a
non-zero external ranking signal on OASIS, but its fixed ADNI operating threshold
does not transfer cleanly across cohorts. This is exactly the distinction between
ranking and deployment calibration: ROC/PR ranking can remain informative while
the sensitivity-constrained threshold shifts under a new acquisition and cohort
distribution. The next scientifically appropriate external step is a pre-specified
OASIS calibration/test protocol, not further internal AUC tuning.
"""
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    reviewer = """# Reviewer-Ready Diagnostic Interpretation

The locked model's strongest behavior is conservative, reproducible ranking under
nested ADNI validation, with explicit scanner/manufacturer auditing and a
sensitivity-constrained readout. It does not collapse to the majority CN class;
instead, the selected operating point trades specificity for AD sensitivity,
producing CN false positives that are visible in the subject-level audit. On the
OASIS pilot, the model retains external ranking signal, but the ADNI-derived
threshold is too high for many OASIS AD_DEMENTIA cases, yielding low sensitivity
despite high specificity. This failure mode is calibration and cohort-shift
related rather than evidence that the VAE should be retrained on OASIS. The
appropriate next step is therefore a locked OASIS calibration/test design:
threshold-only or intercept-only recalibration on a pre-declared calibration
subset, followed by one evaluation on a held-out OASIS test subset. We will not
merge OASIS with ADNI training, retrain the VAE on OASIS, or use OASIS test labels
for model selection.
"""
    (OUTPUT_DIR / "reviewer_ready_interpretation.md").write_text(reviewer, encoding="utf-8")


def main() -> None:
    t0 = now_utc()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    adni, thresholds = load_adni_primary()
    oasis = load_oasis_ensemble_predictions()
    oasis_primary = oasis[oasis["adni_model"] == PRIMARY_MODEL_CANONICAL].copy()

    # ADNI error analysis.
    write_table(adni.sort_values(["fold", "error_type", "SubjectID"]), "adni_subject_error_table", max_md_rows=160)
    fold_rows = []
    for fold, sub in adni.groupby("fold", dropna=False):
        fold_rows.append({"fold": fold, **confusion_counts(sub)})
    write_table(pd.DataFrame(fold_rows), "adni_error_by_fold")
    write_table(
        group_error_summary(adni, ["AgeBin", "Sex", "Manufacturer", "SiteCode", "source_batch"], "ADNI"),
        "adni_error_by_age_sex_manufacturer_sitecode",
    )

    # OASIS error analysis across primary and secondary model scoring outputs.
    write_table(oasis.sort_values(["adni_model", "build_candidate", "error_type", "SubjectID"]), "oasis_error_subjects_all_models", max_md_rows=220)
    oasis_group_cols = ["adni_model", "build_candidate"]
    oasis_summary_rows = []
    for key, sub in oasis.groupby(oasis_group_cols, dropna=False):
        row = dict(zip(oasis_group_cols, key if isinstance(key, tuple) else (key,)))
        row.update(confusion_counts(sub))
        oasis_summary_rows.append(row)
    oasis_summary = pd.DataFrame(oasis_summary_rows)
    write_table(oasis_summary, "oasis_error_summary_by_model_and_build")
    write_table(oasis_primary_secondary_consistency(), "oasis_primary_secondary_consistency")
    stable = oasis_stability(oasis)
    write_table(stable.sort_values(["adni_model", "error_stability", "SubjectID"]), "oasis_error_stability_by_subject", max_md_rows=220)
    write_table(
        stable.groupby(["adni_model", "error_stability"], dropna=False)
        .size()
        .reset_index(name="n_subjects")
        .sort_values(["adni_model", "error_stability"]),
        "oasis_error_stability_summary",
    )

    # Score shift and threshold location.
    score_rows = [score_distribution(adni.rename(columns={"threshold": "threshold"}), "ADNI_outer_test", PRIMARY_MODEL_CANONICAL, "nested_oof")]
    for (model, build), sub in oasis.groupby(["adni_model", "build_candidate"], dropna=False):
        tmp = sub.rename(columns={"adni_threshold": "threshold"})
        score_rows.append(score_distribution(tmp, "OASIS_pilot", model, build))
    score_shift = pd.concat(score_rows, ignore_index=True)
    write_table(score_shift, "score_shift_and_threshold_location")

    # Calibration.
    adni_cal = calibration_bins(adni, "ADNI_outer_test", PRIMARY_MODEL_CANONICAL, "nested_oof")
    oasis_cal_frames = []
    for (model, build), sub in oasis.groupby(["adni_model", "build_candidate"], dropna=False):
        oasis_cal_frames.append(calibration_bins(sub, "OASIS_pilot_descriptive_only", model, build))
    oasis_cal = pd.concat(oasis_cal_frames, ignore_index=True)
    write_table(adni_cal, "calibration_adni_reliability_bins")
    write_table(oasis_cal, "calibration_oasis_descriptive_reliability_bins")

    # Latent QC.
    latent_qc = collect_latent_qc()
    write_table(latent_qc, "latent_space_qc_summary")

    # Thresholds and command log.
    write_table(thresholds, "adni_thresholds_by_fold")
    command_log = {
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "timestamp_utc": t0,
        "output_dir": str(OUTPUT_DIR.relative_to(PROJECT_ROOT)),
        "inputs": {
            "adni_predictions": str(ADNI_PRED.relative_to(PROJECT_ROOT)),
            "adni_thresholds": str(ADNI_THRESHOLDS.relative_to(PROJECT_ROOT)),
            "oasis_primary_predictions": str((OASIS_PRIMARY / "predictions.csv").relative_to(PROJECT_ROOT)),
            "oasis_secondary_predictions": str((OASIS_SECONDARY / "predictions.csv").relative_to(PROJECT_ROOT)),
            "oasis_140tr_predictions": str((OASIS_140TR / "predictions.csv").relative_to(PROJECT_ROOT)),
        },
        "actions": [
            "aggregated_saved_adni_stageb_predictions",
            "aggregated_saved_oasis_ensemble_predictions",
            "computed_error_summaries",
            "computed_descriptive_score_shift_and_calibration",
            "aggregated_existing_adni_latent_qc",
        ],
        "safety": {
            "training_performed": False,
            "new_model_scoring_performed": False,
            "threshold_fitting_performed": False,
            "model_selection_performed": False,
            "modified_tensors_metadata_or_model_outputs": False,
        },
    }
    (OUTPUT_DIR / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")

    write_readme(adni, oasis_primary, stable, adni_cal, oasis_cal, latent_qc, t0)
    print(f"Wrote locked-model diagnostic clinic to {OUTPUT_DIR}")
    print(f"ADNI rows: {len(adni)}")
    print(f"OASIS ensemble rows: {len(oasis)}")


if __name__ == "__main__":
    main()
