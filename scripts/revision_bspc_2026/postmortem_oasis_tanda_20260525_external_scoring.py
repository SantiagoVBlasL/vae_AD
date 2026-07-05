#!/usr/bin/env python
"""Read-only postmortem for OASIS external scoring with locked ADNI model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCORING_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "oasis_tanda_2026_05_25_external_scoring"
)
DEFAULT_CONNECTOME_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "oasis_tanda_2026_05_25_connectomes"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "oasis_tanda_2026_05_25_external_scoring_postmortem"
)
PRIMARY_LEVEL = "ensemble_mean_score_majority_vote"
THRESHOLD_STRATEGY = "inner_oof_target_sens_ge_0p70_max_spec"
RNG_SEED = 20260526


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scoring-dir", type=Path, default=DEFAULT_SCORING_DIR)
    parser.add_argument("--connectome-dir", type=Path, default=DEFAULT_CONNECTOME_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--n-permutations", type=int, default=5000)
    return parser.parse_args()


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def write_csv_md(df: pd.DataFrame, csv_path: Path, md_path: Path, title: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        if df.empty:
            f.write("_No rows._\n")
        else:
            f.write(df.to_markdown(index=False))
            f.write("\n")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def binary_metrics(y_true: Sequence[int], y_score: Sequence[float], y_pred: Sequence[int]) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    return {
        "n": int(len(y)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "accuracy": safe_div(tp + tn, len(y)),
    }


def bootstrap_ci(
    df: pd.DataFrame,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    metrics = ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]
    values = {m: [] for m in metrics}
    n = len(df)
    y = df["y_true"].astype(int).to_numpy()
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        sample = df.iloc[idx]
        m = binary_metrics(sample["y_true"], sample["y_score"], sample["y_pred"])
        for metric in metrics:
            values[metric].append(m[metric])
    out: dict[str, float] = {}
    for metric in metrics:
        arr = np.asarray(values[metric], dtype=float)
        arr = arr[np.isfinite(arr)]
        out[f"{metric}_ci_low"] = float(np.quantile(arr, 0.025)) if arr.size else np.nan
        out[f"{metric}_ci_high"] = float(np.quantile(arr, 0.975)) if arr.size else np.nan
        out[f"{metric}_bootstrap_valid_n"] = int(arr.size)
    return out


def permutation_auc_pvalue(
    y_true: Sequence[int],
    y_score: Sequence[float],
    n_permutations: int,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    observed = float(roc_auc_score(y, score))
    count_ge = 0
    for _ in range(n_permutations):
        yp = rng.permutation(y)
        auc_p = float(roc_auc_score(yp, score))
        if auc_p >= observed:
            count_ge += 1
    p_value = (count_ge + 1) / (n_permutations + 1)
    return observed, float(p_value), int(n_permutations)


def score_overlap_fraction(cn_scores: np.ndarray, ad_scores: np.ndarray) -> float:
    if cn_scores.size == 0 or ad_scores.size == 0:
        return float("nan")
    # Pairwise probability that a CN score is at least as AD-like as an AD score.
    return float((cn_scores[:, None] >= ad_scores[None, :]).mean())


def threshold_candidates(scores: Sequence[float]) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    return np.unique(np.round(np.clip(np.concatenate(([0.0, 0.5, 1.0], s)), 0.0, 1.0), 12))


def threshold_table(y_true: Sequence[int], y_score: Sequence[float]) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    rows = []
    for thr in threshold_candidates(score):
        pred = (score >= thr).astype(int)
        row = {"threshold": float(thr)}
        row.update(binary_metrics(y, score, pred))
        row["youden_j"] = row["sensitivity"] + row["specificity"] - 1.0
        rows.append(row)
    return pd.DataFrame(rows)


def descriptive_oracle_thresholds(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (build_candidate, adni_model), sub in df.groupby(["build_candidate", "adni_model"], dropna=False):
        tbl = threshold_table(sub["y_true"], sub["y_score"])
        youden = tbl.sort_values(["youden_j", "balanced_accuracy", "threshold"], ascending=[False, False, False]).iloc[0]
        eligible = tbl[tbl["sensitivity"] >= 0.70]
        target = (
            eligible.sort_values(["specificity", "sensitivity", "balanced_accuracy", "threshold"], ascending=[False, False, False, False]).iloc[0]
            if not eligible.empty
            else tbl.sort_values(["sensitivity", "specificity", "threshold"], ascending=[False, False, False]).iloc[0]
        )
        for label, row in [
            ("posthoc_oasis_youden", youden),
            ("posthoc_oasis_target_sens_ge_0p70_max_spec", target),
        ]:
            out = {
                "build_candidate": build_candidate,
                "adni_model": adni_model,
                "oracle_threshold_strategy": label,
                "posthoc_descriptive_only": True,
            }
            out.update(row.to_dict())
            rows.append(out)
    return pd.DataFrame(rows)


def primary_metrics_with_ci(primary: pd.DataFrame, n_bootstrap: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for (build_candidate, adni_model), sub in primary.groupby(["build_candidate", "adni_model"], dropna=False):
        row = {
            "build_candidate": build_candidate,
            "adni_model": adni_model,
            "prediction_level": PRIMARY_LEVEL,
            "threshold_strategy": THRESHOLD_STRATEGY,
        }
        row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        row.update(bootstrap_ci(sub, n_bootstrap, rng))
        rows.append(row)
    return pd.DataFrame(rows)


def permutation_table(primary: pd.DataFrame, n_permutations: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for (build_candidate, adni_model), sub in primary.groupby(["build_candidate", "adni_model"], dropna=False):
        auc_obs, p_value, n_perm = permutation_auc_pvalue(sub["y_true"], sub["y_score"], n_permutations, rng)
        rows.append(
            {
                "build_candidate": build_candidate,
                "adni_model": adni_model,
                "prediction_level": PRIMARY_LEVEL,
                "observed_auc": auc_obs,
                "n_permutations": n_perm,
                "one_sided_p_auc_ge_observed": p_value,
                "null_hypothesis": "labels exchangeable; no ranking association",
            }
        )
    return pd.DataFrame(rows)


def score_distribution(primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (build_candidate, adni_model), sub_build in primary.groupby(["build_candidate", "adni_model"], dropna=False):
        cn = sub_build[sub_build["y_true"].eq(0)]["y_score"].to_numpy(float)
        ad = sub_build[sub_build["y_true"].eq(1)]["y_score"].to_numpy(float)
        overlap = score_overlap_fraction(cn, ad)
        for diagnosis, sub in sub_build.groupby("diagnosis", dropna=False):
            q25, q75 = np.quantile(sub["y_score"], [0.25, 0.75])
            rows.append(
                {
                    "build_candidate": build_candidate,
                    "adni_model": adni_model,
                    "diagnosis": diagnosis,
                    "n": int(len(sub)),
                    "score_mean": float(sub["y_score"].mean()),
                    "score_median": float(sub["y_score"].median()),
                    "score_q25": float(q25),
                    "score_q75": float(q75),
                    "score_iqr": float(q75 - q25),
                    "score_min": float(sub["y_score"].min()),
                    "score_max": float(sub["y_score"].max()),
                    "overlap_fraction_cn_score_ge_ad_score": overlap,
                }
            )
    return pd.DataFrame(rows)


def threshold_transfer(pred: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fold_rows = pred[pred["prediction_level"].eq("fold_model")].copy()
    for (build_candidate, adni_model, fold), sub in fold_rows.groupby(["build_candidate", "adni_model", "fold"], dropna=False):
        ad = sub[sub["y_true"].eq(1)]
        q = sub["y_score"].quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).to_dict()
        threshold = float(sub["adni_threshold"].iloc[0])
        rows.append(
            {
                "build_candidate": build_candidate,
                "adni_model": adni_model,
                "prediction_level": "fold_model",
                "fold": fold,
                "adni_threshold": threshold,
                "oasis_score_min": float(q[0.0]),
                "oasis_score_q10": float(q[0.1]),
                "oasis_score_q25": float(q[0.25]),
                "oasis_score_median": float(q[0.5]),
                "oasis_score_q75": float(q[0.75]),
                "oasis_score_q90": float(q[0.9]),
                "oasis_score_max": float(q[1.0]),
                "n_ad": int(len(ad)),
                "n_ad_below_adni_threshold": int((ad["y_score"] < threshold).sum()),
                "fraction_ad_below_adni_threshold": float((ad["y_score"] < threshold).mean()) if len(ad) else np.nan,
            }
        )
    for (build_candidate, adni_model), sub in primary.groupby(["build_candidate", "adni_model"], dropna=False):
        ad = sub[sub["y_true"].eq(1)]
        q = sub["y_score"].quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]).to_dict()
        rows.append(
            {
                "build_candidate": build_candidate,
                "adni_model": adni_model,
                "prediction_level": PRIMARY_LEVEL,
                "fold": "ensemble",
                "adni_threshold": float(sub["adni_threshold"].mean()),
                "oasis_score_min": float(q[0.0]),
                "oasis_score_q10": float(q[0.1]),
                "oasis_score_q25": float(q[0.25]),
                "oasis_score_median": float(q[0.5]),
                "oasis_score_q75": float(q[0.75]),
                "oasis_score_q90": float(q[0.9]),
                "oasis_score_max": float(q[1.0]),
                "n_ad": int(len(ad)),
                "n_ad_below_adni_threshold": int((ad["y_pred"] == 0).sum()),
                "fraction_ad_below_adni_threshold": float((ad["y_pred"] == 0).mean()) if len(ad) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_combination_comparison(primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for adni_model, sub in primary.groupby("adni_model", dropna=False):
        piv = sub.pivot_table(
            index=["SubjectID", "session_id", "diagnosis", "Age", "Sex", "selected_run_ids"],
            columns="build_candidate",
            values=["y_score", "y_pred"],
            aggfunc="first",
        )
        if not {"concatenated_timeseries", "runwise_connectome_average"}.issubset(set(piv["y_score"].columns)):
            continue
        concat_score = piv[("y_score", "concatenated_timeseries")]
        runwise_score = piv[("y_score", "runwise_connectome_average")]
        concat_pred = piv[("y_pred", "concatenated_timeseries")]
        runwise_pred = piv[("y_pred", "runwise_connectome_average")]
        corr = float(concat_score.corr(runwise_score, method="pearson"))
        rows.append(
            {
                "adni_model": adni_model,
                "comparison_level": "summary",
                "n_subject_sessions": int(len(piv)),
                "score_pearson_correlation": corr,
                "mean_concat_minus_runwise_score": float((concat_score - runwise_score).mean()),
                "n_prediction_disagreements": int((concat_pred != runwise_pred).sum()),
                "prediction_disagreement_rate": float((concat_pred != runwise_pred).mean()),
            }
        )
        for idx, row in piv.iterrows():
            if row[("y_pred", "concatenated_timeseries")] != row[("y_pred", "runwise_connectome_average")]:
                idx_vals = dict(zip(piv.index.names, idx))
                rows.append(
                    {
                        "adni_model": adni_model,
                        "comparison_level": "subject_disagreement",
                        **idx_vals,
                        "concat_score": float(row[("y_score", "concatenated_timeseries")]),
                        "runwise_score": float(row[("y_score", "runwise_connectome_average")]),
                        "concat_pred": int(row[("y_pred", "concatenated_timeseries")]),
                        "runwise_pred": int(row[("y_pred", "runwise_connectome_average")]),
                    }
                )
    return pd.DataFrame(rows)


def error_analysis(primary: pd.DataFrame) -> pd.DataFrame:
    concat = primary[primary["build_candidate"].eq("concatenated_timeseries")].copy()
    rows = []
    for _, r in concat.iterrows():
        error_type = "TN" if r["y_true"] == 0 and r["y_pred"] == 0 else "FP" if r["y_true"] == 0 else "FN" if r["y_pred"] == 0 else "TP"
        if error_type not in {"FP", "FN"}:
            continue
        rows.append(
            {
                "SubjectID": r["SubjectID"],
                "session_id": r["session_id"],
                "experiment_id": r.get("experiment_id", ""),
                "diagnosis": r["diagnosis"],
                "error_type": error_type,
                "score": float(r["y_score"]),
                "prediction": int(r["y_pred"]),
                "adni_threshold_mean": float(r["adni_threshold"]),
                "Age": r.get("Age", np.nan),
                "Sex": r.get("Sex", ""),
                "selected_qc_runs": r.get("selected_qc_runs", ""),
                "selected_run_ids": r.get("selected_run_ids", ""),
                "selected_total_timepoints": r.get("selected_total_timepoints", ""),
                "Manufacturer": r.get("Manufacturer", ""),
                "ScannerModel": r.get("ScannerModel", ""),
            }
        )
    return pd.DataFrame(rows).sort_values(["error_type", "score"], ascending=[True, True])


def make_readme(output_dir: Path, primary_metrics: pd.DataFrame, perm: pd.DataFrame) -> None:
    text = f"""# OASIS External Scoring Postmortem

This is a read-only analysis of existing OASIS external predictions. No model was trained, no OASIS threshold was fit for primary metrics, and predictions were not modified.

## Primary Interpretation

Primary prediction level: `{PRIMARY_LEVEL}`

The locked ADNI model shows external ranking signal if AUC is above chance and the permutation test is significant. The ADNI-transferred operating threshold is conservative on OASIS: specificity is high, but many AD_DEMENTIA subjects fall below the ADNI-derived threshold, producing low sensitivity.

## Primary Metrics

{primary_metrics.to_markdown(index=False)}

## AUC Permutation Tests

{perm.to_markdown(index=False)}
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def make_final_recommendation(
    output_dir: Path,
    primary_metrics: pd.DataFrame,
    threshold_df: pd.DataFrame,
    run_compare: pd.DataFrame,
) -> None:
    primary_short = primary_metrics[
        [
            "build_candidate",
            "adni_model",
            "auc",
            "pr_auc",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "f1",
            "auc_ci_low",
            "auc_ci_high",
        ]
    ]
    threshold_primary = threshold_df[threshold_df["prediction_level"].eq(PRIMARY_LEVEL)]
    text = f"""# Final Recommendation

Decision: `external_ranking_signal_with_threshold_transfer_failure`

## Summary

The locked ADNI model has measurable OASIS ranking signal, but the ADNI-derived threshold transfers poorly for sensitivity. The primary external result should therefore emphasize threshold-independent AUC/PR-AUC and report threshold metrics as an operating-point stress test under domain shift.

## Primary Metrics With AUC CI

{primary_short.to_markdown(index=False)}

## Threshold Transfer

{threshold_primary.to_markdown(index=False)}

## Run Combination Sensitivity

{run_compare.to_markdown(index=False)}

## Reporting Guidance

- Do not tune or select thresholds on OASIS.
- Descriptive oracle thresholds are included only to diagnose domain shift and should not be reported as model performance.
- If manuscript space is limited, report `concatenated_timeseries` as the primary OASIS processing path because it follows Martin's recommendation, and report `runwise_connectome_average` as sensitivity.
"""
    (output_dir / "final_recommendation.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    require(args.scoring_dir / "predictions.csv", "OASIS predictions")
    require(args.connectome_dir / "tensor_qc_summary.csv", "OASIS tensor QC summary")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RNG_SEED)

    pred = pd.read_csv(args.scoring_dir / "predictions.csv")
    primary = pred[pred["prediction_level"].astype(str).eq(PRIMARY_LEVEL)].copy()
    if primary.empty:
        raise ValueError(f"No primary prediction rows found for {PRIMARY_LEVEL}")

    primary_metrics = primary_metrics_with_ci(primary, int(args.n_bootstrap), rng)
    perm = permutation_table(primary, int(args.n_permutations), rng)
    dist = score_distribution(primary)
    threshold_df = threshold_transfer(pred, primary)
    oracle = descriptive_oracle_thresholds(primary)
    run_compare = run_combination_comparison(primary)
    errors = error_analysis(primary)

    write_csv_md(primary_metrics, args.output_dir / "primary_metrics_with_ci.csv", args.output_dir / "primary_metrics_with_ci.md", "Primary Metrics With Bootstrap CI")
    write_csv_md(perm, args.output_dir / "permutation_auc_test.csv", args.output_dir / "permutation_auc_test.md", "Permutation AUC Test")
    write_csv_md(dist, args.output_dir / "score_distribution_by_diagnosis.csv", args.output_dir / "score_distribution_by_diagnosis.md", "Score Distribution By Diagnosis")
    write_csv_md(threshold_df, args.output_dir / "threshold_transfer_analysis.csv", args.output_dir / "threshold_transfer_analysis.md", "Threshold Transfer Analysis")
    write_csv_md(oracle, args.output_dir / "descriptive_oracle_thresholds.csv", args.output_dir / "descriptive_oracle_thresholds.md", "Descriptive Oracle Thresholds")
    write_csv_md(run_compare, args.output_dir / "run_combination_comparison.csv", args.output_dir / "run_combination_comparison.md", "Run Combination Comparison")
    write_csv_md(errors, args.output_dir / "error_analysis_subjects.csv", args.output_dir / "error_analysis_subjects.md", "Error Analysis Subjects")
    make_readme(args.output_dir, primary_metrics, perm)
    make_final_recommendation(args.output_dir, primary_metrics, threshold_df, run_compare)

    command_log = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "scoring_dir": str(args.scoring_dir),
        "connectome_dir": str(args.connectome_dir),
        "output_dir": str(args.output_dir),
        "read_only": True,
        "model_training": False,
        "prediction_modification": False,
        "primary_prediction_level": PRIMARY_LEVEL,
        "n_bootstrap": int(args.n_bootstrap),
        "n_permutations": int(args.n_permutations),
        "rng_seed": RNG_SEED,
        "outputs": [
            "README.md",
            "primary_metrics_with_ci.csv",
            "primary_metrics_with_ci.md",
            "permutation_auc_test.csv",
            "permutation_auc_test.md",
            "score_distribution_by_diagnosis.csv",
            "score_distribution_by_diagnosis.md",
            "threshold_transfer_analysis.csv",
            "threshold_transfer_analysis.md",
            "descriptive_oracle_thresholds.csv",
            "descriptive_oracle_thresholds.md",
            "run_combination_comparison.csv",
            "run_combination_comparison.md",
            "error_analysis_subjects.csv",
            "error_analysis_subjects.md",
            "final_recommendation.md",
            "command_log.json",
        ],
    }
    (args.output_dir / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "n_primary_rows": int(len(primary))}, indent=2))


if __name__ == "__main__":
    main()
