#!/usr/bin/env python3
"""Build a non-destructive prediction audit for ADNI expanded v3 retraining."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = REPO_ROOT / "results/revision_bspc_2026/adni_expanded_v3_beta25_static3"
DEFAULT_PREDICTIONS = DEFAULT_RUN_DIR / "pooled_test_predictions_all_folds.csv"
DEFAULT_METADATA = (
    REPO_ROOT
    / "data/revision_bspc_2026/adni_expanded_v3_all_available/"
    / "subject_metadata_adni_expanded_v3_all_available.csv"
)
DEFAULT_OUTDIR = DEFAULT_RUN_DIR / "audit_v3"

OUTPUT_FILES = [
    "global_metrics_by_classifier.csv",
    "metrics_by_manufacturer.csv",
    "metrics_by_site3.csv",
    "metrics_by_sourcecohort.csv",
    "metrics_by_sex.csv",
    "metrics_by_agebin.csv",
    "threshold_analysis.csv",
    "error_table_false_positives.csv",
    "error_table_false_negatives.csv",
    "subject_level_predictions_with_metadata.csv",
    "README.md",
]


class ColumnDetectionError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a scientific prediction audit for ADNI_expanded_v3_all_available."
    )
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing files inside outdir. Default is non-destructive.",
    )
    return parser.parse_args()


def lower_map(columns: Iterable[str]) -> Dict[str, str]:
    return {str(col).strip().lower(): col for col in columns}


def detect_exact_column(
    df: pd.DataFrame,
    role: str,
    preferred_names: Sequence[str],
    fuzzy_tokens: Sequence[str] = (),
) -> str:
    by_lower = lower_map(df.columns)
    exact_matches = [by_lower[name.lower()] for name in preferred_names if name.lower() in by_lower]
    unique_exact = list(dict.fromkeys(exact_matches))
    if len(unique_exact) == 1:
        return unique_exact[0]
    if len(unique_exact) > 1:
        # Preferred names are ordered from most to least canonical. Use the first match.
        return unique_exact[0]

    fuzzy_matches = []
    for col in df.columns:
        col_l = str(col).lower()
        if all(token.lower() in col_l for token in fuzzy_tokens):
            fuzzy_matches.append(col)
    fuzzy_matches = list(dict.fromkeys(fuzzy_matches))
    if len(fuzzy_matches) == 1:
        return fuzzy_matches[0]
    if len(fuzzy_matches) > 1:
        raise ColumnDetectionError(
            f"Ambiguous {role} column candidates: {fuzzy_matches}\n"
            f"Available columns: {list(df.columns)}"
        )
    raise ColumnDetectionError(
        f"Could not detect {role} column.\nAvailable columns: {list(df.columns)}"
    )


def detect_score_column(df: pd.DataFrame) -> str:
    by_lower = lower_map(df.columns)
    preferred = [
        "y_score_final",
        "prob_ad",
        "probability_ad",
        "ad_probability",
        "p_ad",
        "pred_proba_ad",
        "y_score",
        "score_ad",
    ]
    for name in preferred:
        if name in by_lower:
            return by_lower[name]

    score_like = []
    for col in df.columns:
        col_l = str(col).lower()
        if "score" in col_l or "prob" in col_l or "proba" in col_l:
            if "cn" not in col_l and "control" not in col_l:
                score_like.append(col)
    score_like = list(dict.fromkeys(score_like))
    if len(score_like) == 1:
        return score_like[0]
    if len(score_like) > 1:
        raise ColumnDetectionError(
            f"Ambiguous AD probability/score column candidates: {score_like}\n"
            f"Available columns: {list(df.columns)}"
        )
    raise ColumnDetectionError(
        f"Could not detect AD probability/score column.\nAvailable columns: {list(df.columns)}"
    )


def detect_prediction_columns(df: pd.DataFrame) -> Dict[str, str]:
    return {
        "subject_id": detect_exact_column(
            df,
            "SubjectID",
            ["SubjectID", "subject_id", "subject", "PTID", "subjectid"],
            ("subject",),
        ),
        "fold": detect_exact_column(df, "fold", ["fold", "outer_fold", "test_fold"], ("fold",)),
        "classifier": detect_exact_column(
            df,
            "classifier",
            ["classifier", "classifier_type", "clf", "model", "model_name"],
            ("classifier",),
        ),
        "y_true": detect_exact_column(
            df,
            "y_true",
            ["y_true", "true_label", "label", "target", "diagnosis_binary"],
            ("true",),
        ),
        "y_score": detect_score_column(df),
        "y_pred": detect_exact_column(
            df,
            "y_pred",
            ["y_pred", "pred", "prediction", "predicted_label", "y_hat"],
            ("pred",),
        ),
    }


def to_binary_label(series: pd.Series, column_name: str) -> pd.Series:
    def convert(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, np.integer, float, np.floating)):
            if float(value) in (0.0, 1.0):
                return int(value)
        text = str(value).strip().upper()
        mapping = {
            "0": 0,
            "0.0": 0,
            "CN": 0,
            "CONTROL": 0,
            "CONTROLS": 0,
            "NORMAL": 0,
            "1": 1,
            "1.0": 1,
            "AD": 1,
            "DEMENTIA": 1,
            "ALZHEIMER": 1,
            "ALZHEIMER'S DISEASE": 1,
        }
        return mapping.get(text, np.nan)

    converted = series.map(convert)
    if converted.isna().any():
        bad_values = sorted(series[converted.isna()].dropna().astype(str).unique().tolist())
        raise ValueError(f"Could not map {column_name} to CN=0/AD=1. Bad values: {bad_values}")
    return converted.astype(int)


def normalize_subject_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def infer_site3(subject_id: object) -> Optional[str]:
    if pd.isna(subject_id):
        return None
    text = str(subject_id).strip()
    if len(text) >= 3 and text[:3].isdigit():
        return text[:3]
    parts = text.split("_")
    if parts and parts[0].isdigit():
        return parts[0].zfill(3)
    return None


def normalize_site3_value(value: object, subject_id: object) -> Optional[str]:
    inferred = infer_site3(subject_id)
    if pd.isna(value) or str(value).strip() == "":
        return inferred
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text.isdigit():
        return text.zfill(3)
    return text


def make_age_bin(age: object) -> str:
    try:
        age_f = float(age)
    except (TypeError, ValueError):
        return "Unknown"
    if np.isnan(age_f):
        return "Unknown"
    if age_f < 65:
        return "<65"
    if age_f <= 75:
        return "65-75"
    if age_f <= 85:
        return "75-85"
    return ">85"


def prepare_predictions(predictions: pd.DataFrame, columns: Dict[str, str]) -> pd.DataFrame:
    out = predictions.copy()
    rename_map = {
        columns["subject_id"]: "SubjectID",
        columns["fold"]: "fold",
        columns["classifier"]: "classifier",
        columns["y_true"]: "y_true",
        columns["y_score"]: "y_score_ad",
        columns["y_pred"]: "y_pred",
    }
    out = out.rename(columns=rename_map)
    out["SubjectID"] = normalize_subject_id(out["SubjectID"])
    out["classifier"] = out["classifier"].astype(str).str.strip()
    out["y_true"] = to_binary_label(out["y_true"], "y_true")
    out["y_pred"] = to_binary_label(out["y_pred"], "y_pred")
    out["y_score_ad"] = pd.to_numeric(out["y_score_ad"], errors="coerce")
    if out["y_score_ad"].isna().any():
        raise ValueError("AD score/probability column contains non-numeric or missing values.")
    if not np.isfinite(out["y_score_ad"]).all():
        raise ValueError("AD score/probability column contains non-finite values.")
    return out


def prepare_metadata(metadata: pd.DataFrame) -> pd.DataFrame:
    if "SubjectID" not in metadata.columns:
        raise ValueError(f"Metadata must contain SubjectID. Columns: {list(metadata.columns)}")
    out = metadata.copy()
    out["SubjectID"] = normalize_subject_id(out["SubjectID"])
    if out["SubjectID"].duplicated().any():
        n_dupes = int(out["SubjectID"].duplicated().sum())
        print(f"Warning: metadata has {n_dupes} duplicate SubjectID rows; keeping first.", file=sys.stderr)
        out = out.drop_duplicates("SubjectID", keep="first")
    for col in ["Manufacturer", "SourceCohort", "Sex", "Age", "Site3"]:
        if col not in out.columns:
            out[col] = np.nan
    out["Site3"] = [normalize_site3_value(site, sid) for site, sid in zip(out["Site3"], out["SubjectID"])]
    out["AgeBin"] = out["Age"].map(make_age_bin)
    return out


def merge_predictions_metadata(predictions: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    merged = predictions.merge(metadata, on="SubjectID", how="left", suffixes=("", "_metadata"))
    if merged["Manufacturer"].isna().any():
        missing = sorted(merged.loc[merged["Manufacturer"].isna(), "SubjectID"].unique().tolist())
        raise ValueError(
            f"Missing metadata for {len(missing)} predicted subjects. Examples: {missing[:20]}"
        )
    merged["Site3"] = [normalize_site3_value(site, sid) for site, sid in zip(merged["Site3"], merged["SubjectID"])]
    merged["AgeBin"] = merged["Age"].map(make_age_bin)
    return merged


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return np.nan
    return numerator / denominator


def compute_confusion(y_true: Sequence[int], y_pred: Sequence[int]) -> Tuple[int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def compute_metrics(df: pd.DataFrame, require_two_classes: bool = True) -> Dict[str, object]:
    y_true = df["y_true"].astype(int).to_numpy()
    y_pred = df["y_pred"].astype(int).to_numpy()
    y_score = df["y_score_ad"].astype(float).to_numpy()
    tn, fp, fn, tp = compute_confusion(y_true, y_pred)
    n_cn = int((y_true == 0).sum())
    n_ad = int((y_true == 1).sum())
    has_both = n_cn > 0 and n_ad > 0
    row: Dict[str, object] = {
        "n": int(len(df)),
        "n_CN": n_cn,
        "n_AD": n_ad,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }
    if require_two_classes and not has_both:
        row.update(
            {
                "roc_auc": np.nan,
                "pr_auc": np.nan,
                "accuracy": np.nan,
                "balanced_accuracy": np.nan,
                "sensitivity_AD": np.nan,
                "specificity_CN": np.nan,
                "f1": np.nan,
                "brier": np.nan,
            }
        )
        return row
    row.update(
        {
            "roc_auc": roc_auc_score(y_true, y_score) if has_both else np.nan,
            "pr_auc": average_precision_score(y_true, y_score) if has_both else np.nan,
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_accuracy_score(y_true, y_pred) if has_both else np.nan,
            "sensitivity_AD": safe_div(tp, tp + fn),
            "specificity_CN": safe_div(tn, tn + fp),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "brier": brier_score_loss(y_true, y_score),
        }
    )
    return row


def metrics_by_classifier(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for classifier, group in df.groupby("classifier", sort=True):
        row = {"classifier": classifier}
        row.update(compute_metrics(group, require_two_classes=False))
        rows.append(row)
    return pd.DataFrame(rows)


def metrics_by_stratum(df: pd.DataFrame, stratum_col: str) -> pd.DataFrame:
    rows = []
    for (classifier, stratum), group in df.groupby(["classifier", stratum_col], dropna=False, sort=True):
        row = {"classifier": classifier, stratum_col: "NA" if pd.isna(stratum) else stratum}
        row.update(compute_metrics(group, require_two_classes=True))
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, object]:
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = compute_confusion(y_true, y_pred)
    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {
        "threshold": float(threshold),
        "sensitivity_AD": sensitivity,
        "specificity_CN": specificity,
        "balanced_accuracy": np.nanmean([sensitivity, specificity]),
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def candidate_thresholds(scores: np.ndarray) -> np.ndarray:
    unique = np.unique(scores.astype(float))
    mids = (unique[:-1] + unique[1:]) / 2.0 if len(unique) > 1 else np.array([], dtype=float)
    values = np.concatenate([np.array([0.0, 0.5, 1.0]), unique, mids])
    values = values[np.isfinite(values)]
    values = np.clip(values, 0.0, 1.0)
    return np.unique(values)


def pick_best_threshold(table: pd.DataFrame, selector: str) -> Optional[pd.Series]:
    if table.empty:
        return None
    if selector == "youden":
        ranked = table.assign(youden=table["sensitivity_AD"] + table["specificity_CN"] - 1.0)
        ranked = ranked.sort_values(
            ["youden", "balanced_accuracy", "f1", "threshold"],
            ascending=[False, False, False, True],
        )
        return ranked.iloc[0]
    if selector == "balanced_accuracy":
        ranked = table.sort_values(
            ["balanced_accuracy", "f1", "threshold"],
            ascending=[False, False, True],
        )
        return ranked.iloc[0]
    if selector == "high_sensitivity":
        subset = table[table["sensitivity_AD"] >= 0.80]
        if subset.empty:
            return None
        ranked = subset.sort_values(
            ["specificity_CN", "balanced_accuracy", "f1", "threshold"],
            ascending=[False, False, False, False],
        )
        return ranked.iloc[0]
    if selector == "high_specificity":
        subset = table[table["specificity_CN"] >= 0.90]
        if subset.empty:
            return None
        ranked = subset.sort_values(
            ["sensitivity_AD", "balanced_accuracy", "f1", "threshold"],
            ascending=[False, False, False, True],
        )
        return ranked.iloc[0]
    raise ValueError(f"Unknown selector: {selector}")


def threshold_analysis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for classifier, group in df.groupby("classifier", sort=True):
        y_true = group["y_true"].astype(int).to_numpy()
        y_score = group["y_score_ad"].astype(float).to_numpy()
        n_cn = int((y_true == 0).sum())
        n_ad = int((y_true == 1).sum())
        threshold_rows = [threshold_metrics(y_true, y_score, thr) for thr in candidate_thresholds(y_score)]
        full = pd.DataFrame(threshold_rows)

        strategies = [
            ("threshold_0.5", pd.Series(threshold_metrics(y_true, y_score, 0.5))),
            ("youden", pick_best_threshold(full, "youden")),
            ("max_balanced_accuracy", pick_best_threshold(full, "balanced_accuracy")),
            ("high_sensitivity_AD_ge_0.80", pick_best_threshold(full, "high_sensitivity")),
            ("high_specificity_CN_ge_0.90", pick_best_threshold(full, "high_specificity")),
        ]
        for strategy, picked in strategies:
            if picked is None:
                row = {
                    "classifier": classifier,
                    "strategy": strategy,
                    "n": int(len(group)),
                    "n_CN": n_cn,
                    "n_AD": n_ad,
                    "threshold": np.nan,
                    "sensitivity_AD": np.nan,
                    "specificity_CN": np.nan,
                    "balanced_accuracy": np.nan,
                    "f1": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "tn": np.nan,
                    "fp": np.nan,
                    "fn": np.nan,
                    "tp": np.nan,
                }
            else:
                row = {
                    "classifier": classifier,
                    "strategy": strategy,
                    "n": int(len(group)),
                    "n_CN": n_cn,
                    "n_AD": n_ad,
                }
                for col in [
                    "threshold",
                    "sensitivity_AD",
                    "specificity_CN",
                    "balanced_accuracy",
                    "f1",
                    "precision",
                    "recall",
                    "tn",
                    "fp",
                    "fn",
                    "tp",
                ]:
                    row[col] = picked[col]
            rows.append(row)
    return pd.DataFrame(rows)


def make_error_table(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    if kind == "fp":
        subset = df[(df["y_true"] == 0) & (df["y_pred"] == 1)].copy()
    elif kind == "fn":
        subset = df[(df["y_true"] == 1) & (df["y_pred"] == 0)].copy()
    else:
        raise ValueError(kind)
    cols = [
        "SubjectID",
        "classifier",
        "fold",
        "y_score_ad",
        "y_true",
        "y_pred",
        "Manufacturer",
        "Site3",
        "SourceCohort",
        "Age",
        "Sex",
    ]
    present = [col for col in cols if col in subset.columns]
    return subset[present].sort_values(["classifier", "y_score_ad", "SubjectID"], ascending=[True, False, True])


def prepare_outdir(outdir: Path, overwrite: bool) -> None:
    if outdir.exists() and not outdir.is_dir():
        raise FileExistsError(f"Output path exists but is not a directory: {outdir}")
    if not outdir.exists():
        outdir.mkdir(parents=True, exist_ok=False)
        return
    existing = [outdir / name for name in OUTPUT_FILES if (outdir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing audit files. Use --overwrite if intentional:\n"
            + "\n".join(str(path) for path in existing)
        )


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def format_value(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}"
    return str(value)


def find_best_classifier(global_metrics: pd.DataFrame) -> pd.Series:
    valid = global_metrics.dropna(subset=["roc_auc"])
    if valid.empty:
        return global_metrics.iloc[0]
    return valid.sort_values(["roc_auc", "balanced_accuracy", "f1"], ascending=False).iloc[0]


def worst_strata(metrics: pd.DataFrame, stratum_col: str, classifier: str, top_n: int = 5) -> pd.DataFrame:
    subset = metrics[
        (metrics["classifier"] == classifier)
        & metrics["balanced_accuracy"].notna()
        & (metrics["n_CN"] > 0)
        & (metrics["n_AD"] > 0)
    ].copy()
    if subset.empty:
        return subset
    return subset.sort_values(["balanced_accuracy", "n"], ascending=[True, False]).head(top_n)


def recommendation_text(
    best: pd.Series,
    threshold_table: pd.DataFrame,
    worst_manufacturer: pd.DataFrame,
    worst_site: pd.DataFrame,
) -> List[str]:
    lines: List[str] = []
    sensitivity = best.get("sensitivity_AD", np.nan)
    specificity = best.get("specificity_CN", np.nan)
    if pd.notna(sensitivity) and sensitivity < 0.70:
        lines.append(
            "- La sensibilidad AD es baja para un uso de screening; conviene evaluar threshold tuning y revisar falsos negativos AD."
        )
    elif pd.notna(sensitivity) and sensitivity < 0.80:
        lines.append(
            "- La sensibilidad AD queda por debajo de 0.80; threshold tuning puede ser metodologicamente util si el objetivo prioriza detectar AD."
        )
    else:
        lines.append("- La sensibilidad AD no muestra una alerta fuerte con el umbral operativo actual.")

    if pd.notna(specificity) and specificity >= 0.90:
        lines.append("- La especificidad CN es alta; el modelo no parece clasificar masivamente CN como AD en pooled OOF.")
    else:
        lines.append("- La especificidad CN no es alta; revisar falsos positivos y posible sensibilidad a fabricante/sitio.")

    best_classifier = str(best["classifier"])
    candidate = threshold_table[
        (threshold_table["classifier"] == best_classifier)
        & (threshold_table["strategy"] == "max_balanced_accuracy")
    ]
    if not candidate.empty:
        tuned = candidate.iloc[0]
        current_ba = best.get("balanced_accuracy", np.nan)
        tuned_ba = tuned.get("balanced_accuracy", np.nan)
        if pd.notna(current_ba) and pd.notna(tuned_ba) and tuned_ba > current_ba + 0.02:
            lines.append(
                "- El umbral que maximiza balanced accuracy mejora mas de 0.02 puntos; reportar analisis de umbral es recomendable."
            )
        else:
            lines.append("- El ajuste de umbral no cambia sustancialmente la balanced accuracy global.")

    weak_strata = []
    if not worst_manufacturer.empty:
        weak_strata.append("fabricante")
    if not worst_site.empty:
        weak_strata.append("sitio")
    if weak_strata:
        lines.append(
            "- Hay heterogeneidad por "
            + " y ".join(weak_strata)
            + "; una ablation por fabricante/sitio y mas datos en celdas debiles son las siguientes pruebas mas informativas."
        )
    else:
        lines.append("- No hay suficientes estratos con ambas clases para aislar un peor fabricante o sitio.")
    return lines


def write_readme(
    outdir: Path,
    predictions_path: Path,
    metadata_path: Path,
    global_metrics: pd.DataFrame,
    metrics_manufacturer: pd.DataFrame,
    metrics_site: pd.DataFrame,
    threshold_table: pd.DataFrame,
    fp_table: pd.DataFrame,
    fn_table: pd.DataFrame,
) -> None:
    best = find_best_classifier(global_metrics)
    best_classifier = str(best["classifier"])
    worst_manufacturer = worst_strata(metrics_manufacturer, "Manufacturer", best_classifier, top_n=5)
    worst_site = worst_strata(metrics_site, "Site3", best_classifier, top_n=5)
    lines = [
        "# ADNI Expanded V3 Prediction Audit",
        "",
        "This audit summarizes pooled out-of-fold CN/AD predictions from the finished ADNI_expanded_v3_beta25_static3 run. It does not retrain or modify the model.",
        "",
        "## Inputs",
        f"- Predictions: `{predictions_path}`",
        f"- Metadata: `{metadata_path}`",
        "",
        "## Global Result",
        (
            f"- Best classifier by ROC-AUC: `{best_classifier}` "
            f"(ROC-AUC={format_value(best['roc_auc'])}, PR-AUC={format_value(best['pr_auc'])}, "
            f"balanced accuracy={format_value(best['balanced_accuracy'])}, "
            f"sensitivity_AD={format_value(best['sensitivity_AD'])}, "
            f"specificity_CN={format_value(best['specificity_CN'])})."
        ),
        f"- False positives CN->AD at run threshold: {len(fp_table)} rows across classifiers.",
        f"- False negatives AD->CN at run threshold: {len(fn_table)} rows across classifiers.",
        "",
        "## Interpretation",
    ]
    lines.extend(recommendation_text(best, threshold_table, worst_manufacturer, worst_site))
    lines.extend(["", "## Weak Manufacturer Strata"])
    if worst_manufacturer.empty:
        lines.append("- No manufacturer stratum with both CN and AD was available for ranking.")
    else:
        for _, row in worst_manufacturer.iterrows():
            lines.append(
                f"- {row['Manufacturer']} / {row['classifier']}: "
                f"n={int(row['n'])}, CN={int(row['n_CN'])}, AD={int(row['n_AD'])}, "
                f"balanced accuracy={format_value(row['balanced_accuracy'])}, "
                f"sensitivity_AD={format_value(row['sensitivity_AD'])}, "
                f"specificity_CN={format_value(row['specificity_CN'])}."
            )
    lines.extend(["", "## Weak Site3 Strata"])
    if worst_site.empty:
        lines.append("- No Site3 stratum with both CN and AD was available for ranking.")
    else:
        for _, row in worst_site.iterrows():
            lines.append(
                f"- Site3 {row['Site3']} / {row['classifier']}: "
                f"n={int(row['n'])}, CN={int(row['n_CN'])}, AD={int(row['n_AD'])}, "
                f"balanced accuracy={format_value(row['balanced_accuracy'])}, "
                f"sensitivity_AD={format_value(row['sensitivity_AD'])}, "
                f"specificity_CN={format_value(row['specificity_CN'])}."
            )
    lines.extend(
        [
            "",
            "## Output Tables",
            "- `global_metrics_by_classifier.csv`",
            "- `metrics_by_manufacturer.csv`",
            "- `metrics_by_site3.csv`",
            "- `metrics_by_sourcecohort.csv`",
            "- `metrics_by_sex.csv`",
            "- `metrics_by_agebin.csv`",
            "- `threshold_analysis.csv`",
            "- `error_table_false_positives.csv`",
            "- `error_table_false_negatives.csv`",
            "- `subject_level_predictions_with_metadata.csv`",
            "",
        ]
    )
    (outdir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    predictions_path = args.predictions.resolve()
    metadata_path = args.metadata.resolve()
    outdir = args.outdir.resolve()

    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    predictions_raw = pd.read_csv(predictions_path)
    metadata_raw = pd.read_csv(metadata_path)
    try:
        columns = detect_prediction_columns(predictions_raw)
    except ColumnDetectionError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    predictions = prepare_predictions(predictions_raw, columns)
    metadata = prepare_metadata(metadata_raw)
    merged = merge_predictions_metadata(predictions, metadata)

    prepare_outdir(outdir, overwrite=args.overwrite)

    global_metrics = metrics_by_classifier(merged)
    metrics_manufacturer = metrics_by_stratum(merged, "Manufacturer")
    metrics_site = metrics_by_stratum(merged, "Site3")
    metrics_source = metrics_by_stratum(merged, "SourceCohort")
    metrics_sex = metrics_by_stratum(merged, "Sex")
    metrics_agebin = metrics_by_stratum(merged, "AgeBin")
    thresholds = threshold_analysis(merged)
    false_positives = make_error_table(merged, "fp")
    false_negatives = make_error_table(merged, "fn")

    write_csv(global_metrics, outdir / "global_metrics_by_classifier.csv")
    write_csv(metrics_manufacturer, outdir / "metrics_by_manufacturer.csv")
    write_csv(metrics_site, outdir / "metrics_by_site3.csv")
    write_csv(metrics_source, outdir / "metrics_by_sourcecohort.csv")
    write_csv(metrics_sex, outdir / "metrics_by_sex.csv")
    write_csv(metrics_agebin, outdir / "metrics_by_agebin.csv")
    write_csv(thresholds, outdir / "threshold_analysis.csv")
    write_csv(false_positives, outdir / "error_table_false_positives.csv")
    write_csv(false_negatives, outdir / "error_table_false_negatives.csv")
    write_csv(merged, outdir / "subject_level_predictions_with_metadata.csv")
    write_readme(
        outdir=outdir,
        predictions_path=predictions_path,
        metadata_path=metadata_path,
        global_metrics=global_metrics,
        metrics_manufacturer=metrics_manufacturer,
        metrics_site=metrics_site,
        threshold_table=thresholds,
        fp_table=false_positives,
        fn_table=false_negatives,
    )

    print(f"Prediction audit written to: {outdir}")
    print("Detected columns:")
    for role, col in columns.items():
        print(f"  {role}: {col}")
    print("\nGlobal metrics:")
    print(global_metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
