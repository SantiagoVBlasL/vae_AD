#!/usr/bin/env python3
"""
Create Martin59 stress-test tables for original vs expanded ADNI models.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "runs" / "adni_expanded_v1_beta25_static3.json"
DEFAULT_ORIGINAL_PREDICTIONS = Path(
    "/media/diego/Datos/adni_expansion/MARTIN59/"
    "inference_outputs/Tables/martin59_predictions_with_metadata.csv"
)
DEFAULT_MARTIN59_METADATA = (
    PROJECT_ROOT
    / "data"
    / "OneDrive_1_27-4-2026"
    / "metadata_martin59"
    / "subject_metadata_martin59.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "original_vs_expanded_comparison"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FPR stress-test tables by classifier, Manufacturer, and Site3.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--original-predictions", type=Path, default=None)
    parser.add_argument("--expanded-predictions", type=Path, default=None)
    parser.add_argument("--martin59-metadata", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def first_existing(paths: Iterable[Optional[Path]]) -> Optional[Path]:
    for path in paths:
        if path is not None and path.exists():
            return path
    return None


def discover_expanded_predictions(config: Dict[str, object], explicit: Optional[Path]) -> Optional[Path]:
    if explicit is not None:
        return explicit if explicit.exists() else None

    stress_cfg = config.get("stress_tests", {}) if isinstance(config.get("stress_tests"), dict) else {}
    configured = resolve_path(stress_cfg.get("expanded_predictions_path"))
    if configured and configured.exists():
        return configured

    paths_cfg = config.get("paths", {}) if isinstance(config.get("paths"), dict) else {}
    output_dir = resolve_path(paths_cfg.get("output_dir"))
    candidates: List[Path] = []
    if output_dir is not None:
        candidates.extend(
            [
                output_dir / "inference_outputs" / "Tables" / "martin59_predictions_with_metadata.csv",
                output_dir / "Tables" / "martin59_predictions_with_metadata.csv",
                output_dir / "martin59_predictions_with_metadata.csv",
            ]
        )

    candidates.extend(
        sorted(
            (
                PROJECT_ROOT
                / "results"
                / "revision_bspc_2026"
            ).glob("**/martin59_predictions_with_metadata.csv")
        )
    )
    return first_existing(candidates)


def pick_column(df: pd.DataFrame, candidates: Sequence[str], label: str) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise RuntimeError(f"Cannot find {label} column. Tried: {list(candidates)}")


def load_predictions(path: Path, model_name: str, metadata_path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Predictions not found: {path}")
    df = pd.read_csv(path)
    if "SubjectID" not in df.columns:
        raise RuntimeError(f"{path} does not contain SubjectID.")

    classifier_col = pick_column(df, ["classifier", "classifier_type", "actual_classifier_type"], "classifier")
    score_col = pick_column(
        df,
        ["y_score_ensemble", "y_score_final", "y_score", "score", "ad_score", "p_ad"],
        "score",
    )
    pred_col = pick_column(
        df,
        ["y_pred_ensemble", "y_pred", "pred", "prediction", "y_pred_majority_vote"],
        "prediction",
    )

    out = df.copy()
    if classifier_col != "classifier":
        out["classifier"] = out[classifier_col]
    out["score_ad"] = pd.to_numeric(out[score_col], errors="coerce")
    out["pred_ad"] = pd.to_numeric(out[pred_col], errors="coerce").fillna(0).astype(int)
    out["SubjectID"] = out["SubjectID"].astype(str).str.strip()
    out["Model"] = model_name

    meta = pd.read_csv(metadata_path)
    meta["SubjectID"] = meta["SubjectID"].astype(str).str.strip()
    keep_cols = [c for c in ["SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Site3", "Age", "Sex"] if c in meta.columns]
    meta = meta[keep_cols].drop_duplicates("SubjectID", keep="first")

    for col in ["ResearchGroup_Mapped", "Manufacturer", "Site3", "Age", "Sex"]:
        if col in out.columns:
            out = out.drop(columns=[col])

    out = out.merge(meta, on="SubjectID", how="left", validate="many_to_one")
    if out["Manufacturer"].isna().any() or out["Site3"].isna().any():
        missing = out.loc[out["Manufacturer"].isna() | out["Site3"].isna(), "SubjectID"].unique().tolist()
        raise RuntimeError(f"Predictions have subjects missing Martin59 metadata: {missing[:10]}")

    return out


def summarize(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    for keys, sub in df.groupby(list(group_cols), dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = int(len(sub))
        n_pred_ad = int((sub["pred_ad"] == 1).sum())
        frac_pred_ad = float(n_pred_ad / n) if n else np.nan
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update(
            {
                "n": n,
                "n_pred_ad": n_pred_ad,
                "frac_pred_ad": frac_pred_ad,
                "FPR_CN": frac_pred_ad,  # alias kept for backward compatibility
                "mean_score": float(sub["score_ad"].mean()) if n else np.nan,
                "std_score": float(sub["score_ad"].std(ddof=1)) if n > 1 else 0.0,
            }
        )
        if "fold_std" in sub.columns:
            row["mean_fold_std"] = float(sub["fold_std"].mean()) if n else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(list(group_cols)).reset_index(drop=True)


def write_tables_for_model(df: pd.DataFrame, model_name: str, output_dir: Path) -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    specs: Dict[str, List[str]] = {
        "by_classifier": ["Model", "classifier"],
        "by_classifier_manufacturer": ["Model", "classifier", "Manufacturer"],
        "by_classifier_site3": ["Model", "classifier", "Site3"],
        "by_classifier_manufacturer_site3": ["Model", "classifier", "Manufacturer", "Site3"],
    }
    if "SourceCohort" in df.columns:
        specs["by_classifier_source_cohort"] = ["Model", "classifier", "SourceCohort"]

    for suffix, cols in specs.items():
        available = [c for c in cols if c in df.columns]
        if len(available) != len(cols):
            continue
        table = summarize(df, available)
        path = output_dir / f"{model_name}_{suffix}.csv"
        table.to_csv(path, index=False)
        outputs[f"{model_name}_{suffix}"] = str(path)
    return outputs


def make_delta_table(combined: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    table = summarize(combined, ["Model", *group_cols])
    pivot_cols = ["FPR_CN", "n", "n_pred_ad", "mean_score", "std_score"]
    index_cols = list(group_cols)
    wide = table.pivot_table(index=index_cols, columns="Model", values=pivot_cols, aggfunc="first")
    wide.columns = [f"{metric}_{model}" for metric, model in wide.columns]
    wide = wide.reset_index()
    if "FPR_CN_expanded" in wide.columns and "FPR_CN_original" in wide.columns:
        wide["delta_FPR_CN_expanded_minus_original"] = wide["FPR_CN_expanded"] - wide["FPR_CN_original"]
    if "mean_score_expanded" in wide.columns and "mean_score_original" in wide.columns:
        wide["delta_mean_score_expanded_minus_original"] = (
            wide["mean_score_expanded"] - wide["mean_score_original"]
        )
    return wide


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    stress_cfg = config.get("stress_tests", {}) if isinstance(config.get("stress_tests"), dict) else {}

    original_predictions = (
        args.original_predictions
        or resolve_path(stress_cfg.get("original_predictions_path"))
        or DEFAULT_ORIGINAL_PREDICTIONS
    )
    metadata_path = (
        args.martin59_metadata
        or resolve_path(stress_cfg.get("martin59_metadata_path"))
        or DEFAULT_MARTIN59_METADATA
    )
    expanded_predictions = discover_expanded_predictions(config, args.expanded_predictions)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    original = load_predictions(original_predictions, "original", metadata_path)
    outputs = write_tables_for_model(original, "original", args.output_dir)

    combined_frames = [original]
    if expanded_predictions is not None:
        expanded = load_predictions(expanded_predictions, "expanded", metadata_path)
        outputs.update(write_tables_for_model(expanded, "expanded", args.output_dir))
        combined_frames.append(expanded)

    combined = pd.concat(combined_frames, ignore_index=True)
    combined_path = args.output_dir / "martin59_predictions_original_vs_expanded_long.csv"
    combined.to_csv(combined_path, index=False)
    outputs["combined_long"] = str(combined_path)

    if expanded_predictions is not None:
        for suffix, cols in {
            "comparison_by_classifier": ["classifier"],
            "comparison_by_classifier_manufacturer": ["classifier", "Manufacturer"],
            "comparison_by_classifier_site3": ["classifier", "Site3"],
            "comparison_by_classifier_manufacturer_site3": ["classifier", "Manufacturer", "Site3"],
        }.items():
            table = make_delta_table(combined, cols)
            path = args.output_dir / f"{suffix}.csv"
            table.to_csv(path, index=False)
            outputs[suffix] = str(path)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "original_predictions": str(original_predictions),
        "expanded_predictions": str(expanded_predictions) if expanded_predictions else None,
        "martin59_metadata": str(metadata_path),
        "output_dir": str(args.output_dir),
        "outputs": outputs,
        "n_rows_original": int(len(original)),
        "n_rows_expanded": int(len(combined) - len(original)) if expanded_predictions else 0,
    }
    summary_path = args.output_dir / "stress_test_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    readme_lines = [
        "# Original vs Expanded Martin59 Stress Test",
        "",
        f"Original predictions: {original_predictions}",
        f"Expanded predictions: {expanded_predictions if expanded_predictions else 'not found'}",
        "",
        "Metrics:",
        "- FPR_CN = n_pred_ad / n for Martin59 CN subjects.",
        "- score_ad uses the available AD score/probability column.",
        "",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    print(f"Stress-test tables written to: {args.output_dir}")
    print(f"Original rows: {len(original)}")
    if expanded_predictions is None:
        print("Expanded Martin59 predictions not found; wrote original-only tables.")
    else:
        print(f"Expanded rows: {len(combined) - len(original)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
