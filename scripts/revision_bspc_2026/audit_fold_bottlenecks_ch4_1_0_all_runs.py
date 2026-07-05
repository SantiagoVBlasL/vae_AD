#!/usr/bin/env python3
"""Fold-level bottleneck audit for ADNI V4 [4,1,0] runs.

This script reads existing CSV/joblib/QC artifacts only. It does not retrain,
load VAE checkpoints, or load the global tensor.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/auc_sprint_adni_only/fold_bottleneck_audit"
DEFAULT_METADATA = (
    PROJECT_ROOT
    / "data/revision_bspc_2026/adni_expanded_v4_all_available/subject_metadata_adni_expanded_v4_all_available.csv"
)

RUN_CANDIDATES = {
    "baseline_tanh": PROJECT_ROOT / "results/revision_bspc_2026/adni_expanded_v4_beta25_ch4_1_0",
    "linearout": PROJECT_ROOT / "results/revision_bspc_2026/auc_sprint_adni_only/adni_expanded_v4_beta25_ch4_1_0_linearout",
    "ckptselect": PROJECT_ROOT / "results/revision_bspc_2026/auc_sprint_adni_only/adni_expanded_v4_beta25_ch4_1_0_ckptselect",
    "mfrstrat": PROJECT_ROOT / "results/revision_bspc_2026/auc_sprint_adni_only/adni_expanded_v4_beta25_ch4_1_0_mfrstrat",
}

GENERATED_FILES = [
    "fold_bottleneck_table.csv",
    "fold_correlation_table.csv",
    "fold_ranked_issues.csv",
    "README.md",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    path = resolve(path)
    path.mkdir(parents=True, exist_ok=True)
    existing = [path / name for name in GENERATED_FILES if (path / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{path} already contains generated outputs; pass --overwrite")
    if overwrite:
        for file_path in existing:
            file_path.unlink()
    return path


def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, sort_keys=True)
    except Exception:
        return "{}"


def metrics_file(run_dir: Path) -> Optional[Path]:
    files = sorted(run_dir.glob("all_folds_metrics_MULTI*.csv"))
    return files[0] if files else None


def load_metrics(run_dir: Path) -> pd.DataFrame:
    path = metrics_file(run_dir)
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "actual_classifier_type" in df.columns:
        df["classifier"] = df["actual_classifier_type"].astype(str)
    elif "classifier" not in df.columns:
        df["classifier"] = "unknown"
    for col in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1_score"]:
        if col not in df.columns:
            df[col] = np.nan
    return df


def best_epoch_from_rate_distortion(fold_dir: Path, fold: int) -> Dict[str, Any]:
    path = fold_dir / f"fold_{fold}_rate_distortion.csv"
    df = safe_read_csv(path)
    if df is None or df.empty:
        return {
            "best_epoch": np.nan,
            "early_stop_epoch": np.nan,
            "train_recon_at_best": np.nan,
            "val_recon_at_best": np.nan,
            "recon_gap_at_best": np.nan,
            "kl_val_at_best": np.nan,
            "kl_train_at_best": np.nan,
        }
    metric_col = "L_val_betaMax" if "L_val_betaMax" in df.columns else None
    if metric_col is None:
        best_idx = int(df["epoch"].idxmax())
    else:
        best_idx = int(pd.to_numeric(df[metric_col], errors="coerce").idxmin())
    best = df.loc[best_idx]
    train_recon = float(best.get("D_train", np.nan))
    val_recon = float(best.get("D_val", np.nan))
    return {
        "best_epoch": int(best.get("epoch", np.nan)) if pd.notna(best.get("epoch", np.nan)) else np.nan,
        "early_stop_epoch": int(pd.to_numeric(df["epoch"], errors="coerce").max()),
        "train_recon_at_best": train_recon,
        "val_recon_at_best": val_recon,
        "recon_gap_at_best": val_recon - train_recon if np.isfinite(train_recon) and np.isfinite(val_recon) else np.nan,
        "kl_val_at_best": float(best.get("R_val_nats", np.nan)),
        "kl_train_at_best": float(best.get("R_train_nats", np.nan)),
    }


def checkpoint_manifest_epochs(fold_dir: Path, fold: int) -> Dict[str, Any]:
    path = fold_dir / f"vae_checkpoint_manifest_fold_{fold}.csv"
    df = safe_read_csv(path)
    if df is None or df.empty:
        return {}
    best_rows = df[df.get("checkpoint_type", pd.Series(dtype=str)).astype(str).eq("best")]
    terminal_rows = df[df.get("checkpoint_type", pd.Series(dtype=str)).astype(str).isin(["early_stop", "final"])]
    out: Dict[str, Any] = {}
    if not best_rows.empty:
        out["best_epoch"] = int(pd.to_numeric(best_rows["epoch"], errors="coerce").iloc[0])
        for src, dst in [
            ("train_recon", "train_recon_at_best"),
            ("val_recon", "val_recon_at_best"),
            ("val_kld", "kl_val_at_best"),
            ("train_kld", "kl_train_at_best"),
        ]:
            if src in best_rows.columns:
                out[dst] = float(pd.to_numeric(best_rows[src], errors="coerce").iloc[0])
        if "train_recon_at_best" in out and "val_recon_at_best" in out:
            out["recon_gap_at_best"] = out["val_recon_at_best"] - out["train_recon_at_best"]
    if not terminal_rows.empty:
        out["early_stop_epoch"] = int(pd.to_numeric(terminal_rows["epoch"], errors="coerce").max())
    return out


def latent_info(fold_dir: Path, fold: int) -> Dict[str, Any]:
    path = fold_dir / f"fold_{fold}_trainDev_latent_info_summary.csv"
    df = safe_read_csv(path)
    out: Dict[str, Any] = {
        "active_units": np.nan,
        "TC": np.nan,
        "MI_Y": np.nan,
        "MI_Manufacturer": np.nan,
        "MI_Manufacturer_over_MI_Y": np.nan,
    }
    if df is None or df.empty:
        return out
    if "n_active" in df.columns:
        out["active_units"] = float(pd.to_numeric(df["n_active"], errors="coerce").dropna().iloc[0])
    if "total_correlation_nats" in df.columns:
        out["TC"] = float(pd.to_numeric(df["total_correlation_nats"], errors="coerce").dropna().iloc[0])
    for variable, key in [("Y_target", "MI_Y"), ("Manufacturer", "MI_Manufacturer")]:
        row = df[df["variable"].astype(str).eq(variable)]
        if not row.empty:
            out[key] = float(pd.to_numeric(row["mi_sum_nats"], errors="coerce").iloc[0])
    if np.isfinite(out["MI_Y"]) and out["MI_Y"] != 0 and np.isfinite(out["MI_Manufacturer"]):
        out["MI_Manufacturer_over_MI_Y"] = out["MI_Manufacturer"] / out["MI_Y"]
    return out


def scanner_leakage(fold_dir: Path, fold: int) -> Dict[str, Any]:
    path = fold_dir / f"fold_{fold}_scanner_leakage_summary.csv"
    df = safe_read_csv(path)
    if df is None or df.empty:
        return {"acc_site_raw": np.nan, "acc_site_latent": np.nan}
    return {
        "acc_site_raw": float(pd.to_numeric(df.get("acc_site_raw"), errors="coerce").iloc[0]),
        "acc_site_latent": float(pd.to_numeric(df.get("acc_site_latent"), errors="coerce").iloc[0]),
    }


def test_demographics(fold_dir: Path, metadata: pd.DataFrame) -> Dict[str, Any]:
    path = fold_dir / "test_subjects_fold.csv"
    df = safe_read_csv(path)
    if df is None or df.empty:
        return {
            "manufacturer_counts_by_diagnosis_test": "{}",
            "age_mean_std_by_diagnosis_test": "{}",
            "sex_counts_by_diagnosis_test": "{}",
            "AD_age_minus_CN_age": np.nan,
            "manufacturer_imbalance": np.nan,
        }
    df["SubjectID"] = df["SubjectID"].astype(str)
    meta = metadata.copy()
    meta["SubjectID"] = meta["SubjectID"].astype(str)
    merged = df.merge(meta, on="SubjectID", how="left", suffixes=("", "_meta"))
    if "ResearchGroup_Mapped_meta" in merged.columns:
        merged["ResearchGroup_Mapped"] = merged["ResearchGroup_Mapped"].where(
            merged["ResearchGroup_Mapped"].notna(), merged["ResearchGroup_Mapped_meta"]
        )

    mfr_counts = {}
    if {"ResearchGroup_Mapped", "Manufacturer"}.issubset(merged.columns):
        mfr_counts = (
            merged.groupby(["ResearchGroup_Mapped", "Manufacturer"]).size().unstack(fill_value=0).to_dict(orient="index")
        )
    age_stats = {}
    age_delta = np.nan
    if {"ResearchGroup_Mapped", "Age"}.issubset(merged.columns):
        age = merged.copy()
        age["Age"] = pd.to_numeric(age["Age"], errors="coerce")
        stats = age.groupby("ResearchGroup_Mapped")["Age"].agg(["mean", "std", "count"])
        age_stats = stats.to_dict(orient="index")
        if "AD" in stats.index and "CN" in stats.index:
            age_delta = float(stats.loc["AD", "mean"] - stats.loc["CN", "mean"])
    sex_counts = {}
    if {"ResearchGroup_Mapped", "Sex"}.issubset(merged.columns):
        sex_counts = merged.groupby(["ResearchGroup_Mapped", "Sex"]).size().unstack(fill_value=0).to_dict(orient="index")

    manufacturer_imbalance = np.nan
    if mfr_counts and "AD" in mfr_counts and "CN" in mfr_counts:
        all_mfr = sorted(set(mfr_counts.get("AD", {})) | set(mfr_counts.get("CN", {})))
        ad_total = sum(mfr_counts.get("AD", {}).values())
        cn_total = sum(mfr_counts.get("CN", {}).values())
        diffs = []
        for mfr in all_mfr:
            ad_prop = mfr_counts.get("AD", {}).get(mfr, 0) / ad_total if ad_total else 0.0
            cn_prop = mfr_counts.get("CN", {}).get(mfr, 0) / cn_total if cn_total else 0.0
            diffs.append(abs(ad_prop - cn_prop))
        manufacturer_imbalance = max(diffs) if diffs else np.nan

    return {
        "manufacturer_counts_by_diagnosis_test": safe_json(mfr_counts),
        "age_mean_std_by_diagnosis_test": safe_json(age_stats),
        "sex_counts_by_diagnosis_test": safe_json(sex_counts),
        "AD_age_minus_CN_age": age_delta,
        "manufacturer_imbalance": manufacturer_imbalance,
    }


def fold_row(run_name: str, run_dir: Path, fold: int, classifier_row: pd.Series, metadata: pd.DataFrame) -> Dict[str, Any]:
    fold_dir = run_dir / f"fold_{fold}"
    rd = best_epoch_from_rate_distortion(fold_dir, fold)
    rd.update(checkpoint_manifest_epochs(fold_dir, fold))
    li = latent_info(fold_dir, fold)
    sl = scanner_leakage(fold_dir, fold)
    demo = test_demographics(fold_dir, metadata)
    row = {
        "run": run_name,
        "run_dir": str(run_dir),
        "fold": fold,
        "classifier": classifier_row.get("classifier", classifier_row.get("actual_classifier_type", "unknown")),
        "auc": classifier_row.get("auc", np.nan),
        "pr_auc": classifier_row.get("pr_auc", np.nan),
        "balanced_accuracy": classifier_row.get("balanced_accuracy", np.nan),
        "sensitivity": classifier_row.get("sensitivity", np.nan),
        "specificity": classifier_row.get("specificity", np.nan),
        "f1_score": classifier_row.get("f1_score", np.nan),
        "runtime_seconds": np.nan,
    }
    row.update(rd)
    row.update(li)
    row.update(sl)
    row.update(demo)
    return row


def correlation_table(df: pd.DataFrame) -> pd.DataFrame:
    variables = [
        "best_epoch",
        "early_stop_epoch",
        "recon_gap_at_best",
        "MI_Y",
        "MI_Manufacturer",
        "MI_Manufacturer_over_MI_Y",
        "acc_site_latent",
        "AD_age_minus_CN_age_abs",
        "manufacturer_imbalance",
        "TC",
    ]
    rows: List[Dict[str, Any]] = []
    work = df.copy()
    work["AD_age_minus_CN_age_abs"] = pd.to_numeric(work["AD_age_minus_CN_age"], errors="coerce").abs()
    for variable in variables:
        sub = work[["auc", variable]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) < 3 or sub[variable].nunique() < 2 or sub["auc"].nunique() < 2:
            pearson = np.nan
            spearman = np.nan
        else:
            pearson = float(sub["auc"].corr(sub[variable], method="pearson"))
            spearman = float(sub["auc"].corr(sub[variable], method="spearman"))
        rows.append(
            {
                "x": variable,
                "y": "auc",
                "n": int(len(sub)),
                "pearson_r": pearson,
                "spearman_r": spearman,
            }
        )
    return pd.DataFrame(rows)


def ranked_issues(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (run, classifier), group in df.groupby(["run", "classifier"], dropna=False):
        group = group.copy()
        specs = [
            ("best_diagnostic_fold", "auc", False),
            ("worst_diagnostic_fold", "auc", True),
            ("high_scanner_leakage_fold", "acc_site_latent", False),
            ("high_reconstruction_gap_fold", "recon_gap_at_best", False),
            ("high_manufacturer_imbalance_fold", "manufacturer_imbalance", False),
            ("high_AD_age_imbalance_fold", "AD_age_minus_CN_age", False),
        ]
        for issue, col, ascending in specs:
            if col not in group.columns:
                continue
            selected_cols = ["fold", col] if col == "auc" else ["fold", col, "auc"]
            sub = group[selected_cols].copy()
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
            if col == "AD_age_minus_CN_age":
                sub[col] = sub[col].abs()
            sub = sub.dropna(subset=[col])
            if sub.empty:
                continue
            chosen = sub.sort_values(col, ascending=ascending).iloc[0]
            rows.append(
                {
                    "run": run,
                    "classifier": classifier,
                    "issue_type": issue,
                    "fold": int(chosen["fold"]),
                    "value": float(chosen[col]),
                    "auc": float(chosen["auc"]) if pd.notna(chosen["auc"]) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def write_readme(outdir: Path, table: pd.DataFrame, corr: pd.DataFrame, ranked: pd.DataFrame) -> None:
    best = table.sort_values("auc", ascending=False).head(1)
    worst = table.sort_values("auc", ascending=True).head(1)
    high_leak = table.sort_values("acc_site_latent", ascending=False).head(1)
    lines = [
        "# Fold Bottleneck Audit",
        "",
        "Read-only fold audit across existing ADNI v4 [4,1,0] runs. No tensors, checkpoints, or training jobs were loaded.",
        "",
        f"- Rows: {len(table)}",
        f"- Runs included: {', '.join(sorted(table['run'].dropna().unique())) if not table.empty else 'none'}",
        "",
        "## Headline Folds",
        "",
    ]
    if not best.empty:
        r = best.iloc[0]
        lines.append(f"- Best diagnostic row: `{r['run']}` fold {int(r['fold'])} `{r['classifier']}` AUC={r['auc']:.4f}")
    if not worst.empty:
        r = worst.iloc[0]
        lines.append(f"- Worst diagnostic row: `{r['run']}` fold {int(r['fold'])} `{r['classifier']}` AUC={r['auc']:.4f}")
    if not high_leak.empty:
        r = high_leak.iloc[0]
        lines.append(f"- Highest latent scanner leakage: `{r['run']}` fold {int(r['fold'])}, acc_site_latent={r['acc_site_latent']:.4f}")
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- Correlations are exploratory and underpowered because folds are few and classifier rows are not independent.",
            "- Manufacturer/Site variables are used only for diagnostics, never as predictive features.",
            "- Reconstruction values come from existing rate-distortion/QC CSVs.",
            "",
            "## Outputs",
            "",
            "- `fold_bottleneck_table.csv`",
            "- `fold_correlation_table.csv`",
            "- `fold_ranked_issues.csv`",
        ]
    )
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    outdir = prepare_output_dir(args.output_dir, args.overwrite)
    metadata = pd.read_csv(resolve(args.metadata_path))
    rows: List[Dict[str, Any]] = []
    for run_name, run_dir in RUN_CANDIDATES.items():
        if not run_dir.exists():
            continue
        metrics = load_metrics(run_dir)
        if metrics.empty:
            continue
        for _, metric_row in metrics.iterrows():
            fold = int(metric_row["fold"])
            rows.append(fold_row(run_name, run_dir, fold, metric_row, metadata))
    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError("No fold metrics found for requested runs.")
    for col in [
        "auc",
        "pr_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1_score",
        "best_epoch",
        "early_stop_epoch",
        "recon_gap_at_best",
        "MI_Y",
        "MI_Manufacturer",
        "MI_Manufacturer_over_MI_Y",
        "acc_site_latent",
        "manufacturer_imbalance",
    ]:
        if col in table.columns:
            table[col] = pd.to_numeric(table[col], errors="coerce")
    corr = correlation_table(table)
    ranked = ranked_issues(table)
    table.to_csv(outdir / "fold_bottleneck_table.csv", index=False)
    corr.to_csv(outdir / "fold_correlation_table.csv", index=False)
    ranked.to_csv(outdir / "fold_ranked_issues.csv", index=False)
    write_readme(outdir, table, corr, ranked)
    print(table[["run", "fold", "classifier", "auc", "best_epoch", "early_stop_epoch", "recon_gap_at_best", "MI_Manufacturer_over_MI_Y", "acc_site_latent", "manufacturer_imbalance"]].to_string(index=False))
    print("\nCorrelations:")
    print(corr.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
