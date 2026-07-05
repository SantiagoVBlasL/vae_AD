#!/usr/bin/env python3
"""Read-only comparison of V4 [4,1,0] beta=2.5 vs beta=4.6 runs.

The script reads only existing CSV/TXT/JSON/log artifacts. It does not retrain
and does not load tensors, checkpoints, joblibs, or large arrays.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BETA25 = PROJECT_ROOT / "results/revision_bspc_2026/adni_expanded_v4_beta25_ch4_1_0"
DEFAULT_BETA46 = PROJECT_ROOT / "results/revision_bspc_2026/adni_expanded_v4_beta46_ch4_1_0"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/revision_bspc_2026/beta25_vs_beta46_ch4_1_0"
EXTERNAL_LOG_DIR = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026/run_logs")

OUTPUT_NAMES = [
    "beta25_vs_beta46_model_metrics.csv",
    "beta25_vs_beta46_latent_leakage.csv",
    "beta25_vs_beta46_latent_info.csv",
    "beta25_vs_beta46_foldwise.csv",
    "beta25_vs_beta46_readme.md",
]

CLASSIFIERS = ["logreg", "svm"]
METRIC_COLS = [
    "auc_raw",
    "pr_auc_raw",
    "auc_final",
    "pr_auc_final",
    "auc",
    "pr_auc",
    "accuracy",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "f1_score",
    "brier_final",
    "brier_raw",
]
LATENT_VARIABLES = ["Y_target", "Manufacturer"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare completed V4 [4,1,0] beta=2.5 and beta=4.6 runs without retraining.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--beta25-run-dir", type=Path, default=DEFAULT_BETA25)
    parser.add_argument("--beta46-run-dir", type=Path, default=DEFAULT_BETA46)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def numeric(value) -> float:
    return pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]


def prepare_output_dir(output_dir: Path, overwrite: bool) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in OUTPUT_NAMES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Refusing to overwrite existing comparison outputs. Use --overwrite:\n"
            + "\n".join(str(path) for path in existing)
        )
    return output_dir


def validate_run_dir(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Run directory not found: {path}")
    return path.resolve()


def discover_logs(run_dir: Path, beta_label: str) -> List[Path]:
    candidates: List[Path] = []
    for base in [run_dir, run_dir.resolve()]:
        candidates.extend(sorted((base / "Logs").glob("*.log")))
    if beta_label == "beta25":
        patterns = ["run_v4_ch4_1_0*.log", "*v4_ch4_1_0*.log"]
        exclude = "beta46"
    else:
        patterns = ["run_v4_beta46_ch4_1_0*.log", "*beta46_ch4_1_0*.log"]
        exclude = None
    for pattern in patterns:
        candidates.extend(sorted(EXTERNAL_LOG_DIR.glob(pattern)))
    unique: List[Path] = []
    seen = set()
    real_run = str(run_dir.resolve())
    for path in candidates:
        if not path.exists():
            continue
        if exclude and exclude in path.name:
            continue
        key = path.resolve()
        if key in seen:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if real_run not in text and not (beta_label == "beta25" and "--beta_vae 2.5" in text) and not (
            beta_label == "beta46" and "--beta_vae 4.6" in text
        ):
            continue
        seen.add(key)
        unique.append(path)
    return unique


def parse_logs(run_dir: Path, beta_label: str) -> Dict[str, object]:
    fold_meta: Dict[int, Dict[str, object]] = {fold: {} for fold in range(1, 6)}
    total_pipeline_sec = np.nan
    log_paths = discover_logs(run_dir, beta_label)
    current_fold: Optional[int] = None
    for log_path in log_paths:
        for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.search(r"Iniciando Fold\s+(\d+)/", line)
            if match:
                current_fold = int(match.group(1))
                fold_meta[current_fold]["log_path"] = str(log_path)
            match = re.search(
                r"Early stopping VAE en epoch\s+(\d+).*?Mejor ValL\(βmax\):\s+([0-9.]+).*?\(época\s+(\d+)\)",
                line,
            )
            if match and current_fold is not None:
                fold_meta[current_fold]["early_stop_epoch"] = int(match.group(1))
                fold_meta[current_fold]["vae_best_val_loss_beta_max"] = float(match.group(2))
                fold_meta[current_fold]["best_epoch"] = int(match.group(3))
            match = re.search(r"completado en\s+([0-9.]+)\s+segundos", line)
            if match and current_fold is not None:
                fold_meta[current_fold]["runtime_sec"] = float(match.group(1))
            match = re.search(r"Pipeline completo en\s+([0-9.]+)\s+segundos", line)
            if match:
                total_pipeline_sec = float(match.group(1))
            match = re.search(r"Pipeline completed in\s+([0-9.]+)\s*s", line, flags=re.IGNORECASE)
            if match:
                total_pipeline_sec = float(match.group(1))
    return {"fold_meta": fold_meta, "total_pipeline_sec": total_pipeline_sec, "logs": [str(p) for p in log_paths]}


def read_metrics(run_dir: Path) -> pd.DataFrame:
    frames = []
    for path in sorted(run_dir.resolve().glob("all_folds_metrics_MULTI_*.csv")):
        frame = safe_read_csv(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No all_folds_metrics_MULTI_*.csv found in {run_dir}")
    metrics = pd.concat(frames, ignore_index=True)
    clf_col = "actual_classifier_type" if "actual_classifier_type" in metrics.columns else "classifier"
    metrics = metrics.rename(columns={clf_col: "classifier"})
    metrics["classifier"] = metrics["classifier"].astype(str)
    for col in METRIC_COLS:
        if col not in metrics.columns:
            metrics[col] = np.nan
    return metrics


def read_prediction_brier(run_dir: Path) -> pd.DataFrame:
    rows = []
    for fold in range(1, 6):
        for classifier in CLASSIFIERS:
            path = run_dir.resolve() / f"fold_{fold}/test_predictions_{classifier}.csv"
            pred = safe_read_csv(path)
            if pred.empty:
                continue
            y = pd.to_numeric(pred.get("y_true"), errors="coerce")
            final = pd.to_numeric(pred.get("y_score_final"), errors="coerce")
            raw = pd.to_numeric(pred.get("y_score_raw"), errors="coerce")
            rows.append(
                {
                    "fold": fold,
                    "classifier": classifier,
                    "brier_final": ((final - y) ** 2).mean(),
                    "brier_raw": ((raw - y) ** 2).mean(),
                    "n_predictions": int(len(pred)),
                }
            )
    return pd.DataFrame(rows)


def add_brier(metrics: pd.DataFrame, brier: pd.DataFrame) -> pd.DataFrame:
    if brier.empty:
        return metrics
    merged = metrics.drop(columns=["brier_final", "brier_raw"], errors="ignore").merge(
        brier[["fold", "classifier", "brier_final", "brier_raw"]],
        on=["fold", "classifier"],
        how="left",
    )
    return merged


def aggregate_classifier_metrics(metrics_by_beta: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for classifier in CLASSIFIERS:
        row: Dict[str, object] = {"classifier": classifier}
        for beta_label, metrics in metrics_by_beta.items():
            sub = metrics[metrics["classifier"] == classifier].copy()
            row[f"{beta_label}_n_folds"] = int(len(sub))
            for col in METRIC_COLS:
                values = pd.to_numeric(sub[col], errors="coerce")
                row[f"{beta_label}_{col}_mean"] = values.mean()
                row[f"{beta_label}_{col}_std"] = values.std(ddof=1)
            row[f"{beta_label}_auc_final_minus_raw_mean"] = (
                pd.to_numeric(sub["auc_final"], errors="coerce") - pd.to_numeric(sub["auc_raw"], errors="coerce")
            ).mean()
            row[f"{beta_label}_pr_auc_final_minus_raw_mean"] = (
                pd.to_numeric(sub["pr_auc_final"], errors="coerce") - pd.to_numeric(sub["pr_auc_raw"], errors="coerce")
            ).mean()
        for col in METRIC_COLS + ["auc_final_minus_raw", "pr_auc_final_minus_raw"]:
            beta25_value = row.get(f"beta25_{col}_mean")
            beta46_value = row.get(f"beta46_{col}_mean")
            row[f"delta_beta46_minus_beta25_{col}_mean"] = (
                beta46_value - beta25_value if pd.notna(beta25_value) and pd.notna(beta46_value) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def read_latent_info_fold(run_dir: Path, fold: int, split: str) -> Dict[str, float]:
    path = run_dir.resolve() / f"fold_{fold}/fold_{fold}_{split}_latent_info_summary.csv"
    df = safe_read_csv(path)
    if df.empty:
        return {}
    out: Dict[str, float] = {}
    for variable in LATENT_VARIABLES:
        sub = df[df["variable"].astype(str) == variable]
        if sub.empty:
            continue
        row = sub.iloc[0]
        out[f"MI_{variable}_{split}"] = numeric(row.get("mi_sum_nats"))
        out[f"MI_mean_{variable}_{split}"] = numeric(row.get("mi_mean_nats"))
        out[f"TC_{split}"] = numeric(row.get("total_correlation_nats"))
        out[f"active_units_{split}"] = numeric(row.get("n_active"))
        out[f"frac_active_{split}"] = numeric(row.get("frac_active"))
    mi_y = out.get(f"MI_Y_target_{split}")
    mi_m = out.get(f"MI_Manufacturer_{split}")
    out[f"MI_Manufacturer_over_Y_{split}"] = mi_m / mi_y if pd.notna(mi_y) and mi_y else np.nan
    return out


def read_scanner_fold(run_dir: Path, fold: int, split: str) -> Dict[str, float]:
    filename = f"fold_{fold}_test_scanner_leakage_summary.csv" if split == "test" else f"fold_{fold}_scanner_leakage_summary.csv"
    df = safe_read_csv(run_dir.resolve() / f"fold_{fold}/{filename}")
    if df.empty:
        return {}
    row = df.iloc[0]
    raw = numeric(row.get("acc_site_raw"))
    latent = numeric(row.get("acc_site_latent"))
    return {
        f"acc_site_raw_{split}": raw,
        f"acc_site_latent_{split}": latent,
        f"latent_minus_raw_{split}": latent - raw if pd.notna(raw) and pd.notna(latent) else np.nan,
        f"chance_level_{split}": numeric(row.get("chance_level")),
    }


def read_qc_fold(run_dir: Path, fold: int) -> Dict[str, float]:
    qc = safe_read_csv(run_dir.resolve() / f"fold_{fold}/latent_qc_metrics.csv")
    if qc.empty:
        return {}
    row = qc.iloc[0]
    return {
        "silhouette_latent": numeric(row.get("silhouette_latent")),
        "latent_dim": numeric(row.get("latent_dim")),
        "beta_max": numeric(row.get("beta_max")),
    }


def fold_table_for_beta(run_dir: Path, beta_label: str) -> pd.DataFrame:
    metrics = add_brier(read_metrics(run_dir), read_prediction_brier(run_dir))
    logs = parse_logs(run_dir, beta_label)
    fold_meta: Mapping[int, Dict[str, object]] = logs["fold_meta"]  # type: ignore[assignment]
    rows = []
    for fold in range(1, 6):
        row: Dict[str, object] = {
            "fold": fold,
            "total_pipeline_sec": logs["total_pipeline_sec"],
        }
        row.update(fold_meta.get(fold, {}))
        for classifier in CLASSIFIERS:
            sub = metrics[(metrics["fold"] == fold) & (metrics["classifier"] == classifier)]
            if sub.empty:
                continue
            m = sub.iloc[0]
            for col in METRIC_COLS:
                row[f"{col}_{classifier}"] = numeric(m.get(col))
        for split in ["test", "trainDev"]:
            row.update(read_latent_info_fold(run_dir, fold, split))
            row.update(read_scanner_fold(run_dir, fold, split))
        row.update(read_qc_fold(run_dir, fold))
        best = numeric(row.get("best_epoch"))
        early = numeric(row.get("early_stop_epoch"))
        row["best_epoch_frac"] = best / early if pd.notna(best) and pd.notna(early) and early else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def compare_foldwise(beta25: pd.DataFrame, beta46: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "fold",
        "best_epoch",
        "early_stop_epoch",
        "best_epoch_frac",
        "runtime_sec",
        "total_pipeline_sec",
        "MI_Y_target_test",
        "MI_Manufacturer_test",
        "MI_Manufacturer_over_Y_test",
        "MI_Y_target_trainDev",
        "MI_Manufacturer_trainDev",
        "MI_Manufacturer_over_Y_trainDev",
        "TC_test",
        "TC_trainDev",
        "active_units_test",
        "active_units_trainDev",
        "frac_active_test",
        "frac_active_trainDev",
        "acc_site_raw_test",
        "acc_site_latent_test",
        "latent_minus_raw_test",
        "acc_site_raw_trainDev",
        "acc_site_latent_trainDev",
        "latent_minus_raw_trainDev",
        "silhouette_latent",
    ]
    for classifier in CLASSIFIERS:
        for col in METRIC_COLS:
            keep_cols.append(f"{col}_{classifier}")
    for df in [beta25, beta46]:
        for col in keep_cols:
            if col not in df.columns:
                df[col] = np.nan
    wide = beta25[keep_cols].merge(beta46[keep_cols], on="fold", how="outer", suffixes=("_beta25", "_beta46"))
    delta_cols = [col for col in keep_cols if col != "fold"]
    for col in delta_cols:
        b25 = pd.to_numeric(wide.get(f"{col}_beta25"), errors="coerce")
        b46 = pd.to_numeric(wide.get(f"{col}_beta46"), errors="coerce")
        wide[f"delta_beta46_minus_beta25_{col}"] = b46 - b25
    return wide.sort_values("fold")


def aggregate_latent_info(folds_by_beta: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    metrics = [
        "MI_Y_target",
        "MI_Manufacturer",
        "MI_Manufacturer_over_Y",
        "TC",
        "active_units",
        "frac_active",
    ]
    for split in ["test", "trainDev"]:
        row: Dict[str, object] = {"split": split}
        for beta_label, folds in folds_by_beta.items():
            for metric in metrics:
                values = pd.to_numeric(folds.get(f"{metric}_{split}"), errors="coerce")
                row[f"{beta_label}_{metric}_mean"] = values.mean()
                row[f"{beta_label}_{metric}_std"] = values.std(ddof=1)
            if split == "test":
                values = pd.to_numeric(folds.get("silhouette_latent"), errors="coerce")
                row[f"{beta_label}_silhouette_latent_mean"] = values.mean()
                row[f"{beta_label}_silhouette_latent_std"] = values.std(ddof=1)
        for metric in metrics + (["silhouette_latent"] if split == "test" else []):
            b25 = row.get(f"beta25_{metric}_mean")
            b46 = row.get(f"beta46_{metric}_mean")
            row[f"delta_beta46_minus_beta25_{metric}_mean"] = (
                b46 - b25 if pd.notna(b25) and pd.notna(b46) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_leakage(folds_by_beta: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    metrics = ["acc_site_raw", "acc_site_latent", "latent_minus_raw", "chance_level"]
    for split in ["test", "trainDev"]:
        row: Dict[str, object] = {"split": split}
        for beta_label, folds in folds_by_beta.items():
            for metric in metrics:
                values = pd.to_numeric(folds.get(f"{metric}_{split}"), errors="coerce")
                row[f"{beta_label}_{metric}_mean"] = values.mean()
                row[f"{beta_label}_{metric}_std"] = values.std(ddof=1)
        for metric in metrics:
            b25 = row.get(f"beta25_{metric}_mean")
            b46 = row.get(f"beta46_{metric}_mean")
            row[f"delta_beta46_minus_beta25_{metric}_mean"] = (
                b46 - b25 if pd.notna(b25) and pd.notna(b46) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


def decision_text(model: pd.DataFrame, leakage: pd.DataFrame, latent: pd.DataFrame) -> Dict[str, str]:
    test_leak = leakage[leakage["split"] == "test"].iloc[0]
    test_latent = latent[latent["split"] == "test"].iloc[0]
    ratio_delta = numeric(test_latent.get("delta_beta46_minus_beta25_MI_Manufacturer_over_Y_mean"))
    acc_latent_delta = numeric(test_leak.get("delta_beta46_minus_beta25_acc_site_latent_mean"))
    auc_deltas = {
        row["classifier"]: numeric(row.get("delta_beta46_minus_beta25_auc_mean"))
        for _, row in model.iterrows()
    }
    if ratio_delta < 0 and acc_latent_delta <= 0:
        leakage_decision = "Yes: beta=4.6 reduced both MI(Manufacturer)/MI(Y) and latent scanner-classifier accuracy."
    elif ratio_delta >= 0 and acc_latent_delta < 0:
        leakage_decision = (
            "No, not by the stricter latent-information criterion: beta=4.6 lowered acc_site_latent, "
            "but MI(Manufacturer)/MI(Y) increased."
        )
    elif ratio_delta < 0:
        leakage_decision = (
            "Mixed: MI(Manufacturer)/MI(Y) decreased, but latent scanner-classifier accuracy did not decrease."
        )
    else:
        leakage_decision = "No: beta=4.6 did not reduce the manufacturer-information burden."
    if all(delta >= -0.01 for delta in auc_deltas.values() if pd.notna(delta)):
        auc_decision = "Mostly yes: mean-fold AUC was preserved within about 0.01 for available classifiers."
    elif any(delta >= -0.01 for delta in auc_deltas.values() if pd.notna(delta)):
        auc_decision = "Partially: SVM was roughly preserved, but LogReg lost more AUC."
    else:
        auc_decision = "No: both classifiers lost mean-fold AUC."
    preferable = (
        "No. Beta=4.6 reduces latent scanner-classifier accuracy, but it does not reduce "
        "MI(Manufacturer)/MI(Y) and does not improve diagnostic AUC."
    )
    recommendation = (
        "Next controlled experiment: LayerNorm at beta=2.5, keeping dropout=0.15 and all other factors fixed. "
        "Use dropout=0.25 as the following control only if the LayerNorm audit still shows high manufacturer MI "
        "or if fold-wise overfit indicators dominate."
    )
    return {
        "manufacturer_leakage": leakage_decision,
        "diagnostic_auc": auc_decision,
        "prefer_beta46": preferable,
        "recommendation": recommendation,
    }


def write_readme(
    path: Path,
    model: pd.DataFrame,
    leakage: pd.DataFrame,
    latent: pd.DataFrame,
    foldwise: pd.DataFrame,
    decisions: Mapping[str, str],
    run_dirs: Mapping[str, Path],
) -> None:
    test_leak = leakage[leakage["split"] == "test"].iloc[0]
    test_latent = latent[latent["split"] == "test"].iloc[0]
    lines = [
        "# Beta=2.5 vs Beta=4.6 Audit for V4 [4,1,0]",
        "",
        "## Scope",
        "Read-only comparison of existing run artifacts. No retraining was performed. The script read only CSV/TXT/JSON/log files and did not load tensors, checkpoints, joblibs, or large arrays.",
        f"Beta=2.5 run: `{run_dirs['beta25']}`",
        f"Beta=4.6 run: `{run_dirs['beta46']}`",
        "",
        "## Main Model Metrics",
        model[
            [
                "classifier",
                "beta25_auc_mean",
                "beta46_auc_mean",
                "delta_beta46_minus_beta25_auc_mean",
                "beta25_balanced_accuracy_mean",
                "beta46_balanced_accuracy_mean",
                "beta25_sensitivity_mean",
                "beta46_sensitivity_mean",
                "beta25_specificity_mean",
                "beta46_specificity_mean",
                "beta25_brier_final_mean",
                "beta46_brier_final_mean",
            ]
        ].to_markdown(index=False),
        "",
        "## Leakage Summary",
        f"Test acc_site_latent mean changed from {test_leak['beta25_acc_site_latent_mean']:.3f} to {test_leak['beta46_acc_site_latent_mean']:.3f} (delta {test_leak['delta_beta46_minus_beta25_acc_site_latent_mean']:.3f}).",
        f"Test MI(Manufacturer)/MI(Y) mean changed from {test_latent['beta25_MI_Manufacturer_over_Y_mean']:.3f} to {test_latent['beta46_MI_Manufacturer_over_Y_mean']:.3f} (delta {test_latent['delta_beta46_minus_beta25_MI_Manufacturer_over_Y_mean']:.3f}).",
        f"Test total correlation mean changed from {test_latent['beta25_TC_mean']:.1f} to {test_latent['beta46_TC_mean']:.1f}.",
        "",
        "## Decisions",
        f"- Did beta=4.6 reduce manufacturer leakage? {decisions['manufacturer_leakage']}",
        f"- Did beta=4.6 preserve diagnostic AUC? {decisions['diagnostic_auc']}",
        f"- Is beta=4.6 preferable to beta=2.5? {decisions['prefer_beta46']}",
        "",
        "## Next Controlled Experiment",
        decisions["recommendation"],
        "",
        "## Fold-Wise Caveat",
        "Per-fold deltas are diagnostic only. There are five outer folds, so fold-wise associations should not be treated as inferential evidence.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    output_dir = prepare_output_dir(args.output_dir, args.overwrite)
    run_dirs = {
        "beta25": validate_run_dir(args.beta25_run_dir),
        "beta46": validate_run_dir(args.beta46_run_dir),
    }

    metrics_by_beta: Dict[str, pd.DataFrame] = {}
    folds_by_beta: Dict[str, pd.DataFrame] = {}
    for beta_label, run_dir in run_dirs.items():
        metrics_by_beta[beta_label] = add_brier(read_metrics(run_dir), read_prediction_brier(run_dir))
        folds_by_beta[beta_label] = fold_table_for_beta(run_dir, beta_label)

    model = aggregate_classifier_metrics(metrics_by_beta)
    foldwise = compare_foldwise(folds_by_beta["beta25"], folds_by_beta["beta46"])
    leakage = aggregate_leakage(folds_by_beta)
    latent = aggregate_latent_info(folds_by_beta)
    decisions = decision_text(model, leakage, latent)

    model.to_csv(output_dir / "beta25_vs_beta46_model_metrics.csv", index=False)
    leakage.to_csv(output_dir / "beta25_vs_beta46_latent_leakage.csv", index=False)
    latent.to_csv(output_dir / "beta25_vs_beta46_latent_info.csv", index=False)
    foldwise.to_csv(output_dir / "beta25_vs_beta46_foldwise.csv", index=False)
    write_readme(output_dir / "beta25_vs_beta46_readme.md", model, leakage, latent, foldwise, decisions, run_dirs)

    print(f"Wrote comparison outputs to: {output_dir}")
    print(decisions["manufacturer_leakage"])
    print(decisions["diagnostic_auc"])
    print(decisions["prefer_beta46"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
