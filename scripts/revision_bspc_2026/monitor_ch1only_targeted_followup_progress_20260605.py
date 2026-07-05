#!/usr/bin/env python3
"""Read-only progress monitor for ch1-only targeted FULL follow-up runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/ch1only_targeted_followup_progress_monitor_20260605"

REFERENCE_RUN = "recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5"
FOLLOWUP_RUNS = [
    "ch1only_latent256_beta3p75_T80_h10000_p560_full5x5",
    "ch1only_latent384_beta3p25_T80_h10000_p560_full5x5",
    "ch1only_latent256_beta3p25_T80_h10000_p560_full5x5",
    "ch1only_latent384_beta4p25_T80_h10000_p560_full5x5",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def write_table(df: pd.DataFrame, stem: str, max_rows: int = 250) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / f"{stem}.csv", index=False)
    view = df.head(max_rows)
    try:
        md = view.to_markdown(index=False)
    except Exception:
        md = view.to_string(index=False)
    if len(df) > max_rows:
        md += f"\n\n_Showing {max_rows} of {len(df)} rows._"
    (OUTPUT_DIR / f"{stem}.md").write_text(md + "\n", encoding="utf-8")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def run_path(run_id: str) -> Path:
    return PROJECT_ROOT / "results/revision_bspc_2026" / run_id


def resolve_path(path: Path) -> str:
    if path.exists() or path.is_symlink():
        try:
            return str(path.resolve())
        except Exception:
            return str(path)
    return ""


def latest_existing(paths: Iterable[Path]) -> Optional[Path]:
    candidates = [p for p in paths if p.exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_history(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        hist = joblib.load(path)
    except Exception:
        return None
    return hist if isinstance(hist, dict) else None


def history_metric(hist: Optional[Dict[str, Any]], key: str, idx: int = -1) -> Optional[float]:
    if not hist or key not in hist:
        return None
    values = hist.get(key)
    if not isinstance(values, (list, tuple, np.ndarray)) or len(values) == 0:
        return None
    try:
        return safe_float(values[idx])
    except Exception:
        return None


def history_len(hist: Optional[Dict[str, Any]]) -> Optional[int]:
    if not hist:
        return None
    for key in ["val_loss_modelsel", "val_loss", "train_loss"]:
        values = hist.get(key)
        if isinstance(values, (list, tuple, np.ndarray)):
            return len(values)
    return None


def best_epoch_from_history(hist: Optional[Dict[str, Any]]) -> Optional[int]:
    if not hist:
        return None
    key = "val_loss_modelsel" if "val_loss_modelsel" in hist else "val_loss"
    values = hist.get(key)
    if not isinstance(values, (list, tuple, np.ndarray)) or len(values) == 0:
        return None
    arr = np.asarray(values, dtype=float)
    if np.all(~np.isfinite(arr)):
        return None
    return int(np.nanargmin(arr) + 1)


def load_rate_distortion(fold_dir: Path, fold: int) -> Optional[pd.DataFrame]:
    path = fold_dir / f"fold_{fold}_rate_distortion.csv"
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def rd_row(rd: Optional[pd.DataFrame], epoch: Optional[int]) -> Optional[pd.Series]:
    if rd is None or rd.empty:
        return None
    if epoch is None or "epoch" not in rd.columns:
        return rd.iloc[-1]
    matches = rd[rd["epoch"] == epoch]
    if matches.empty:
        return rd.iloc[-1]
    return matches.iloc[-1]


def latent_info_summary(fold_dir: Path, fold: int, split: str = "test") -> Dict[str, Any]:
    path = fold_dir / f"fold_{fold}_{split}_latent_info_summary.csv"
    out: Dict[str, Any] = {
        f"{split}_active_units": None,
        f"{split}_frac_active": None,
        f"{split}_total_correlation_nats": None,
        f"{split}_mi_y_sum_nats": None,
        f"{split}_mi_manufacturer_sum_nats": None,
        f"{split}_mi_manufacturer_over_y": None,
    }
    if not path.exists():
        return out
    try:
        df = pd.read_csv(path)
    except Exception:
        return out
    if df.empty:
        return out
    first = df.iloc[0]
    out[f"{split}_active_units"] = safe_int(first.get("n_active"))
    out[f"{split}_frac_active"] = safe_float(first.get("frac_active"))
    out[f"{split}_total_correlation_nats"] = safe_float(first.get("total_correlation_nats"))
    y = df[df["variable"].astype(str) == "Y_target"]
    m = df[df["variable"].astype(str) == "Manufacturer"]
    y_val = safe_float(y.iloc[0].get("mi_sum_nats")) if not y.empty else None
    m_val = safe_float(m.iloc[0].get("mi_sum_nats")) if not m.empty else None
    out[f"{split}_mi_y_sum_nats"] = y_val
    out[f"{split}_mi_manufacturer_sum_nats"] = m_val
    out[f"{split}_mi_manufacturer_over_y"] = (m_val / y_val) if y_val and m_val is not None else None
    return out


def classifier_stage_a_metrics(run_dir: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    metrics_path = latest_existing(run_dir.glob("all_folds_metrics_MULTI_logreg_*.csv"))
    if metrics_path is None:
        return out
    try:
        df = pd.read_csv(metrics_path)
    except Exception:
        return out
    if "actual_classifier_type" in df.columns:
        df = df[df["actual_classifier_type"].astype(str) == "logreg"]
    for _, row in df.iterrows():
        fold = safe_int(row.get("fold"))
        if fold is None:
            continue
        out[fold] = {
            "stageA_logreg_auc_raw": safe_float(row.get("auc_raw")),
            "stageA_logreg_pr_auc_raw": safe_float(row.get("pr_auc_raw")),
            "stageA_logreg_auc_final": safe_float(row.get("auc_final", row.get("auc"))),
            "stageA_logreg_pr_auc_final": safe_float(row.get("pr_auc_final", row.get("pr_auc"))),
        }
    return out


def run_level_classifier_status(run_dir: Path) -> Dict[str, Any]:
    clf_dir = run_dir / "classifier_only_readout"
    pooled = clf_dir / "classifier_sweep_pooled_metrics.csv"
    foldwise = clf_dir / "classifier_sweep_foldwise_metrics.csv"
    return {
        "stageB_classifier_only_dir_exists": bool(clf_dir.exists()),
        "stageB_pooled_metrics_exists": bool(pooled.exists()),
        "stageB_foldwise_metrics_exists": bool(foldwise.exists()),
    }


def fold_snapshot(run_id: str, fold: int, is_reference: bool = False) -> Dict[str, Any]:
    rdir = run_path(run_id)
    fold_dir = rdir / f"fold_{fold}"
    hist_path = fold_dir / f"vae_train_history_fold_{fold}.joblib"
    hist = load_history(hist_path)
    current_epoch = history_len(hist)
    best_epoch = best_epoch_from_history(hist)
    rd = load_rate_distortion(fold_dir, fold)
    current_rd = rd_row(rd, current_epoch)
    best_rd = rd_row(rd, best_epoch)
    beta_current = history_metric(hist, "beta", -1)
    val_recon_current = history_metric(hist, "val_recon", -1)
    val_kld_current = history_metric(hist, "val_kld", -1)
    train_recon_current = history_metric(hist, "train_recon", -1)
    train_kld_current = history_metric(hist, "train_kld", -1)
    beta_max = None
    run_config = rdir / "run_config.json"
    if run_config.exists():
        try:
            cfg = json.loads(run_config.read_text(encoding="utf-8"))
            beta_max = safe_float(cfg.get("parameters", {}).get("beta_vae"))
        except Exception:
            beta_max = None
    if beta_max is None:
        beta_max = beta_current

    row: Dict[str, Any] = {
        "run_id": run_id,
        "is_reference": is_reference,
        "run_path": str(rdir),
        "run_resolved_path": resolve_path(rdir),
        "run_exists": bool(rdir.exists()),
        "fold": fold,
        "fold_dir_exists": bool(fold_dir.exists()),
        "vae_history_exists": bool(hist_path.exists()),
        "vae_current_epoch": current_epoch,
        "vae_best_epoch": best_epoch,
        "vae_model_exists": bool((fold_dir / f"vae_model_fold_{fold}.pt").exists()),
        "vae_norm_params_exists": bool((fold_dir / "vae_norm_params.joblib").exists()),
        "rate_distortion_exists": bool((fold_dir / f"fold_{fold}_rate_distortion.csv").exists()),
        "classifier_logreg_pipeline_exists": bool((fold_dir / f"classifier_logreg_final_pipeline_fold_{fold}.joblib").exists()),
        "classifier_logreg_predictions_exist": bool((fold_dir / "test_predictions_logreg.csv").exists()),
        "classifier_any_artifact_exists": any(
            [
                (fold_dir / f"classifier_logreg_final_pipeline_fold_{fold}.joblib").exists(),
                (fold_dir / "test_predictions_logreg.csv").exists(),
            ]
        ),
        "train_recon_current": train_recon_current,
        "val_recon_current": val_recon_current,
        "train_kld_nats_current": train_kld_current,
        "val_kld_nats_current": val_kld_current,
        "train_kld_bits_current": (train_kld_current / np.log(2)) if train_kld_current is not None else None,
        "val_kld_bits_current": (val_kld_current / np.log(2)) if val_kld_current is not None else None,
        "beta_current": beta_current,
        "beta_max": beta_max,
        "val_beta_schedule_kld_over_D_current": history_metric(hist, "val_beta_kld_over_recon", -1),
        "val_beta_max_kld_over_D_current": (
            beta_max * val_kld_current / val_recon_current
            if beta_max is not None and val_kld_current is not None and val_recon_current not in (None, 0)
            else None
        ),
        "best_train_recon": history_metric(hist, "train_recon", (best_epoch or 1) - 1) if best_epoch else None,
        "best_val_recon": history_metric(hist, "val_recon", (best_epoch or 1) - 1) if best_epoch else None,
        "best_val_loss_modelsel": history_metric(hist, "val_loss_modelsel", (best_epoch or 1) - 1) if best_epoch else None,
        "rd_current_D_train": safe_float(current_rd.get("D_train")) if current_rd is not None else None,
        "rd_current_D_val": safe_float(current_rd.get("D_val")) if current_rd is not None else None,
        "rd_current_R_val_bits": safe_float(current_rd.get("R_val_bits")) if current_rd is not None else None,
        "rd_current_L_val_betaMax": safe_float(current_rd.get("L_val_betaMax")) if current_rd is not None else None,
        "rd_best_D_val": safe_float(best_rd.get("D_val")) if best_rd is not None else None,
        "rd_best_R_val_bits": safe_float(best_rd.get("R_val_bits")) if best_rd is not None else None,
        "rd_best_L_val_betaMax": safe_float(best_rd.get("L_val_betaMax")) if best_rd is not None else None,
    }
    row.update(latent_info_summary(fold_dir, fold, "test"))
    row.update(latent_info_summary(fold_dir, fold, "trainDev"))
    return row


def add_stage_a(rows: List[Dict[str, Any]], run_id: str) -> None:
    metrics = classifier_stage_a_metrics(run_path(run_id))
    for row in rows:
        if row["run_id"] != run_id:
            continue
        row.update(
            metrics.get(
                row["fold"],
                {
                    "stageA_logreg_auc_raw": None,
                    "stageA_logreg_pr_auc_raw": None,
                    "stageA_logreg_auc_final": None,
                    "stageA_logreg_pr_auc_final": None,
                },
            )
        )


def build_progress_table() -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fold in range(1, 6):
        rows.append(fold_snapshot(REFERENCE_RUN, fold, is_reference=True))
    for run_id in FOLLOWUP_RUNS:
        for fold in range(1, 6):
            rows.append(fold_snapshot(run_id, fold, is_reference=False))
    for run_id in [REFERENCE_RUN] + FOLLOWUP_RUNS:
        add_stage_a(rows, run_id)
        status = run_level_classifier_status(run_path(run_id))
        for row in rows:
            if row["run_id"] == run_id:
                row.update(status)
    return pd.DataFrame(rows)


def build_reference_comparison(progress: pd.DataFrame) -> pd.DataFrame:
    ref = progress[progress["run_id"] == REFERENCE_RUN].set_index("fold")
    rows: List[Dict[str, Any]] = []
    compare_cols = [
        "vae_current_epoch",
        "vae_best_epoch",
        "train_recon_current",
        "val_recon_current",
        "val_kld_bits_current",
        "val_beta_max_kld_over_D_current",
        "test_active_units",
        "test_total_correlation_nats",
        "stageA_logreg_auc_final",
        "stageA_logreg_pr_auc_final",
    ]
    for _, row in progress[progress["run_id"] != REFERENCE_RUN].iterrows():
        fold = int(row["fold"])
        out = {
            "run_id": row["run_id"],
            "fold": fold,
            "run_exists": row["run_exists"],
            "vae_model_exists": row["vae_model_exists"],
            "classifier_any_artifact_exists": row["classifier_any_artifact_exists"],
        }
        if fold in ref.index:
            for col in compare_cols:
                out[f"{col}_candidate"] = row.get(col)
                out[f"{col}_reference"] = ref.loc[fold].get(col)
                c = safe_float(row.get(col))
                r = safe_float(ref.loc[fold].get(col))
                out[f"{col}_delta_vs_reference"] = (c - r) if c is not None and r is not None else None
        rows.append(out)
    return pd.DataFrame(rows)


def build_run_summary(progress: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for run_id, grp in progress.groupby("run_id", sort=False):
        rows.append(
            {
                "run_id": run_id,
                "is_reference": bool(grp["is_reference"].iloc[0]),
                "run_exists": bool(grp["run_exists"].any()),
                "fold_dirs_present": int(grp["fold_dir_exists"].sum()),
                "vae_histories_present": int(grp["vae_history_exists"].sum()),
                "vae_models_present": int(grp["vae_model_exists"].sum()),
                "rate_distortion_present": int(grp["rate_distortion_exists"].sum()),
                "stageA_logreg_fold_metrics_present": int(grp["stageA_logreg_auc_final"].notna().sum()),
                "classifier_fold_artifacts_present": int(grp["classifier_any_artifact_exists"].sum()),
                "stageB_classifier_only_dir_exists": bool(grp["stageB_classifier_only_dir_exists"].iloc[0]),
                "stageB_pooled_metrics_exists": bool(grp["stageB_pooled_metrics_exists"].iloc[0]),
                "stageB_foldwise_metrics_exists": bool(grp["stageB_foldwise_metrics_exists"].iloc[0]),
                "mean_current_epoch": safe_float(grp["vae_current_epoch"].dropna().mean()) if grp["vae_current_epoch"].notna().any() else None,
                "mean_best_epoch": safe_float(grp["vae_best_epoch"].dropna().mean()) if grp["vae_best_epoch"].notna().any() else None,
                "mean_val_recon_current": safe_float(grp["val_recon_current"].dropna().mean()) if grp["val_recon_current"].notna().any() else None,
                "mean_val_kld_bits_current": safe_float(grp["val_kld_bits_current"].dropna().mean()) if grp["val_kld_bits_current"].notna().any() else None,
                "mean_beta_max_kld_over_D_current": safe_float(grp["val_beta_max_kld_over_D_current"].dropna().mean()) if grp["val_beta_max_kld_over_D_current"].notna().any() else None,
                "mean_stageA_logreg_auc_final": safe_float(grp["stageA_logreg_auc_final"].dropna().mean()) if grp["stageA_logreg_auc_final"].notna().any() else None,
                "mean_stageA_logreg_pr_auc_final": safe_float(grp["stageA_logreg_pr_auc_final"].dropna().mean()) if grp["stageA_logreg_pr_auc_final"].notna().any() else None,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress = build_progress_table()
    comparison = build_reference_comparison(progress)
    summary = build_run_summary(progress)

    write_table(progress, "progress_by_fold")
    write_table(comparison, "reference_comparison_by_fold")
    write_table(summary, "run_completion_summary")

    readme = [
        "# ch1-only Targeted Follow-up Progress Monitor",
        "",
        f"Snapshot UTC: `{now_utc()}`",
        "",
        "This monitor is read-only with respect to training/model artifacts. It reads VAE histories, rate-distortion CSVs, latent-information summaries, and classifier artifacts from the current run directories.",
        "",
        "Tracked follow-up runs:",
    ]
    for run_id in FOLLOWUP_RUNS:
        readme.append(f"- `{run_id}`")
    readme.extend(
        [
            "",
            f"Reference: `{REFERENCE_RUN}`",
            "",
            "No training, killing, resuming, threshold fitting, or model-artifact modification was performed.",
        ]
    )
    (OUTPUT_DIR / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    write_json(
        OUTPUT_DIR / "command_log.json",
        {
            "created_utc": now_utc(),
            "reference_run": REFERENCE_RUN,
            "followup_runs": FOLLOWUP_RUNS,
            "outputs": [
                "progress_by_fold.csv/.md",
                "reference_comparison_by_fold.csv/.md",
                "run_completion_summary.csv/.md",
            ],
            "guardrails": [
                "no training",
                "no kill/resume",
                "no tensor modification",
                "no metadata modification",
                "no model artifact modification",
            ],
        },
    )
    print(f"Wrote monitor snapshot to {OUTPUT_DIR}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
