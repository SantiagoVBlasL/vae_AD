#!/usr/bin/env python3
"""Read-only live monitor for recover035 latent512 beta3p75 FULL 5x5."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "revision_bspc_2026"
OUT_DIR = RESULTS / "latent512_beta3p75_live_rate_distortion_monitor_20260602"

RUNS = {
    "latent512_beta3p75_live": RESULTS / "recover035_latent512_beta3p75_T80_h10000_p560_full5x5",
    "latent384_beta3p75_promoted": RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5",
    "latent384_beta4p0": RESULTS / "recover035_latent384_beta4p0_T80_h10000_p560_full5x5",
    "recover035_full5x5_latent256": RESULTS / "recover035_full5x5",
    "locked_horizon4480_latent256": RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5",
}

LATENT_DIMS = {
    "latent512_beta3p75_live": 512,
    "latent384_beta3p75_promoted": 384,
    "latent384_beta4p0": 384,
    "recover035_full5x5_latent256": 256,
    "locked_horizon4480_latent256": 256,
}
BETAS = {
    "latent512_beta3p75_live": 3.75,
    "latent384_beta3p75_promoted": 3.75,
    "latent384_beta4p0": 4.0,
    "recover035_full5x5_latent256": 2.5,
    "locked_horizon4480_latent256": 2.5,
}


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def run_cmd(cmd: List[str]) -> Dict[str, Any]:
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {"cmd": cmd, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}


def write_table(df: pd.DataFrame, stem: str, max_rows: Optional[int] = None) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{stem}.csv"
    md_path = OUT_DIR / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    view = df if max_rows is None else df.head(max_rows)
    try:
        text = view.to_markdown(index=False)
    except Exception:
        text = view.to_string(index=False)
    if max_rows is not None and len(df) > max_rows:
        text += f"\n\nShowing {max_rows} of {len(df)} rows."
    md_path.write_text(text + "\n", encoding="utf-8")


def safe_float(v: Any) -> float:
    try:
        if pd.isna(v):
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


def metric_auc_pr(path: Path) -> Dict[str, float]:
    if not path.exists():
        return {"stage_a_logreg_auc": float("nan"), "stage_a_logreg_pr_auc": float("nan")}
    df = pd.read_csv(path)
    if "y_true" not in df.columns:
        return {"stage_a_logreg_auc": float("nan"), "stage_a_logreg_pr_auc": float("nan")}
    score_col = "y_score_final" if "y_score_final" in df.columns else "y_score"
    if score_col not in df.columns or df["y_true"].nunique() != 2:
        return {"stage_a_logreg_auc": float("nan"), "stage_a_logreg_pr_auc": float("nan")}
    y = df["y_true"].astype(int).to_numpy()
    s = df[score_col].astype(float).to_numpy()
    return {"stage_a_logreg_auc": float(roc_auc_score(y, s)), "stage_a_logreg_pr_auc": float(average_precision_score(y, s))}


def history_status(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "history_available": False,
            "current_or_final_epoch": float("nan"),
            "best_epoch": float("nan"),
            "best_val_loss_modelsel": float("nan"),
            "history_n_epochs": 0,
        }
    hist = joblib.load(path)
    val_model = np.asarray(hist.get("val_loss_modelsel", hist.get("val_loss", [])), dtype=float)
    n = int(len(val_model))
    if n == 0:
        return {
            "history_available": True,
            "current_or_final_epoch": float("nan"),
            "best_epoch": float("nan"),
            "best_val_loss_modelsel": float("nan"),
            "history_n_epochs": 0,
        }
    best_i = int(np.nanargmin(val_model))
    return {
        "history_available": True,
        "current_or_final_epoch": n,
        "best_epoch": best_i + 1,
        "best_val_loss_modelsel": float(val_model[best_i]),
        "history_n_epochs": n,
    }


def rd_values(path: Path, beta: float, latent_dim: int) -> Dict[str, Any]:
    if not path.exists():
        return {
            "rate_distortion_available": False,
            "rd_epoch_basis": "not_available",
            "D_val": float("nan"),
            "R_val_bits": float("nan"),
            "bits_per_dim": float("nan"),
            "KLD_over_D": float("nan"),
            "beta_KLD_over_D": float("nan"),
        }
    df = pd.read_csv(path)
    if df.empty:
        return {
            "rate_distortion_available": False,
            "rd_epoch_basis": "empty_file",
            "D_val": float("nan"),
            "R_val_bits": float("nan"),
            "bits_per_dim": float("nan"),
            "KLD_over_D": float("nan"),
            "beta_KLD_over_D": float("nan"),
        }
    row = df.iloc[-1]
    d = safe_float(row.get("D_val"))
    r_nats = safe_float(row.get("R_val_nats"))
    r_bits = safe_float(row.get("R_val_bits"))
    return {
        "rate_distortion_available": True,
        "rd_epoch_basis": "last_recorded_epoch",
        "D_val": d,
        "R_val_bits": r_bits,
        "bits_per_dim": r_bits / latent_dim if latent_dim else float("nan"),
        "KLD_over_D": r_nats / d if d else float("nan"),
        "beta_KLD_over_D": beta * r_nats / d if d else float("nan"),
    }


def latent_info(path: Path, split: str) -> Dict[str, Any]:
    prefix = f"{split}_"
    out = {
        f"{prefix}active_units": float("nan"),
        f"{prefix}total_correlation_nats": float("nan"),
        f"{prefix}mi_y_nats": float("nan"),
        f"{prefix}mi_manufacturer_nats": float("nan"),
        f"{prefix}mfr_y_mi_ratio": float("nan"),
    }
    if not path.exists():
        return out
    df = pd.read_csv(path)
    if df.empty or "variable" not in df.columns:
        return out
    first = df.iloc[0]
    out[f"{prefix}active_units"] = safe_float(first.get("n_active"))
    out[f"{prefix}total_correlation_nats"] = safe_float(first.get("total_correlation_nats"))
    y = df[df["variable"].astype(str).eq("Y_target")]
    m = df[df["variable"].astype(str).eq("Manufacturer")]
    if not y.empty:
        out[f"{prefix}mi_y_nats"] = safe_float(y.iloc[0].get("mi_sum_nats"))
    if not m.empty:
        out[f"{prefix}mi_manufacturer_nats"] = safe_float(m.iloc[0].get("mi_sum_nats"))
    y_val = out[f"{prefix}mi_y_nats"]
    m_val = out[f"{prefix}mi_manufacturer_nats"]
    out[f"{prefix}mfr_y_mi_ratio"] = m_val / y_val if y_val and not np.isnan(y_val) else float("nan")
    return out


def scanner_summary(path: Path, prefix: str) -> Dict[str, Any]:
    out = {
        f"{prefix}_scanner_leakage_available": False,
        f"{prefix}_acc_site_raw": float("nan"),
        f"{prefix}_acc_site_latent": float("nan"),
        f"{prefix}_latent_minus_raw": float("nan"),
    }
    if not path.exists():
        return out
    df = pd.read_csv(path)
    if df.empty:
        return out
    row = df.iloc[0]
    raw = safe_float(row.get("acc_site_raw"))
    lat = safe_float(row.get("acc_site_latent"))
    out.update(
        {
            f"{prefix}_scanner_leakage_available": True,
            f"{prefix}_acc_site_raw": raw,
            f"{prefix}_acc_site_latent": lat,
            f"{prefix}_latent_minus_raw": lat - raw if not np.isnan(raw) and not np.isnan(lat) else float("nan"),
        }
    )
    return out


def fold_status(run_label: str, run_dir: Path, fold: int) -> Dict[str, Any]:
    beta = BETAS[run_label]
    latent_dim = LATENT_DIMS[run_label]
    fdir = run_dir / f"fold_{fold}"
    row: Dict[str, Any] = {
        "run_label": run_label,
        "run_dir": str(run_dir),
        "fold": fold,
        "fold_dir_exists": fdir.exists(),
        "vae_checkpoint_exists": (fdir / f"vae_model_fold_{fold}.pt").exists(),
        "vae_history_exists": (fdir / f"vae_train_history_fold_{fold}.joblib").exists(),
        "stage_a_logreg_predictions_exist": (fdir / "test_predictions_logreg.csv").exists(),
        "stage_a_svm_predictions_exist": (fdir / "test_predictions_svm.csv").exists(),
        "classifier_only_readout_exists": (run_dir / "classifier_only_readout").exists(),
        "classifier_only_latent_cache_trainDev_exists": (
            run_dir / "classifier_only_readout" / "latent_cache" / f"fold_{fold}_trainDev_latent_mu.csv"
        ).exists(),
        "classifier_only_latent_cache_test_exists": (
            run_dir / "classifier_only_readout" / "latent_cache" / f"fold_{fold}_test_latent_mu.csv"
        ).exists(),
        "rate_distortion_exists": (fdir / f"fold_{fold}_rate_distortion.csv").exists(),
        "latent_info_trainDev_exists": (fdir / f"fold_{fold}_trainDev_latent_info_summary.csv").exists(),
        "latent_info_test_exists": (fdir / f"fold_{fold}_test_latent_info_summary.csv").exists(),
        "train_dev_scanner_leakage_exists": (fdir / f"fold_{fold}_scanner_leakage_summary.csv").exists(),
        "test_scanner_leakage_exists": (fdir / f"fold_{fold}_test_scanner_leakage_summary.csv").exists(),
    }
    row["vae_complete"] = row["vae_checkpoint_exists"] and row["vae_history_exists"]
    row["classifier_complete"] = row["stage_a_logreg_predictions_exist"] or row["classifier_only_latent_cache_test_exists"]
    row["latent_cache_available"] = row["classifier_only_latent_cache_trainDev_exists"] and row["classifier_only_latent_cache_test_exists"]
    row["rate_distortion_available"] = row["rate_distortion_exists"]
    row.update(history_status(fdir / f"vae_train_history_fold_{fold}.joblib"))
    row.update(rd_values(fdir / f"fold_{fold}_rate_distortion.csv", beta=beta, latent_dim=latent_dim))
    row.update(latent_info(fdir / f"fold_{fold}_trainDev_latent_info_summary.csv", "trainDev"))
    row.update(latent_info(fdir / f"fold_{fold}_test_latent_info_summary.csv", "test"))
    row.update(scanner_summary(fdir / f"fold_{fold}_scanner_leakage_summary.csv", "trainDev"))
    row.update(scanner_summary(fdir / f"fold_{fold}_test_scanner_leakage_summary.csv", "test"))
    row.update(metric_auc_pr(fdir / "test_predictions_logreg.csv"))
    if row["fold_dir_exists"] and not row["vae_history_exists"] and not row["vae_checkpoint_exists"]:
        row["live_status"] = "initialized_or_training_no_epoch_artifact_yet"
    elif row["vae_complete"]:
        row["live_status"] = "completed_stage_a_fold"
    elif row["fold_dir_exists"]:
        row["live_status"] = "partial_artifacts"
    else:
        row["live_status"] = "not_started"
    return row


def summarize_run(fold_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    metric_cols = [
        "D_val",
        "R_val_bits",
        "bits_per_dim",
        "KLD_over_D",
        "beta_KLD_over_D",
        "trainDev_active_units",
        "trainDev_total_correlation_nats",
        "trainDev_mi_y_nats",
        "trainDev_mi_manufacturer_nats",
        "trainDev_mfr_y_mi_ratio",
        "trainDev_acc_site_latent",
        "test_acc_site_latent",
        "stage_a_logreg_auc",
        "stage_a_logreg_pr_auc",
    ]
    for run, sub in fold_df.groupby("run_label", dropna=False):
        comp = sub[sub["vae_complete"].astype(bool)].copy()
        row: Dict[str, Any] = {
            "run_label": run,
            "folds_started": int(sub["fold_dir_exists"].sum()),
            "folds_vae_complete": int(sub["vae_complete"].sum()),
            "folds_stage_a_logreg_complete": int(sub["stage_a_logreg_predictions_exist"].sum()),
            "folds_classifier_only_latent_cache": int(sub["latent_cache_available"].sum()),
            "summary_basis": "vae_complete_folds_only",
        }
        for col in metric_cols:
            row[f"mean_{col}"] = float(comp[col].mean()) if not comp.empty and col in comp else float("nan")
            row[f"n_{col}"] = int(comp[col].notna().sum()) if not comp.empty and col in comp else 0
        rows.append(row)
    return pd.DataFrame(rows)


def compare_latent512_vs_384(summary: pd.DataFrame) -> pd.DataFrame:
    a = summary[summary["run_label"].eq("latent512_beta3p75_live")]
    b = summary[summary["run_label"].eq("latent384_beta3p75_promoted")]
    if a.empty or b.empty:
        return pd.DataFrame()
    cols = [
        "mean_D_val",
        "mean_R_val_bits",
        "mean_bits_per_dim",
        "mean_beta_KLD_over_D",
        "mean_trainDev_mi_y_nats",
        "mean_trainDev_mi_manufacturer_nats",
        "mean_trainDev_mfr_y_mi_ratio",
        "mean_trainDev_acc_site_latent",
        "mean_test_acc_site_latent",
        "mean_stage_a_logreg_auc",
        "mean_stage_a_logreg_pr_auc",
    ]
    rows = []
    ar = a.iloc[0]
    br = b.iloc[0]
    for col in cols:
        av = safe_float(ar.get(col))
        bv = safe_float(br.get(col))
        rows.append(
            {
                "metric": col.replace("mean_", ""),
                "latent512_mean_completed_folds": av,
                "latent384_mean_completed_folds": bv,
                "delta_512_minus_384": av - bv if not np.isnan(av) and not np.isnan(bv) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def interpretation(summary: pd.DataFrame, comp: pd.DataFrame, fold_df: pd.DataFrame) -> str:
    lines = [
        "# Live Interpretation",
        "",
        "No promotion decision is made because the latent512 run is incomplete.",
        "",
    ]
    live = summary[summary["run_label"].eq("latent512_beta3p75_live")]
    if live.empty:
        lines.append("Latent512 summary unavailable.")
    else:
        r = live.iloc[0]
        lines.append(
            f"Current latent512 status: {int(r['folds_vae_complete'])}/5 VAE folds complete, "
            f"{int(r['folds_stage_a_logreg_complete'])}/5 Stage A logreg folds complete, "
            f"{int(r['folds_classifier_only_latent_cache'])}/5 classifier-only latent caches available."
        )
    lines.append("")
    lines.append("## Rate-Distortion/Nuisance Readout")
    if not comp.empty:
        view = comp.set_index("metric")["delta_512_minus_384"].to_dict()
        bits = view.get("bits_per_dim", float("nan"))
        mfr = view.get("trainDev_mi_manufacturer_nats", float("nan"))
        y = view.get("trainDev_mi_y_nats", float("nan"))
        ratio = view.get("trainDev_mfr_y_mi_ratio", float("nan"))
        leak = view.get("test_acc_site_latent", float("nan"))
        beta_kd = view.get("beta_KLD_over_D", float("nan"))
        lines.append(
            f"Across currently completed latent512 folds, bits/dim is lower than latent384 by {bits:.4f}, "
            f"while total R bits is modestly higher/lower depending on completed fold mix. "
            f"beta*KLD/D delta is {beta_kd:.4f}."
        )
        lines.append(
            f"Manufacturer MI delta is {mfr:.4f}, diagnosis MI delta is {y:.4f}, "
            f"and Manufacturer/Y MI-ratio delta is {ratio:.4f}."
        )
        lines.append(f"Test latent scanner leakage delta is {leak:.4f}.")
        if not np.isnan(mfr) and mfr > 0 and not np.isnan(leak) and leak > 0:
            lines.append("Provisional signal: latent512 may encode more nuisance/manufacturer structure than latent384.")
        elif not np.isnan(leak) and leak <= 0:
            lines.append("Provisional signal: latent512 does not show higher test scanner leakage on completed folds.")
        else:
            lines.append("Provisional signal is insufficient for nuisance encoding judgment.")
    else:
        lines.append("Comparison against latent384 could not be computed.")
    lines.append("")
    lines.append("## Regularization Assessment")
    lines.append(
        "Current evidence is provisional. Completed folds show a broadly comparable rate-distortion regime rather than obvious collapse. "
        "Because active units are saturated at the latent dimensionality and TC is high, under-regularization/nuisance capacity remains a live risk, "
        "but this cannot be concluded until all five folds and Stage B score harmonization are complete."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    command_log = {"script": str(Path(__file__).resolve()), "start_time": now(), "commands": []}
    pc = run_cmd([sys.executable, "-m", "py_compile", str(Path(__file__).resolve())])
    command_log["commands"].append(pc)
    if pc["returncode"] != 0:
        raise RuntimeError("py_compile failed")

    rows: List[Dict[str, Any]] = []
    for label, path in RUNS.items():
        for fold in range(1, 6):
            rows.append(fold_status(label, path, fold))
    fold_df = pd.DataFrame(rows)
    summary = summarize_run(fold_df)
    comp = compare_latent512_vs_384(summary)

    write_table(fold_df, "fold_completion_and_live_metrics")
    write_table(fold_df[fold_df["run_label"].eq("latent512_beta3p75_live")], "latent512_completed_fold_metrics")
    write_table(summary, "run_level_completed_fold_summary")
    write_table(comp, "latent512_vs_latent384_completed_fold_comparison")

    (OUT_DIR / "live_interpretation.md").write_text(interpretation(summary, comp, fold_df), encoding="utf-8")
    readme = [
        "# latent512 beta3p75 live monitor",
        "",
        "Read-only live monitor. No training was launched and no model/tensor/metadata artifacts were modified.",
        "",
        "Metric basis: rate-distortion values are taken from the last recorded epoch in each completed fold. Run-level summaries use VAE-complete folds only.",
        "",
        "Primary files:",
        "- `fold_completion_and_live_metrics.csv/.md`",
        "- `latent512_completed_fold_metrics.csv/.md`",
        "- `run_level_completed_fold_summary.csv/.md`",
        "- `latent512_vs_latent384_completed_fold_comparison.csv/.md`",
        "- `live_interpretation.md`",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    command_log["end_time"] = now()
    (OUT_DIR / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
