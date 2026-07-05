#!/usr/bin/env python3
"""Read-only VAE training dynamics audit: latent384_beta3p75 vs latent384_beta2p5.

Compares training convergence for three models with identical schedules:
  1. recover035_longpatience256 (latent_dim=256, beta=2.5, T80/h10000/p560)
  2. recover035_latent384_beta2p5 (latent_dim=384, beta=2.5, T80/h10000/p560) — base
  3. recover035_latent384_beta3p75 (latent_dim=384, beta=3.75, T80/h10000/p560) — candidate

Primary questions:
  - Does the stronger beta bottleneck (3.75 vs 2.5) change convergence speed or stability?
  - Does beta3.75 reach best_epoch later (slower convergence) or earlier (faster)?
  - At best epoch: compare val_recon, val_KLD, beta*KLD/recon across folds
  - Is the Fold 1 score-range anomaly reduced? (check via Stage B pooled vs foldwise AUC)

Key metrics per fold:
  - best_epoch, pct_horizon_used, epochs_after_best, at_max_horizon
  - val_recon_at_best, val_kld_at_best, beta_kld_recon_ratio_at_best
  - classifier AUC/PR-AUC pooled and per fold

Primary classifier readout: logreg_l2 / inner_oof_target_sens_ge_0p70_max_spec

Does NOT train, fit thresholds, modify tensors, metadata, ledgers, or model outputs.
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"
BIG_DISK = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026")

DEFAULT_OUTPUT = RESULTS / "vae_training_dynamics_beta3p75_vs_beta2p5_latent384"

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "longpatience256": {
        "label": "recover035_longpatience256_T80_h10000_p560",
        "run_dir_name": "recover035_longpatience_T80_h10000_p560_full5x5",
        "latent_dim": 256,
        "beta_vae": 2.5,
        "epochs_vae": 10000,
        "n_cycles": 125,
        "patience": 560,
        "has_readout_feature_set": False,
    },
    "latent384_beta2p5": {
        "label": "recover035_latent384_beta2.5_T80_h10000_p560",
        "run_dir_name": "recover035_latent384_T80_h10000_p560_full5x5",
        "latent_dim": 384,
        "beta_vae": 2.5,
        "epochs_vae": 10000,
        "n_cycles": 125,
        "patience": 560,
        "has_readout_feature_set": False,
    },
    "latent384_beta3p75": {
        "label": "recover035_latent384_beta3.75_T80_h10000_p560",
        "run_dir_name": "recover035_latent384_beta3p75_T80_h10000_p560_full5x5",
        "latent_dim": 384,
        "beta_vae": 3.75,
        "epochs_vae": 10000,
        "n_cycles": 125,
        "patience": 560,
        "has_readout_feature_set": False,
    },
}

MODEL_KEYS = ["longpatience256", "latent384_beta2p5", "latent384_beta3p75"]
N_FOLDS = 5

PRIMARY_MODEL_NAME = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"

LOCKED_AUC = 0.782951
LOCKED_PR_AUC = 0.559873


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--beta3p75-run-dir", type=Path, default=None,
        help="Override run dir for the latent384_beta3p75 candidate model.",
    )
    parser.add_argument(
        "--beta2p5-run-dir", type=Path, default=None,
        help="Override run dir for the latent384_beta2p5 base model.",
    )
    parser.add_argument(
        "--longpatience256-run-dir", type=Path, default=None,
        help="Override run dir for the longpatience256 reference model.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and structure without reading artifact files.",
    )
    return parser.parse_args()


def resolve_run_dir(key: str, overrides: Dict[str, Optional[Path]]) -> Path:
    override = overrides.get(key)
    if override is not None:
        return override if override.is_absolute() else PROJECT_ROOT / override
    name = MODEL_REGISTRY[key]["run_dir_name"]
    local = RESULTS / name
    big = BIG_DISK / name
    if local.exists():
        return local
    if big.exists():
        return big
    return local


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def md_table(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    text = view.to_markdown(index=False)
    if len(df) > max_rows:
        text += f"\n\n_Showing {max_rows} of {len(df)} rows._"
    return text + "\n"


def write_table(outdir: Path, stem: str, df: pd.DataFrame, max_rows: int = 40) -> None:
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def find_val_loss_col(hist: pd.DataFrame) -> Optional[str]:
    for col in ["val_loss_modelsel", "val_loss", "val_total_loss"]:
        if col in hist.columns:
            return col
    return None


def find_recon_col(hist: pd.DataFrame) -> Optional[str]:
    for col in ["val_recon_loss", "val_reconstruction_loss", "val_recon"]:
        if col in hist.columns:
            return col
    return None


def find_kld_col(hist: pd.DataFrame) -> Optional[str]:
    for col in ["val_kld_loss", "val_kl_loss", "val_kld", "val_kl"]:
        if col in hist.columns:
            return col
    return None


def load_fold_history(run_dir: Path, fold: int) -> Optional[pd.DataFrame]:
    fold_dir = run_dir / f"fold_{fold}"
    candidates = sorted(fold_dir.glob(f"vae_training_history_fold_{fold}*.csv"))
    if not candidates:
        return None
    return pd.read_csv(candidates[0])


def per_fold_dynamics(run_dir: Path, model_key: str) -> pd.DataFrame:
    reg = MODEL_REGISTRY[model_key]
    epochs_vae = reg["epochs_vae"]
    beta_vae = reg["beta_vae"]
    rows: List[Dict[str, Any]] = []
    for fold in range(1, N_FOLDS + 1):
        hist = load_fold_history(run_dir, fold)
        row: Dict[str, Any] = {
            "model_key": model_key,
            "label": reg["label"],
            "latent_dim": reg["latent_dim"],
            "beta_vae": beta_vae,
            "fold": fold,
            "best_epoch": None,
            "pct_horizon": None,
            "epochs_after_best": None,
            "patience": reg["patience"],
            "at_max_horizon": None,
            "val_recon_at_best": None,
            "val_kld_at_best": None,
            "beta_kld_recon_ratio": None,
            "note": "",
        }
        if hist is None:
            row["note"] = "training history not found"
            rows.append(row)
            continue
        val_col = find_val_loss_col(hist)
        if val_col is None:
            row["note"] = "no val_loss column"
            rows.append(row)
            continue
        valid = hist[val_col].dropna()
        if valid.empty:
            row["note"] = "val_loss all NaN"
            rows.append(row)
            continue
        best_ep = int(valid.idxmin())
        row["best_epoch"] = best_ep
        row["pct_horizon"] = float(best_ep / epochs_vae)
        row["epochs_after_best"] = int(epochs_vae - best_ep)
        row["at_max_horizon"] = bool(epochs_vae - best_ep <= reg["patience"])
        recon_col = find_recon_col(hist)
        kld_col = find_kld_col(hist)
        if recon_col and best_ep < len(hist):
            row["val_recon_at_best"] = float(hist.loc[best_ep, recon_col])
        if kld_col and best_ep < len(hist):
            row["val_kld_at_best"] = float(hist.loc[best_ep, kld_col])
        if row["val_recon_at_best"] and row["val_kld_at_best"] and row["val_recon_at_best"] > 0:
            row["beta_kld_recon_ratio"] = float(beta_vae * row["val_kld_at_best"] / row["val_recon_at_best"])
        rows.append(row)
    return pd.DataFrame(rows)


def load_classifier_pooled(run_dir: Path, model_key: str) -> Dict[str, Any]:
    reg = MODEL_REGISTRY[model_key]
    readout = run_dir / "classifier_only_readout"
    path = readout / "classifier_sweep_pooled_metrics.csv"
    base = {"model_key": model_key, "label": reg["label"], "latent_dim": reg["latent_dim"], "beta_vae": reg["beta_vae"]}
    if not path.exists():
        return {**base, "auc": float("nan"), "pr_auc": float("nan"), "note": "pooled_metrics not found"}
    df = pd.read_csv(path)
    mask = df["model_name"].astype(str).eq(PRIMARY_MODEL_NAME)
    mask &= df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    df = df[mask].copy()
    if df.empty:
        return {**base, "auc": float("nan"), "pr_auc": float("nan"), "note": "no matching row"}
    row = df.iloc[0]
    return {
        **base,
        "n": int(row.get("n", 0)),
        "auc": float(row.get("auc", float("nan"))),
        "pr_auc": float(row.get("pr_auc", float("nan"))),
        "balanced_accuracy": float(row.get("balanced_accuracy", float("nan"))),
        "sensitivity": float(row.get("sensitivity", float("nan"))),
        "specificity": float(row.get("specificity", float("nan"))),
        "f1": float(row.get("f1", float("nan"))),
        "note": "",
    }


def main() -> int:
    args = parse_args()
    outdir = resolve(args.output_dir)

    overrides: Dict[str, Optional[Path]] = {
        "latent384_beta3p75": args.beta3p75_run_dir,
        "latent384_beta2p5": args.beta2p5_run_dir,
        "longpatience256": args.longpatience256_run_dir,
    }

    run_dirs = {key: resolve_run_dir(key, overrides) for key in MODEL_KEYS}

    print("Training dynamics audit: beta3p75 vs beta2p5 latent384")
    for key, rd in run_dirs.items():
        label = MODEL_REGISTRY[key]["label"]
        beta = MODEL_REGISTRY[key]["beta_vae"]
        status = "exists" if rd.exists() else "NOT FOUND"
        print(f"  [{key}] beta={beta}  {rd}  ({status})")

    if args.dry_run:
        print("\nDry-run complete. No artifact files read.")
        print(f"Output would go to: {outdir}")
        return 0

    outdir.mkdir(parents=True, exist_ok=True)

    all_dynamics: List[pd.DataFrame] = []
    for key in MODEL_KEYS:
        rd = run_dirs[key]
        if not rd.exists():
            print(f"  [{key}] run_dir not found — skipping dynamics")
            continue
        df = per_fold_dynamics(rd, key)
        all_dynamics.append(df)

    if all_dynamics:
        dynamics_df = pd.concat(all_dynamics, ignore_index=True)
    else:
        dynamics_df = pd.DataFrame()

    write_table(outdir, "per_fold_dynamics", dynamics_df, max_rows=100)

    summary_rows: List[Dict[str, Any]] = []
    for key in MODEL_KEYS:
        reg = MODEL_REGISTRY[key]
        sub = dynamics_df[dynamics_df["model_key"] == key] if not dynamics_df.empty else pd.DataFrame()
        valid = sub.dropna(subset=["best_epoch"]) if not sub.empty else pd.DataFrame()
        folds_at_max = int(sub["at_max_horizon"].eq(True).sum()) if not sub.empty and "at_max_horizon" in sub.columns else 0
        summary_rows.append({
            "model_key": key,
            "label": reg["label"],
            "latent_dim": reg["latent_dim"],
            "beta_vae": reg["beta_vae"],
            "epochs_vae": reg["epochs_vae"],
            "patience": reg["patience"],
            "mean_best_epoch": float(valid["best_epoch"].mean()) if not valid.empty else float("nan"),
            "mean_pct_horizon": float(valid["pct_horizon"].mean()) if not valid.empty else float("nan"),
            "mean_epochs_after_best": float(valid["epochs_after_best"].mean()) if not valid.empty and "epochs_after_best" in valid.columns else float("nan"),
            "mean_val_recon_at_best": float(valid["val_recon_at_best"].mean()) if not valid.empty and "val_recon_at_best" in valid.columns else float("nan"),
            "mean_val_kld_at_best": float(valid["val_kld_at_best"].mean()) if not valid.empty and "val_kld_at_best" in valid.columns else float("nan"),
            "mean_beta_kld_recon_ratio": float(valid["beta_kld_recon_ratio"].mean()) if not valid.empty and "beta_kld_recon_ratio" in valid.columns else float("nan"),
            "folds_at_max_horizon": folds_at_max,
            "folds_available": int(len(valid)),
        })
    summary_df = pd.DataFrame(summary_rows)
    write_table(outdir, "dynamics_summary", summary_df)

    pooled_rows: List[Dict[str, Any]] = []
    for key in MODEL_KEYS:
        rd = run_dirs[key]
        if rd.exists():
            pooled_rows.append(load_classifier_pooled(rd, key))
        else:
            reg = MODEL_REGISTRY[key]
            pooled_rows.append({
                "model_key": key, "label": reg["label"],
                "latent_dim": reg["latent_dim"], "beta_vae": reg["beta_vae"],
                "auc": float("nan"), "pr_auc": float("nan"),
                "note": "run_dir not found",
            })
    pooled_df = pd.DataFrame(pooled_rows)
    write_table(outdir, "classifier_pooled_metrics", pooled_df)

    now = datetime.now(timezone.utc).isoformat()
    beta3_row = pooled_df[pooled_df["model_key"] == "latent384_beta3p75"]
    beta2_row = pooled_df[pooled_df["model_key"] == "latent384_beta2p5"]
    beta3_auc = float(beta3_row["auc"].iloc[0]) if not beta3_row.empty else float("nan")
    beta3_prauc = float(beta3_row["pr_auc"].iloc[0]) if not beta3_row.empty else float("nan")
    beta2_auc = float(beta2_row["auc"].iloc[0]) if not beta2_row.empty else float("nan")
    beta2_prauc = float(beta2_row["pr_auc"].iloc[0]) if not beta2_row.empty else float("nan")

    report_lines = [
        "# Training Dynamics Audit: beta3.75 vs beta2.5 latent384",
        "",
        f"Generated: {now}",
        "",
        "## Models",
        "| model_key | beta_vae | latent_dim | epochs | patience |",
        "|---|---|---|---|---|",
    ]
    for key in MODEL_KEYS:
        reg = MODEL_REGISTRY[key]
        report_lines.append(f"| {key} | {reg['beta_vae']} | {reg['latent_dim']} | {reg['epochs_vae']} | {reg['patience']} |")
    report_lines += [
        "",
        "## Classifier Performance",
        f"| model_key | beta_vae | AUC | PR-AUC | vs locked AUC | vs locked PR-AUC |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in pooled_df.iterrows():
        auc = r.get("auc", float("nan"))
        prauc = r.get("pr_auc", float("nan"))
        delta_auc = f"{auc - LOCKED_AUC:+.4f}" if not np.isnan(auc) else "—"
        delta_prauc = f"{prauc - LOCKED_PR_AUC:+.4f}" if not np.isnan(prauc) else "—"
        report_lines.append(f"| {r['model_key']} | {r.get('beta_vae', '?')} | {auc:.4f} | {prauc:.4f} | {delta_auc} | {delta_prauc} |")
    if not (np.isnan(beta3_auc) or np.isnan(beta2_auc)):
        report_lines += [
            "",
            "## beta3.75 vs beta2.5 delta",
            f"- AUC: {beta3_auc:.4f} - {beta2_auc:.4f} = {beta3_auc - beta2_auc:+.4f}",
            f"- PR-AUC: {beta3_prauc:.4f} - {beta2_prauc:.4f} = {beta3_prauc - beta2_prauc:+.4f}",
            f"- beta3.75 promotes AUC: {beta3_auc > LOCKED_AUC}",
            f"- beta3.75 promotes PR-AUC: {beta3_prauc >= LOCKED_PR_AUC}",
        ]
    report_lines += [
        "",
        "## Read-only guarantee",
        "Did not train, fit thresholds, modify tensors, metadata, ledger, configs, or model outputs.",
    ]
    (outdir / "dynamics_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\nOutput: {outdir}")
    print(f"beta3.75 AUC={beta3_auc:.4f}, PR-AUC={beta3_prauc:.4f}")
    print(f"beta2.5  AUC={beta2_auc:.4f}, PR-AUC={beta2_prauc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
