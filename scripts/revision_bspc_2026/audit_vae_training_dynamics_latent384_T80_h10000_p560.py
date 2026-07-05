#!/usr/bin/env python3
"""Read-only VAE training dynamics posthoc audit hook for recover035_latent384_T80_h10000_p560.

Compares:
  1. locked_horizon4480_cycles56 (latent_dim=256):
       adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5
  2. recover035_longpatience_T80_h10000_p560_full5x5 (latent_dim=256):
       the direct base of latent384 (same schedule, different latent_dim)
  3. recover035_latent384_T80_h10000_p560_full5x5 (latent_dim=384):
       the candidate — single diff vs longpatience256 is latent_dim only

Primary questions:
  - Does the larger latent space (384 vs 256) affect training convergence speed or stability?
  - Does latent384 reach best_epoch later (more epochs to converge) or earlier?
  - Does latent384 use its patience window differently vs longpatience256?

Key metrics per fold:
  - best_epoch, pct_horizon_used
  - epochs_after_best (how much patience was consumed)
  - at_max_horizon (was convergence horizon-constrained?)
  - classifier AUC/PR-AUC pooled and per fold

Primary classifier readout: logreg_l2 / inner_oof_target_sens_ge_0p70_max_spec

Does NOT train, fit thresholds, do model selection, modify tensors,
metadata, ledgers, or existing model outputs.
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

DEFAULT_OUTPUT = RESULTS / "vae_training_dynamics_latent384_T80_h10000_p560"

MODEL_REGISTRY: Dict[str, Dict[str, Any]] = {
    "locked": {
        "label": "locked_horizon4480_cycles56_latent256",
        "run_dir_name": "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5",
        "latent_dim": 256,
        "epochs_vae": 4480,
        "n_cycles": 56,
        "patience": 320,
        "has_readout_feature_set": False,
    },
    "longpatience256": {
        "label": "recover035_longpatience256_T80_h10000_p560",
        "run_dir_name": "recover035_longpatience_T80_h10000_p560_full5x5",
        "latent_dim": 256,
        "epochs_vae": 10000,
        "n_cycles": 125,
        "patience": 560,
        "has_readout_feature_set": False,
    },
    "latent384": {
        "label": "recover035_latent384_T80_h10000_p560",
        "run_dir_name": "recover035_latent384_T80_h10000_p560_full5x5",
        "latent_dim": 384,
        "epochs_vae": 10000,
        "n_cycles": 125,
        "patience": 560,
        "has_readout_feature_set": False,
    },
}

MODEL_KEYS = ["locked", "longpatience256", "latent384"]
N_FOLDS = 5

PRIMARY_MODEL_NAME = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"

LOCKED_AUC = 0.782951
LOCKED_PR_AUC = 0.559873


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--latent384-run-dir", type=Path, default=None,
        help="Override run dir for the latent384 candidate model.",
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


def load_fold_history(run_dir: Path, fold: int) -> Optional[pd.DataFrame]:
    fold_dir = run_dir / f"fold_{fold}"
    candidates = sorted(fold_dir.glob(f"vae_training_history_fold_{fold}*.csv"))
    if not candidates:
        return None
    return pd.read_csv(candidates[0])


def per_fold_maturity(run_dir: Path, model_key: str) -> pd.DataFrame:
    reg = MODEL_REGISTRY[model_key]
    epochs_vae = reg["epochs_vae"]
    rows: List[Dict[str, Any]] = []
    for fold in range(1, N_FOLDS + 1):
        hist = load_fold_history(run_dir, fold)
        row: Dict[str, Any] = {
            "model_key": model_key,
            "label": reg["label"],
            "latent_dim": reg["latent_dim"],
            "fold": fold,
            "best_epoch": None,
            "pct_horizon": None,
            "epochs_after_best": None,
            "patience": reg["patience"],
            "at_max_horizon": None,
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
        rows.append(row)
    return pd.DataFrame(rows)


def load_classifier_foldwise(run_dir: Path, model_key: str) -> pd.DataFrame:
    reg = MODEL_REGISTRY[model_key]
    readout = run_dir / "classifier_only_readout"
    path = readout / "classifier_sweep_foldwise_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    mask = df["model_name"].astype(str).eq(PRIMARY_MODEL_NAME)
    mask &= df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    if "readout_feature_set" in df.columns and reg.get("has_readout_feature_set"):
        mask &= df["readout_feature_set"].astype(str).eq("z_plus_age_sex")
    df = df[mask].copy()
    df["model_key"] = model_key
    df["label"] = reg["label"]
    df["latent_dim"] = reg["latent_dim"]
    return df


def load_classifier_pooled(run_dir: Path, model_key: str) -> Dict[str, Any]:
    reg = MODEL_REGISTRY[model_key]
    readout = run_dir / "classifier_only_readout"
    path = readout / "classifier_sweep_pooled_metrics.csv"
    base = {"model_key": model_key, "label": reg["label"], "latent_dim": reg["latent_dim"]}
    if not path.exists():
        return {**base, "auc": float("nan"), "pr_auc": float("nan"), "note": "pooled_metrics not found"}
    df = pd.read_csv(path)
    mask = df["model_name"].astype(str).eq(PRIMARY_MODEL_NAME)
    mask &= df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    if "readout_feature_set" in df.columns and reg.get("has_readout_feature_set"):
        mask &= df["readout_feature_set"].astype(str).eq("z_plus_age_sex")
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


def maturity_summary(maturity_all: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for key in MODEL_KEYS:
        sub = maturity_all[maturity_all["model_key"] == key]
        if sub.empty:
            continue
        reg = MODEL_REGISTRY[key]
        valid = sub.dropna(subset=["best_epoch"])
        folds_at_max = int(sub["at_max_horizon"].eq(True).sum()) if "at_max_horizon" in sub.columns else 0
        rows.append({
            "model_key": key,
            "label": reg["label"],
            "latent_dim": reg["latent_dim"],
            "epochs_vae": reg["epochs_vae"],
            "patience": reg["patience"],
            "n_cycles": reg["n_cycles"],
            "mean_best_epoch": float(valid["best_epoch"].mean()) if not valid.empty else float("nan"),
            "mean_pct_horizon": float(valid["pct_horizon"].mean()) if not valid.empty else float("nan"),
            "mean_epochs_after_best": float(valid["epochs_after_best"].mean()) if "epochs_after_best" in valid.columns and not valid.empty else float("nan"),
            "folds_at_max_horizon": folds_at_max,
            "folds_available": int(len(valid)),
        })
    return pd.DataFrame(rows)


def write_recommendation(
    outdir: Path,
    maturity_summary_df: pd.DataFrame,
    pooled_metrics: pd.DataFrame,
    dry_run: bool,
) -> None:
    lines = [
        "# VAE Training Dynamics — recover035_latent384_T80_h10000_p560",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Mode: {'DRY-RUN' if dry_run else 'FULL ARTIFACT AUDIT'}",
        "",
        "Read-only. No VAE training, threshold fitting, model selection, or model-output modification.",
        "",
        "## Key questions",
        "",
        "1. Does latent_dim=384 require more epochs to converge than latent_dim=256?",
        "   (A larger latent space has more parameters; optimization may be slower.)",
        "2. Does latent384 use its patience window differently from longpatience256?",
        "3. Is the classifier improvement (if any) associated with richer training dynamics?",
        "",
        "## Schedule comparison",
        "",
        "| run | latent_dim | epochs_vae | n_cycles | cycle_len | patience | patience_cycles |",
        "|---|---|---|---|---|---|---|",
        "| locked_horizon4480_cycles56 | 256 | 4480 | 56 | 80 | 320 | 4 |",
        "| recover035_longpatience256 | 256 | 10000 | 125 | 80 | 560 | 7 |",
        "| recover035_latent384 | 384 | 10000 | 125 | 80 | 560 | 7 |",
        "",
        "longpatience256 and latent384 have identical schedule. Only latent_dim differs.",
        "",
    ]
    if not dry_run and not maturity_summary_df.empty:
        lines += [
            "## VAE training maturity summary",
            "",
            maturity_summary_df.to_markdown(index=False),
            "",
        ]
    if not dry_run and not pooled_metrics.empty:
        lines += [
            "## Pooled classifier metrics (logreg_l2 / inner_oof_target_sens_ge_0p70_max_spec)",
            "",
            pooled_metrics[["model_key", "label", "latent_dim", "n", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]].to_markdown(index=False),
            "",
            f"Promotion thresholds (locked reference): AUC > {LOCKED_AUC}, PR-AUC >= {LOCKED_PR_AUC}",
            "",
        ]
        lat384 = pooled_metrics[pooled_metrics["model_key"] == "latent384"]
        lp256 = pooled_metrics[pooled_metrics["model_key"] == "longpatience256"]
        if not lat384.empty:
            auc = float(lat384.iloc[0]["auc"])
            pr_auc = float(lat384.iloc[0]["pr_auc"])
            beats_auc = auc > LOCKED_AUC
            beats_pr = pr_auc >= LOCKED_PR_AUC
            lines += [
                f"**latent384 AUC > locked AUC: {'YES' if beats_auc else 'NO'} ({auc:.6f} vs {LOCKED_AUC})**",
                f"**latent384 PR-AUC >= locked PR-AUC: {'YES' if beats_pr else 'NO'} ({pr_auc:.6f} vs {LOCKED_PR_AUC})**",
                f"**Promotion: {'YES — meets both thresholds' if (beats_auc and beats_pr) else 'NO — does not meet both thresholds simultaneously'}**",
                "",
            ]
        if not lat384.empty and not lp256.empty:
            d_auc = float(lat384.iloc[0]["auc"]) - float(lp256.iloc[0]["auc"])
            d_pr = float(lat384.iloc[0]["pr_auc"]) - float(lp256.iloc[0]["pr_auc"])
            lines += [
                f"**Latent capacity delta (latent384 - latent256): AUC {d_auc:+.6f}, PR-AUC {d_pr:+.6f}**",
                "",
            ]
    lines += [
        "## Interpretation notes",
        "",
        "- If latent384 mean_best_epoch > longpatience256 mean_best_epoch, the larger model",
        "  needs more time to converge — consistent with harder optimization.",
        "- If mean_pct_horizon is similar between latent384 and longpatience256, convergence",
        "  speed (relative to horizon) is unchanged despite larger capacity.",
        "- folds_at_max_horizon > 0 indicates the model was horizon-constrained.",
        "- mean_epochs_after_best ≈ patience means the model stopped at the patience limit.",
        "- If latent384 and longpatience256 have similar dynamics but different AUC,",
        "  the classifier improvement is from representation quality, not training behavior.",
        "",
        "## Integrity",
        "",
        "This audit does NOT train, fit thresholds, modify tensors, metadata, ledger, configs,",
        "or existing model outputs.",
    ]
    (outdir / "final_recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    outdir = resolve(args.output_dir)

    overrides = {"latent384": args.latent384_run_dir, "longpatience256": args.longpatience256_run_dir}
    run_dirs: Dict[str, Path] = {key: resolve_run_dir(key, overrides) for key in MODEL_KEYS}

    print("Run directories:")
    for key, rd in run_dirs.items():
        reg = MODEL_REGISTRY[key]
        exists = rd.exists()
        print(f"  {key:20s} (latent_dim={reg['latent_dim']}, epochs={reg['epochs_vae']}): {rd}  ({'EXISTS' if exists else 'MISSING'})")

    if args.dry_run:
        missing = [key for key, rd in run_dirs.items() if not rd.exists()]
        if missing:
            print(f"\nDry-run: {len(missing)} run dir(s) not yet available (expected pre-training): {missing}")
        else:
            print("\nDry-run: all run directories exist.")
        print("Dry-run complete. No artifact files read, no output files written.")
        return 0

    if outdir.exists() and not args.overwrite:
        raise FileExistsError(f"{outdir} exists; pass --overwrite")
    outdir.mkdir(parents=True, exist_ok=True)

    maturity_parts: List[pd.DataFrame] = []
    for key in MODEL_KEYS:
        rd = run_dirs[key]
        if not rd.exists():
            print(f"  [{key}] run_dir not found, skipping maturity: {rd}")
            continue
        mat = per_fold_maturity(rd, key)
        maturity_parts.append(mat)

    maturity_all = pd.concat(maturity_parts, ignore_index=True) if maturity_parts else pd.DataFrame()
    if not maturity_all.empty:
        write_table(outdir, "maturity_by_fold", maturity_all, max_rows=30)

    summary = maturity_summary(maturity_all) if not maturity_all.empty else pd.DataFrame()
    if not summary.empty:
        write_table(outdir, "maturity_summary", summary)

    pooled_rows: List[Dict[str, Any]] = []
    foldwise_parts: List[pd.DataFrame] = []
    for key in MODEL_KEYS:
        rd = run_dirs[key]
        if not rd.exists():
            continue
        pooled_rows.append(load_classifier_pooled(rd, key))
        fw = load_classifier_foldwise(rd, key)
        if not fw.empty:
            foldwise_parts.append(fw)

    pooled_df = pd.DataFrame(pooled_rows)
    if not pooled_df.empty:
        write_table(outdir, "pooled_metrics_by_model", pooled_df)

    if foldwise_parts:
        foldwise_df = pd.concat(foldwise_parts, ignore_index=True)
        write_table(outdir, "foldwise_metrics_by_model", foldwise_df, max_rows=30)

    write_recommendation(outdir, summary, pooled_df, dry_run=False)

    cl = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "output_dir": str(outdir),
        "models_audited": MODEL_KEYS,
        "primary_comparison": "latent384 vs longpatience256 (latent_dim 256->384, all else equal)",
        "training_launched": False,
        "tensor_modified": False,
        "original_metadata_modified": False,
        "ledger_modified": False,
        "locked_model_outputs_modified": False,
        "longpatience256_outputs_modified": False,
    }
    (outdir / "command_log.json").write_text(json.dumps(cl, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nTraining dynamics audit complete. Output: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
