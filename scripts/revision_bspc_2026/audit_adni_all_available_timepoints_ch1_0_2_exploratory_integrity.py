#!/usr/bin/env python3
"""Integrity and confounding audit for exploratory ADNI all-timepoints FULL run."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"
DEFAULT_RUN = RESULTS / "adni_all_available_timepoints_ch1_0_2_exploratory_full_5x5"
DEFAULT_READOUT = DEFAULT_RUN / "classifier_only_readout"
DEFAULT_OUTPUT = RESULTS / "adni_all_available_timepoints_ch1_0_2_exploratory_full_5x5_integrity_audit"
CONFIG = PROJECT_ROOT / "configs/runs/adni_all_available_timepoints_ch1_0_2_exploratory_full_5x5.json"
TENSOR_BUILD_QC = RESULTS / "adni_all_available_timepoints_ch1_0_2_exploratory_tensor_build"
OASIS_SCORING_DIR = RESULTS / "oasis_tanda_2026_05_25_external_scoring"
PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--candidate-run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--candidate-readout-dir", type=Path, default=DEFAULT_READOUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_table(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    (out_dir / f"{stem}.md").write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def inventory(args: argparse.Namespace) -> pd.DataFrame:
    cfg = load_json(args.config)
    paths = {
        "config": args.config,
        "candidate_run_dir": args.candidate_run_dir,
        "candidate_readout_dir": args.candidate_readout_dir,
        "tensor_build_qc": TENSOR_BUILD_QC,
        "branch_tensor": Path(cfg["paths"]["global_tensor_path"]),
        "branch_metadata": Path(cfg["paths"]["metadata_path"]),
        "oasis_external_scoring_reference": OASIS_SCORING_DIR,
    }
    return pd.DataFrame(
        [
            {
                "item": item,
                "path": str(path),
                "exists": bool(path.exists()),
                "is_symlink": bool(path.is_symlink()),
                "resolved_path": str(path.resolve()) if path.exists() else "",
            }
            for item, path in paths.items()
        ]
    )


def checkpoint_integrity(run_dir: Path) -> pd.DataFrame:
    rows = []
    for fold in range(1, 6):
        fold_dir = run_dir / f"fold_{fold}"
        ckpt = fold_dir / f"vae_model_fold_{fold}.pt"
        hist = fold_dir / f"vae_train_history_fold_{fold}.joblib"
        rows.append(
            {
                "fold": fold,
                "fold_dir": str(fold_dir),
                "fold_dir_exists": bool(fold_dir.exists()),
                "checkpoint_exists": bool(ckpt.exists()),
                "checkpoint_mtime": datetime.fromtimestamp(ckpt.stat().st_mtime).isoformat() if ckpt.exists() else "",
                "history_exists": bool(hist.exists()),
            }
        )
    return pd.DataFrame(rows)


def readout_integrity(readout_dir: Path) -> pd.DataFrame:
    files = [
        "classifier_sweep_pooled_metrics.csv",
        "classifier_sweep_foldwise_metrics.csv",
        "classifier_sweep_predictions.csv",
        "classifier_sweep_thresholds_by_fold.csv",
        "latent_cache",
    ]
    rows = []
    for name in files:
        path = readout_dir / name
        rows.append({"item": name, "path": str(path), "exists": bool(path.exists())})
    if (readout_dir / "classifier_sweep_pooled_metrics.csv").exists():
        pooled = pd.read_csv(readout_dir / "classifier_sweep_pooled_metrics.csv")
        mask = pooled["model_name"].astype(str).eq(PRIMARY_MODEL) & pooled["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
        rows.append({"item": "primary_operating_point", "path": PRIMARY_THRESHOLD, "exists": bool(mask.any())})
    return pd.DataFrame(rows)


def load_latents(readout_dir: Path) -> pd.DataFrame:
    cache = readout_dir / "latent_cache"
    parts = []
    for path in sorted(cache.glob("fold_*_test_latent_mu.csv")):
        parts.append(pd.read_csv(path))
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def add_metadata(latents: pd.DataFrame, config_path: Path) -> pd.DataFrame:
    if latents.empty:
        return latents
    cfg = load_json(config_path)
    meta = pd.read_csv(cfg["paths"]["metadata_path"])
    if "original_n_TR" not in meta.columns and "original_n_timepoints" in meta.columns:
        meta["original_n_TR"] = meta["original_n_timepoints"]
    keep = [c for c in ["SubjectID", "original_n_TR", "SiteCode", "Manufacturer", "ResearchGroup_Mapped"] if c in meta.columns]
    out = latents.merge(meta[keep].drop_duplicates("SubjectID"), on="SubjectID", how="left", suffixes=("", "_meta"))
    if "SiteCode" not in out.columns:
        out["SiteCode"] = out["SubjectID"].astype(str).str.slice(0, 3)
    return out


def latent_confound_predictability(latents: pd.DataFrame) -> pd.DataFrame:
    if latents.empty:
        return pd.DataFrame(
            [{"target": "latent_cache", "status": "missing", "metric": "", "value": np.nan, "details": "No latent cache found."}]
        )
    mu_cols = [c for c in latents.columns if c.startswith("mu_")]
    x = latents[mu_cols].to_numpy(dtype=float)
    rows = []
    if "original_n_TR" in latents.columns:
        y = pd.to_numeric(latents["original_n_TR"], errors="coerce")
        mask = y.notna()
        if mask.sum() >= 20:
            model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
            pred = cross_val_predict(model, x[mask], y[mask].to_numpy(dtype=float), cv=5)
            rows.append({"target": "original_n_TR", "status": "ok", "metric": "cv_r2", "value": float(r2_score(y[mask], pred)), "details": f"n={int(mask.sum())}"})
    for target in ["Manufacturer", "SiteCode"]:
        if target not in latents.columns:
            continue
        y = latents[target].astype(str)
        counts = y.value_counts()
        valid_labels = counts[counts >= 5].index
        mask = y.isin(valid_labels)
        if mask.sum() < 30 or y[mask].nunique() < 2:
            rows.append({"target": target, "status": "insufficient_counts", "metric": "balanced_accuracy", "value": np.nan, "details": counts.to_dict()})
            continue
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced", multi_class="auto"))
        pred = cross_val_predict(model, x[mask], y[mask], cv=cv)
        rows.append({"target": target, "status": "ok", "metric": "cv_balanced_accuracy", "value": float(balanced_accuracy_score(y[mask], pred)), "details": f"n={int(mask.sum())}; labels={','.join(sorted(y[mask].unique()))}"})
    return pd.DataFrame(rows)


def scanner_leakage_summary(run_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(run_dir.glob("fold_*/fold_*_scanner_leakage_summary.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        row["fold"] = int(path.parent.name.replace("fold_", ""))
        row["file"] = str(path)
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame([{"status": "missing", "details": "No scanner leakage summaries found."}])


def subgroup_summary(readout_dir: Path) -> pd.DataFrame:
    path = readout_dir / "classifier_sweep_subgroup_metrics_by_manufacturer.csv"
    if path.exists():
        df = pd.read_csv(path)
        return df[
            df["model_name"].astype(str).eq(PRIMARY_MODEL)
            & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
        ].copy()
    return pd.DataFrame([{"status": "missing", "details": str(path)}])


def oasis_preflight() -> str:
    if not OASIS_SCORING_DIR.exists():
        return "OASIS scoring directory is absent; external scoring preflight pending after candidate artifacts exist."
    return (
        "OASIS pilot scoring outputs exist. After this ADNI branch finishes, run a dedicated external scoring script "
        "against the all-timepoints fold artifacts without OASIS threshold fitting or model selection."
    )


def final_decision(checkpoints: pd.DataFrame, readout: pd.DataFrame, latent_pred: pd.DataFrame) -> str:
    ckpt_ok = bool(checkpoints["checkpoint_exists"].all()) if "checkpoint_exists" in checkpoints else False
    readout_ok = bool(readout.loc[readout["item"].eq("primary_operating_point"), "exists"].any()) if "item" in readout else False
    if not ckpt_ok or not readout_ok:
        return "PENDING: candidate training/readout artifacts are incomplete."
    leakage_warning = ""
    if not latent_pred.empty and "target" in latent_pred.columns:
        ntr = latent_pred[latent_pred["target"].eq("original_n_TR")]
        if not ntr.empty and pd.to_numeric(ntr["value"], errors="coerce").iloc[0] > 0.10:
            leakage_warning = " n_TR is meaningfully predictable from z; treat any AUC gain as confounded unless external validation improves."
    return "PASS_WITH_EXPLORATORY_GUARDRAIL: artifacts present, but promotion requires comparison plus confounding/OASIS review." + leakage_warning


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inv = inventory(args)
    ckpt = checkpoint_integrity(args.candidate_run_dir)
    readout = readout_integrity(args.candidate_readout_dir)
    if args.dry_run:
        latent_pred = pd.DataFrame([{"target": "dry_run", "status": "not_run", "metric": "", "value": np.nan, "details": "Dry-run only."}])
        leakage = pd.DataFrame([{"status": "dry_run", "details": "Dry-run only."}])
        subgroup = pd.DataFrame([{"status": "dry_run", "details": "Dry-run only."}])
    else:
        latents = add_metadata(load_latents(args.candidate_readout_dir), args.config)
        latent_pred = latent_confound_predictability(latents)
        leakage = scanner_leakage_summary(args.candidate_run_dir)
        subgroup = subgroup_summary(args.candidate_readout_dir)
    write_table(inv, args.output_dir, "file_inventory")
    write_table(ckpt, args.output_dir, "checkpoint_integrity")
    write_table(readout, args.output_dir, "stageb_readout_integrity")
    write_table(latent_pred, args.output_dir, "ntr_predictability_from_latent")
    write_table(leakage, args.output_dir, "scanner_manufacturer_leakage")
    write_table(subgroup, args.output_dir, "manufacturer_subgroup_metrics")
    (args.output_dir / "oasis_external_scoring_preflight.md").write_text("# OASIS External Scoring Preflight\n\n" + oasis_preflight() + "\n", encoding="utf-8")
    decision = final_decision(ckpt, readout, latent_pred)
    (args.output_dir / "final_integrity_decision.md").write_text("# Final Integrity Decision\n\n" + decision + "\n", encoding="utf-8")
    command_log = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "argv": __import__("sys").argv,
        "dry_run": bool(args.dry_run),
        "training_launched": False,
        "candidate_run_dir": str(args.candidate_run_dir),
        "candidate_readout_dir": str(args.candidate_readout_dir),
        "decision": decision,
    }
    (args.output_dir / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
