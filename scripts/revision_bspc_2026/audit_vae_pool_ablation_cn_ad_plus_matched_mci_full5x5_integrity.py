#!/usr/bin/env python3
"""Integrity and leakage audit for the exploratory VAE-pool FULL 5x5 run."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CANDIDATE_RUN = PROJECT_ROOT / "results/revision_bspc_2026/vae_pool_ablation_cn_ad_plus_matched_mci_full5x5"
REFERENCE_RUN = PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5"
DEFAULT_OUTPUT = CANDIDATE_RUN / "integrity_audit"
CONFIG_PATH = PROJECT_ROOT / "configs/runs/adni_v5_1b_ch1_0_2_cn_ad_plus_matched_mci_pool_horizon4480_cycles56_full_5x5.json"
PRIMARY_MODEL = "logreg_l2"
PRIMARY_READOUT = "z_plus_age_sex"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--candidate-run-dir", type=Path, default=CANDIDATE_RUN)
    parser.add_argument("--reference-run-dir", type=Path, default=REFERENCE_RUN)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def md_table(df: pd.DataFrame, max_rows: int = 160) -> str:
    if df.empty:
        return "_No rows._\n"
    sub = df.head(max_rows)
    lines = [
        "| " + " | ".join(sub.columns) + " |",
        "| " + " | ".join(["---"] * len(sub.columns)) + " |",
    ]
    for _, row in sub.iterrows():
        vals: List[str] = []
        for col in sub.columns:
            value = row[col]
            if isinstance(value, (float, np.floating)):
                vals.append(f"{value:.6f}" if np.isfinite(value) else "")
            elif pd.isna(value):
                vals.append("")
            else:
                vals.append(str(value).replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def write_df_pair(root: Path, stem: str, df: pd.DataFrame, max_rows: int = 160) -> None:
    root.mkdir(parents=True, exist_ok=True)
    df.to_csv(root / f"{stem}.csv", index=False)
    (root / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_primary(readout_dir: Path, filename: str) -> pd.DataFrame:
    path = readout_dir / filename
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "readout_feature_set" not in df.columns:
        df["readout_feature_set"] = PRIMARY_READOUT
    return df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
        & df["readout_feature_set"].astype(str).eq(PRIMARY_READOUT)
    ].copy()


def artifact_inventory(run_dir: Path) -> pd.DataFrame:
    rows = []
    for fold in range(1, 6):
        fold_dir = run_dir / f"fold_{fold}"
        for name in [
            f"vae_model_fold_{fold}.pt",
            f"vae_train_history_fold_{fold}.joblib",
            "vae_training_pool_tensor_idx.npy",
            "train_dev_tensor_idx.npy",
            "test_tensor_idx.npy",
            "train_dev_subjects_fold.csv",
            "test_subjects_fold.csv",
            f"vae_pool_composition_strategy_summary_fold_{fold}.csv",
            f"fold_{fold}_scanner_leakage_summary.csv",
            f"fold_{fold}_test_scanner_leakage_summary.csv",
            "latent_qc_metrics.csv",
        ]:
            p = fold_dir / name
            rows.append(
                {
                    "fold": fold,
                    "file": name,
                    "path": str(p),
                    "exists": p.exists(),
                    "size_bytes": p.stat().st_size if p.exists() else np.nan,
                    "mtime_utc": datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).isoformat(timespec="seconds") if p.exists() else "",
                }
            )
    return pd.DataFrame(rows)


def load_idx(path: Path) -> np.ndarray:
    if not path.exists():
        return np.array([], dtype=int)
    return np.asarray(np.load(path, allow_pickle=False), dtype=int)


def pool_leakage_audit(run_dir: Path) -> pd.DataFrame:
    rows = []
    for fold in range(1, 6):
        fold_dir = run_dir / f"fold_{fold}"
        vae_pool = set(load_idx(fold_dir / "vae_training_pool_tensor_idx.npy").tolist())
        test = set(load_idx(fold_dir / "test_tensor_idx.npy").tolist())
        train_dev = set(load_idx(fold_dir / "train_dev_tensor_idx.npy").tolist())
        row: Dict[str, Any] = {
            "fold": fold,
            "vae_pool_n": len(vae_pool),
            "classifier_train_dev_n": len(train_dev),
            "classifier_test_n": len(test),
            "outer_test_overlap_n": len(vae_pool.intersection(test)),
            "train_dev_test_overlap_n": len(train_dev.intersection(test)),
        }
        summary_path = fold_dir / f"vae_pool_composition_strategy_summary_fold_{fold}.csv"
        if summary_path.exists():
            summary = pd.read_csv(summary_path).iloc[0].to_dict()
            row.update({f"strategy_{k}": v for k, v in summary.items() if k not in row})
        rows.append(row)
    return pd.DataFrame(rows)


def classifier_pool_identity(run_dir: Path) -> pd.DataFrame:
    rows = []
    all_test_subjects: List[pd.DataFrame] = []
    for fold in range(1, 6):
        fold_dir = run_dir / f"fold_{fold}"
        train_path = fold_dir / "train_dev_subjects_fold.csv"
        test_path = fold_dir / "test_subjects_fold.csv"
        if not train_path.exists() or not test_path.exists():
            rows.append({"fold": fold, "status": "missing_subject_files"})
            continue
        train = pd.read_csv(train_path)
        test = pd.read_csv(test_path)
        all_test_subjects.append(test.assign(fold=fold))
        for label, df in [("train_dev", train), ("test", test)]:
            vc = df["ResearchGroup_Mapped"].astype(str).value_counts()
            rows.append(
                {
                    "fold": fold,
                    "split": label,
                    "n": int(len(df)),
                    "cn": int(vc.get("CN", 0)),
                    "ad": int(vc.get("AD", 0)),
                    "mci": int(vc.get("MCI", 0)),
                    "unique_subjects": int(df["SubjectID"].astype(str).nunique()) if "SubjectID" in df.columns else np.nan,
                }
            )
    if all_test_subjects:
        pooled = pd.concat(all_test_subjects, ignore_index=True)
        vc = pooled["ResearchGroup_Mapped"].astype(str).value_counts()
        rows.append(
            {
                "fold": "pooled_test_once",
                "split": "all_outer_tests",
                "n": int(len(pooled)),
                "cn": int(vc.get("CN", 0)),
                "ad": int(vc.get("AD", 0)),
                "mci": int(vc.get("MCI", 0)),
                "unique_subjects": int(pooled["SubjectID"].astype(str).nunique()) if "SubjectID" in pooled.columns else np.nan,
                "duplicate_subject_rows": int(len(pooled) - pooled["SubjectID"].astype(str).nunique()) if "SubjectID" in pooled.columns else np.nan,
            }
        )
    return pd.DataFrame(rows)


def collect_scanner_leakage(run_dir: Path, run_label: str) -> pd.DataFrame:
    rows = []
    for fold in range(1, 6):
        for scope, filename in [
            ("train_dev", f"fold_{fold}_scanner_leakage_summary.csv"),
            ("test", f"fold_{fold}_test_scanner_leakage_summary.csv"),
        ]:
            path = run_dir / f"fold_{fold}" / filename
            if path.exists():
                df = pd.read_csv(path)
                df.insert(0, "run_label", run_label)
                df.insert(1, "fold", fold)
                df.insert(2, "scope", scope)
                rows.extend(df.to_dict("records"))
    return pd.DataFrame(rows)


def collect_training_maturity(run_dir: Path) -> pd.DataFrame:
    rows = []
    for fold in range(1, 6):
        path = run_dir / f"fold_{fold}" / f"vae_train_history_fold_{fold}.joblib"
        row: Dict[str, Any] = {"fold": fold, "history_exists": path.exists()}
        if path.exists():
            try:
                hist = joblib.load(path)
                if isinstance(hist, dict):
                    for key in ["best_epoch", "early_stop_epoch", "final_epoch", "best_val_loss_beta_max", "best_val_loss"]:
                        if key in hist:
                            row[key] = hist[key]
                    val = hist.get("val_loss_beta_max") or hist.get("val_loss") or []
                    if len(val):
                        arr = np.asarray(val, dtype=float)
                        row["final_epoch_inferred"] = int(len(arr))
                        row["last_val_loss"] = float(arr[-1])
                        if "best_epoch" in row and pd.notna(row["best_epoch"]):
                            row["epochs_after_best"] = int(len(arr) - int(row["best_epoch"]))
                elif isinstance(hist, pd.DataFrame):
                    row["final_epoch_inferred"] = int(len(hist))
            except Exception as exc:
                row["history_error"] = str(exc)
        rows.append(row)
    return pd.DataFrame(rows)


def collect_foldwise(readout_dir: Path, run_label: str) -> pd.DataFrame:
    df = read_primary(readout_dir, "classifier_sweep_foldwise_metrics.csv")
    if not df.empty:
        df.insert(0, "run_label", run_label)
    return df


def collect_fold14(foldwise: pd.DataFrame) -> pd.DataFrame:
    if foldwise.empty or "fold" not in foldwise.columns:
        return pd.DataFrame()
    return foldwise[foldwise["fold"].astype(int).isin([1, 4])].copy()


def oasis_preflight(run_dir: Path) -> pd.DataFrame:
    rows = []
    for fold in range(1, 6):
        ckpt = run_dir / f"fold_{fold}" / f"vae_model_fold_{fold}.pt"
        rows.append({"fold": fold, "vae_checkpoint_exists": ckpt.exists(), "checkpoint_path": str(ckpt)})
    readout = run_dir / "classifier_only_readout"
    rows.append(
        {
            "fold": "stage_b",
            "vae_checkpoint_exists": (readout / "classifier_sweep_pooled_metrics.csv").exists(),
            "checkpoint_path": str(readout / "classifier_sweep_pooled_metrics.csv"),
        }
    )
    return pd.DataFrame(rows)


def final_decision(inventory: pd.DataFrame, pool: pd.DataFrame, identity: pd.DataFrame) -> str:
    missing = inventory[~inventory["exists"]]
    overlap_ok = (not pool.empty) and pool["outer_test_overlap_n"].fillna(1).eq(0).all()
    pooled = identity[identity["split"].eq("all_outer_tests")] if "split" in identity.columns else pd.DataFrame()
    pool_ok = False
    if not pooled.empty:
        r = pooled.iloc[0]
        pool_ok = int(r["n"]) == 396 and int(r["cn"]) == 300 and int(r["ad"]) == 96 and int(r.get("duplicate_subject_rows", 1)) == 0
    status = "PASS" if missing.empty and overlap_ok and pool_ok else "FAIL_OR_INCOMPLETE"
    lines = [
        "# Final Integrity Decision",
        "",
        f"Integrity status: `{status}`.",
        "",
        f"- Required artifact inventory complete: `{missing.empty}`.",
        f"- VAE pool has zero outer-test overlap in every fold: `{overlap_ok}`.",
        f"- Classifier pooled outer tests remain n=396, CN=300, AD=96 with no duplicate subject rows: `{pool_ok}`.",
        "",
        "This branch remains exploratory because diagnosis labels influence fold-local VAE pool composition.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    run_dir = resolve(args.candidate_run_dir)
    ref_dir = resolve(args.reference_run_dir)
    outdir = resolve(args.output_dir)
    print(f"Candidate run: {run_dir}")
    print(f"Reference run: {ref_dir}")
    if args.dry_run:
        if not resolve(args.config).exists():
            raise FileNotFoundError(resolve(args.config))
        print("Dry-run complete. Candidate artifacts are not required yet.")
        return 0
    cfg = load_config(resolve(args.config))
    if cfg["parameters"].get("vae_pool_composition_strategy") != "cn_ad_plus_matched_mci_pool":
        raise RuntimeError("Candidate config does not use cn_ad_plus_matched_mci_pool")
    inventory = artifact_inventory(run_dir)
    pool = pool_leakage_audit(run_dir)
    identity = classifier_pool_identity(run_dir)
    scanner = pd.concat(
        [
            collect_scanner_leakage(ref_dir, "locked_v5_1b_ch1_0_2"),
            collect_scanner_leakage(run_dir, "cn_ad_plus_matched_mci_pool"),
        ],
        ignore_index=True,
    )
    maturity = collect_training_maturity(run_dir)
    foldwise = pd.concat(
        [
            collect_foldwise(ref_dir / "classifier_only_readout", "locked_v5_1b_ch1_0_2"),
            collect_foldwise(run_dir / "classifier_only_readout", "cn_ad_plus_matched_mci_pool"),
        ],
        ignore_index=True,
    )
    fold14 = collect_fold14(foldwise)
    oasis = oasis_preflight(run_dir)
    write_df_pair(outdir, "artifact_inventory", inventory, max_rows=240)
    write_df_pair(outdir, "pool_composition_leakage_audit", pool)
    write_df_pair(outdir, "classifier_pool_identity", identity)
    write_df_pair(outdir, "scanner_leakage_by_fold", scanner, max_rows=240)
    write_df_pair(outdir, "foldwise_metrics", foldwise, max_rows=240)
    write_df_pair(outdir, "fold1_fold4_behavior", fold14)
    write_df_pair(outdir, "training_maturity", maturity)
    write_df_pair(outdir, "oasis_external_scoring_preflight", oasis)
    (outdir / "final_integrity_decision.md").write_text(final_decision(inventory, pool, identity), encoding="utf-8")
    write_json(
        outdir / "command_log.json",
        {
            "timestamp": now(),
            "candidate_run": str(run_dir),
            "reference_run": str(ref_dir),
            "dry_run": False,
            "training_launched": False,
            "tensor_modified": False,
            "metadata_modified": False,
            "ledger_modified": False,
            "model_outputs_modified": False,
        },
    )
    print(f"Wrote integrity audit: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
