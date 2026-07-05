#!/usr/bin/env python3
"""Read-only integrity audit for Manufacturer-conditioned FULL 5x5 branch."""

from __future__ import annotations

import argparse
import json
import py_compile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT_DEFAULT = PROJECT_ROOT / (
    "results/revision_bspc_2026/"
    "conditional_beta_vae_manufacturer_full5x5_mfrrecovered035_clfpoollocked"
)
EXPECTED_CANDIDATES = [
    "ch1_0_2_baseline_unconditioned",
    "ch1_0_2_decoder_only_manufacturer",
]
EXPECTED_CLASSIFIER_N = 396
EXPECTED_CN = 300
EXPECTED_AD = 96
N_FOLDS = 5
PATCHED_SUBJECT = "035_S_6927"
UNRESOLVED_SUBJECT = "128_S_2002"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._\n"
    lines = [
        "| " + " | ".join(df.columns) + " |",
        "| " + " | ".join(["---"] * len(df.columns)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for col in df.columns:
            value = row[col]
            if isinstance(value, float):
                vals.append(f"{value:.6f}")
            elif pd.isna(value):
                vals.append("")
            else:
                vals.append(str(value).replace("|", "\\|").replace("\n", " "))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def write_pair(root: Path, stem: str, df: pd.DataFrame) -> None:
    df.to_csv(root / f"{stem}.csv", index=False)
    (root / f"{stem}.md").write_text(md_table(df), encoding="utf-8")


def read_manifest(root: Path) -> pd.DataFrame:
    path = root / "run_manifest.csv"
    if path.exists():
        return pd.read_csv(path)
    rows = []
    for candidate in EXPECTED_CANDIDATES:
        run_dir = root / "runs" / candidate
        rows.append({
            "candidate_id": candidate,
            "run_dir": str(run_dir),
            "readout_dir": str(run_dir / "classifier_only_readout_z_plus_age_sex"),
            "vae_conditioning_mode": "decoder_only" if "decoder_only" in candidate else "none",
            "vae_conditioning_vars": "manufacturer" if "decoder_only" in candidate else "none",
            "channels": "[1, 0, 2]",
        })
    return pd.DataFrame(rows)


def audit_run(row: pd.Series) -> Dict[str, Any]:
    candidate = str(row["candidate_id"])
    run_dir = Path(str(row["run_dir"]))
    readout_dir = Path(str(row["readout_dir"]))
    fold_ckpts = []
    fold_histories = []
    supervised_035 = False
    supervised_128 = False
    supervised_rows = 0
    missing_supervised = 0
    for fold in range(1, N_FOLDS + 1):
        fold_dir = run_dir / f"fold_{fold}"
        fold_ckpts.append((fold_dir / f"vae_model_fold_{fold}.pt").exists())
        fold_histories.append((fold_dir / f"vae_train_history_fold_{fold}.joblib").exists())
        for filename in ["train_dev_subjects_fold.csv", "test_subjects_fold.csv"]:
            path = fold_dir / filename
            if not path.exists():
                missing_supervised += 1
                continue
            df = pd.read_csv(path)
            supervised_rows += len(df)
            subjects = set(df.get("SubjectID", pd.Series(dtype=str)).astype(str))
            supervised_035 = supervised_035 or PATCHED_SUBJECT in subjects
            supervised_128 = supervised_128 or UNRESOLVED_SUBJECT in subjects
    pooled_path = readout_dir / "classifier_sweep_pooled_metrics.csv"
    pool_ok = False
    n = n_cn = n_ad = None
    if pooled_path.exists():
        pooled = pd.read_csv(pooled_path)
        mask = (
            pooled.get("model_name", pd.Series(dtype=str)).astype(str).eq("logreg_l2")
            & pooled.get("threshold_strategy", pd.Series(dtype=str)).astype(str).eq("inner_oof_target_sens_ge_0p70_max_spec")
        )
        sub = pooled[mask].copy()
        if not sub.empty:
            r = sub.iloc[0]
            n = int(r.get("n", -1))
            n_cn = int(r.get("n_cn", -1))
            n_ad = int(r.get("n_ad", -1))
            pool_ok = (n == EXPECTED_CLASSIFIER_N and n_cn == EXPECTED_CN and n_ad == EXPECTED_AD)
    return {
        "candidate_id": candidate,
        "run_dir": str(run_dir),
        "run_dir_exists": run_dir.exists(),
        "run_dir_is_symlink": run_dir.is_symlink(),
        "run_config_exists": (run_dir / "run_config.json").exists(),
        "fold_checkpoints_present": int(sum(fold_ckpts)),
        "fold_histories_present": int(sum(fold_histories)),
        "all_fold_checkpoints_present": all(fold_ckpts),
        "all_fold_histories_present": all(fold_histories),
        "readout_dir_exists": readout_dir.exists(),
        "pooled_metrics_exists": pooled_path.exists(),
        "classifier_pool_n": n,
        "classifier_pool_cn": n_cn,
        "classifier_pool_ad": n_ad,
        "classifier_pool_locked_ok": pool_ok,
        "supervised_rows_checked": supervised_rows,
        "missing_supervised_split_files": missing_supervised,
        "supervised_contains_035_S_6927": supervised_035,
        "supervised_contains_128_S_2002": supervised_128,
        "integrity_pass": (
            all(fold_ckpts)
            and all(fold_histories)
            and pool_ok
            and not supervised_035
            and not supervised_128
            and missing_supervised == 0
        ),
    }


def main() -> int:
    args = parse_args()
    root = resolve(args.root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        py_compile.compile(str(Path(__file__).resolve()), doraise=True)
        py_compile_status = "ok"
    except Exception as exc:
        py_compile_status = f"failed: {exc}"

    manifest = read_manifest(root)
    rows = [audit_run(row) for _, row in manifest.iterrows()]
    audit = pd.DataFrame(rows)
    decision = "PASS" if not audit.empty and audit["integrity_pass"].all() else "PENDING_OR_FAIL"

    command_log = {
        "created_utc": now_utc(),
        "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "root": str(root),
        "dry_run": bool(args.dry_run),
        "py_compile": py_compile_status,
        "training_launched": False,
        "tensor_modified": False,
        "metadata_original_modified": False,
        "ledger_modified": False,
        "model_outputs_modified": False,
        "decision": decision,
    }
    if args.dry_run:
        print(json.dumps(command_log, indent=2))
        print(audit.to_string(index=False))
        (root / "command_log_integrity_dry_run.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
        return 0

    write_pair(root, "integrity_audit", audit)
    text = [
        "# FULL 5x5 Manufacturer-conditioned integrity audit",
        "",
        f"Generated UTC: {now_utc()}",
        "",
        f"Decision: **{decision}**",
        "",
        "PASS requires all five fold checkpoints and histories, Stage B pooled classifier pool n=396/CN=300/AD=96, and no 035_S_6927 or 128_S_2002 in supervised train/test splits.",
        "",
        md_table(audit),
    ]
    (root / "integrity_decision.md").write_text("\n".join(text), encoding="utf-8")
    (root / "command_log_integrity.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
    print(json.dumps(command_log, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
