#!/usr/bin/env python3
"""Real launcher for the VAE pool-composition FAST 3x3 ablation.

The preflight package already contains validated Stage A and Stage B commands.
This launcher consumes that manifest, refuses real execution unless
``--confirm-training`` is passed, and runs Stage A followed by classifier-only
Stage B for each selected candidate.
"""

from __future__ import annotations

import argparse
import json
import py_compile
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = PROJECT_ROOT / "results/revision_bspc_2026/vae_pool_ablation_fast3x3"
AGGREGATOR = PROJECT_ROOT / "scripts/revision_bspc_2026/aggregate_vae_pool_ablation_fast3x3.py"
CANDIDATES = (
    "current_all_pool",
    "cn_ad_only_pool",
    "balanced_cn_ad_mci_pool",
    "cn_ad_plus_matched_mci_pool",
)
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
LOCKED_TENSOR_STEM = "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
STALE_MARKERS = (
    "fold_1",
    "fold_2",
    "fold_3",
    "classifier_only_readout",
    "latent_cache",
    "all_folds_metrics",
    "summary_metrics",
    "run_config.json",
    "command_log.json",
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--package-root", type=Path, default=PACKAGE_ROOT)
    parser.add_argument("--candidate", choices=[*CANDIDATES, "all"], default="all")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print commands only; no training.")
    parser.add_argument("--confirm-training", action="store_true", help="Required for real Stage A and Stage B execution.")
    parser.add_argument("--resume", action="store_true", help="Skip candidates with complete Stage B outputs.")
    parser.add_argument("--force-clean", action="store_true", help="Quarantine existing candidate outputs before confirmed training.")
    parser.add_argument("--skip-aggregation", action="store_true", help="Do not run the aggregator after confirmed training.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_manifest(package_root: Path) -> pd.DataFrame:
    path = package_root / "run_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing preflight run manifest: {path}")
    df = pd.read_csv(path)
    required = {
        "candidate",
        "channels_to_use",
        "vae_pool_composition_strategy",
        "stage_a_command",
        "stage_a_dry_run_command",
        "stage_b_command",
        "run_dir",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"run_manifest.csv is missing columns: {missing}")
    seen = tuple(df["candidate"].astype(str).tolist())
    if seen != CANDIDATES:
        raise RuntimeError(f"Unexpected candidate order in run_manifest.csv: {seen}")
    return df


def selected_rows(df: pd.DataFrame, candidate: str) -> pd.DataFrame:
    if candidate == "all":
        return df.copy()
    out = df[df["candidate"].astype(str).eq(candidate)].copy()
    if out.empty:
        raise RuntimeError(f"Unknown candidate {candidate!r}")
    return out


def stage_b_complete(run_dir: Path) -> bool:
    readout = run_dir / "classifier_only_readout"
    required = (
        readout / "classifier_sweep_pooled_metrics.csv",
        readout / "classifier_sweep_foldwise_metrics.csv",
        readout / "classifier_sweep_predictions.csv",
        readout / "classifier_sweep_thresholds_by_fold.csv",
    )
    return all(p.exists() for p in required)


def stage_a_complete(run_dir: Path) -> bool:
    required = [run_dir / f"fold_{fold}" / f"vae_model_fold_{fold}.pt" for fold in range(1, 4)]
    return all(p.exists() for p in required) and (run_dir / "run_config.json").exists()


def stale_paths(run_dir: Path) -> List[Path]:
    if not run_dir.exists():
        return []
    return [run_dir / marker for marker in STALE_MARKERS if (run_dir / marker).exists()]


def quarantine_path(run_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return run_dir.parent / f"{run_dir.name}_quarantine_{stamp}"


def prepare_run_dir(run_dir: Path, resume: bool, force_clean: bool, real_run: bool) -> Dict[str, Any]:
    markers = stale_paths(run_dir)
    status: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "exists": run_dir.exists(),
        "stale_markers": [str(p) for p in markers],
        "action": "none",
    }
    if not real_run:
        status["action"] = "dry_run_no_filesystem_change"
        return status
    if stage_b_complete(run_dir) and resume:
        status["action"] = "skip_complete_resume"
        return status
    if markers and not resume and not force_clean:
        preview = "\n".join(str(p) for p in markers[:20])
        raise RuntimeError(f"Refusing existing candidate outputs in {run_dir}. Use --resume or --force-clean.\n{preview}")
    if markers and force_clean:
        target = quarantine_path(run_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(run_dir), str(target))
        run_dir.mkdir(parents=True, exist_ok=True)
        status["action"] = "quarantined_existing_output"
        status["quarantine_path"] = str(target)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        status["action"] = "created_or_reused_empty_dir"
    return status


def validate_pool_identity(package_root: Path) -> Dict[str, Any]:
    path = package_root / "pool_composition_by_fold.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError("pool_composition_by_fold.csv is empty")
    df["classifier_pool_n"] = df["classifier_train_dev_n"].astype(int) + df["classifier_test_n"].astype(int)
    df["classifier_pool_cn"] = df["classifier_train_dev_cn"].astype(int) + df["classifier_test_cn"].astype(int)
    df["classifier_pool_ad"] = df["classifier_train_dev_ad"].astype(int) + df["classifier_test_ad"].astype(int)
    bad = df[
        ~(
            df["classifier_pool_n"].eq(396)
            & df["classifier_pool_cn"].eq(300)
            & df["classifier_pool_ad"].eq(96)
            & df["outer_test_overlap_n"].eq(0)
        )
    ]
    if not bad.empty:
        raise RuntimeError("Classifier pool identity or VAE/test overlap guard failed:\n" + bad.to_string(index=False))
    return {
        "pool_composition_file": str(path),
        "classifier_pool_n": 396,
        "classifier_pool_cn": 300,
        "classifier_pool_ad": 96,
        "outer_test_overlap_all_zero": True,
    }


def validate_manifest_rows(rows: pd.DataFrame) -> List[Dict[str, Any]]:
    checks: List[Dict[str, Any]] = []
    for _, row in rows.iterrows():
        candidate = str(row["candidate"])
        stage_a = str(row["stage_a_command"])
        stage_b = str(row["stage_b_command"])
        run_dir = resolve(Path(str(row["run_dir"])))
        errors: List[str] = []
        if "--dry-run" in shlex.split(stage_a):
            errors.append("stage_a_command unexpectedly contains --dry-run")
        if "--vae_pool_composition_strategy" not in shlex.split(stage_a):
            errors.append("stage_a_command missing --vae_pool_composition_strategy")
        if f"--vae_pool_composition_strategy {candidate}" not in stage_a:
            errors.append("stage_a_command strategy does not match candidate")
        if "--channels_to_use 1 0 2" not in stage_a:
            errors.append("stage_a_command does not use [1,0,2]")
        if "--outer_folds 3" not in stage_a or "--inner_folds 3" not in stage_a:
            errors.append("stage_a_command is not 3x3")
        if "--epochs_vae 960" not in stage_a or "--cyclical_beta_n_cycles 12" not in stage_a or "--lr_scheduler_T0 80" not in stage_a:
            errors.append("stage_a_command FAST horizon/cycles/T0 mismatch")
        if "--beta_vae 2.5" not in stage_a or "--latent_dim 256" not in stage_a or "--batch_size 64" not in stage_a:
            errors.append("stage_a_command beta/latent/batch mismatch")
        if "--dropout_rate_vae 0.15" not in stage_a:
            errors.append("stage_a_command dropout mismatch")
        if "--classifier_types logreg" not in stage_a or "--n_iter_logreg 1" not in stage_a:
            errors.append("stage_a_command is not dummy logreg n_iter=1")
        if "--metadata_features Age Sex" not in stage_a:
            errors.append("stage_a_command missing Age/Sex metadata features")
        if "OASIS" in stage_a or "oasis" in stage_a:
            errors.append("stage_a_command references OASIS")
        if LOCKED_TENSOR_STEM not in stage_a:
            errors.append("stage_a_command does not reference locked ADNI 140TR tensor")
        if "--models logreg_l2" not in stage_b or "--readout-feature-sets z_plus_age_sex" not in stage_b:
            errors.append("stage_b_command is not logreg_l2 z+Age/Sex")
        if "--outer-folds 3" not in stage_b or "--inner-folds 3" not in stage_b:
            errors.append("stage_b_command is not 3x3")
        checks.append(
            {
                "candidate": candidate,
                "run_dir": str(run_dir),
                "stage_a_complete": stage_a_complete(run_dir),
                "stage_b_complete": stage_b_complete(run_dir),
                "errors": errors,
                "ok": not errors,
            }
        )
    failed = [c for c in checks if not c["ok"]]
    if failed:
        raise RuntimeError("Manifest validation failed:\n" + json.dumps(failed, indent=2))
    return checks


def run_logged(command: str, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# Started: {now()}\n")
        log.write(f"# Command: {command}\n\n")
        log.flush()
        proc = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            shell=True,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        code = proc.wait()
        log.write(f"\n# Finished: {now()}\n# Return code: {code}\n")
    return int(code)


def append_log(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            data = [data]
    else:
        data = []
    data.append(dict(payload))
    write_json(path, data)


def main() -> int:
    args = parse_args()
    package_root = resolve(args.package_root)
    if args.dry_run and args.confirm_training:
        raise SystemExit("Use either --dry-run or --confirm-training, not both.")

    # Default is dry-run unless confirmed.
    real_run = bool(args.confirm_training)
    dry_run = bool(args.dry_run or not real_run)

    manifest = load_manifest(package_root)
    rows = selected_rows(manifest, args.candidate)
    pool_guard = validate_pool_identity(package_root)
    manifest_checks = validate_manifest_rows(rows)

    py_compile.compile(str(Path(__file__).resolve()), doraise=True)
    py_compile.compile(str(AGGREGATOR), doraise=True)

    run_log: Dict[str, Any] = {
        "timestamp": now(),
        "package_root": str(package_root),
        "candidate": args.candidate,
        "dry_run": dry_run,
        "confirm_training": real_run,
        "resume": bool(args.resume),
        "force_clean": bool(args.force_clean),
        "pool_guard": pool_guard,
        "manifest_checks": manifest_checks,
        "events": [],
    }

    print(f"Package root: {package_root}")
    print(f"Mode        : {'REAL TRAINING' if real_run else 'DRY-RUN'}")
    print(f"Candidates  : {', '.join(rows['candidate'].astype(str).tolist())}")

    for _, row in rows.iterrows():
        candidate = str(row["candidate"])
        run_dir = resolve(Path(str(row["run_dir"])))
        prep = prepare_run_dir(run_dir, resume=args.resume, force_clean=args.force_clean, real_run=real_run)
        event: Dict[str, Any] = {"candidate": candidate, "run_dir": str(run_dir), "prepare": prep}
        if stage_b_complete(run_dir) and args.resume:
            event["status"] = "skipped_complete"
            run_log["events"].append(event)
            print(f"[{candidate}] complete; skipped under --resume.")
            continue

        stage_a_command = str(row["stage_a_command"])
        stage_b_command = str(row["stage_b_command"])
        print(f"\n[{candidate}] Stage A:")
        print(stage_a_command)
        print(f"[{candidate}] Stage B:")
        print(stage_b_command)

        if dry_run:
            event["status"] = "dry_run_not_launched"
            run_log["events"].append(event)
            continue

        log_dir = package_root / "logs"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stage_a_log = log_dir / f"vae_pool_ablation_{candidate}_stageA_{stamp}.log"
        code = run_logged(stage_a_command, stage_a_log)
        event["stage_a_log"] = str(stage_a_log)
        event["stage_a_returncode"] = code
        if code != 0:
            event["status"] = "stage_a_failed"
            run_log["events"].append(event)
            append_log(package_root / "command_log_real_training.json", run_log)
            return code
        if not stage_a_complete(run_dir):
            event["status"] = "stage_a_missing_expected_outputs"
            run_log["events"].append(event)
            append_log(package_root / "command_log_real_training.json", run_log)
            raise RuntimeError(f"Stage A finished but expected fold checkpoints/run_config are missing for {candidate}")

        stage_b_log = log_dir / f"vae_pool_ablation_{candidate}_stageB_{stamp}.log"
        code = run_logged(stage_b_command, stage_b_log)
        event["stage_b_log"] = str(stage_b_log)
        event["stage_b_returncode"] = code
        if code != 0:
            event["status"] = "stage_b_failed"
            run_log["events"].append(event)
            append_log(package_root / "command_log_real_training.json", run_log)
            return code
        if not stage_b_complete(run_dir):
            event["status"] = "stage_b_missing_expected_outputs"
            run_log["events"].append(event)
            append_log(package_root / "command_log_real_training.json", run_log)
            raise RuntimeError(f"Stage B finished but expected classifier-only outputs are missing for {candidate}")
        event["status"] = "completed"
        run_log["events"].append(event)

    if real_run and not args.skip_aggregation:
        aggregate_cmd = shlex.join([sys.executable, str(AGGREGATOR), "--output-root", str(package_root)])
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        aggregate_log = package_root / "logs" / f"vae_pool_ablation_aggregate_{stamp}.log"
        code = run_logged(aggregate_cmd, aggregate_log)
        run_log["aggregation"] = {"command": aggregate_cmd, "log": str(aggregate_log), "returncode": code}
        if code != 0:
            append_log(package_root / "command_log_real_training.json", run_log)
            return code

    if dry_run:
        write_json(package_root / "command_log_launcher_dry_run.json", run_log)
        print("\nDry-run complete. No training launched.")
    else:
        append_log(package_root / "command_log_real_training.json", run_log)
        print("\nReal execution complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
