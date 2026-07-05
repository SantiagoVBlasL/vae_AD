#!/usr/bin/env python3
"""Real launcher for conditional Age/Sex beta-VAE FAST 3x3 candidates.

This script intentionally separates real execution from the preflight-only
preparation script.  It reads the already validated preflight configs, rewrites
only the output paths into this experiment root, removes ``--dry-run`` from
Stage A only when ``--confirm-training`` is provided, then runs the classifier-
only Stage B readout.

Default behavior is dry-run/no training.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import py_compile
import shlex
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYTHON = "/home/diego/anaconda3/envs/vae_ad/bin/python"
PREFLIGHT_ROOT = PROJECT_ROOT / "results/revision_bspc_2026/conditional_beta_vae_age_sex_fast3x3_preflight"
OUTPUT_ROOT = PROJECT_ROOT / "results/revision_bspc_2026/conditional_beta_vae_age_sex_fast3x3"
BIG_DISK_RUN_ROOT = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026/conditional_beta_vae_age_sex_fast3x3/runs")

TRAIN_SCRIPT = PROJECT_ROOT / "scripts/run_vae_clf_ad_inference.py"
STAGE_B_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py"
AGGREGATOR_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/aggregate_conditional_beta_vae_age_sex_fast3x3.py"

PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
EXPECTED_CANDIDATES = [
    "ch1_baseline_current_z_plus_age_sex",
    "ch1_baseline_current_z_only",
    "ch1_decoder_only_sex_lambda0_z_only",
    "ch1_decoder_only_sex_lambda001_z_only",
    "ch1_decoder_only_age_sex_lambda0_z_only",
    "ch1_decoder_only_age_sex_lambda001_z_only",
    "ch1_decoder_only_age_sex_lambda001_z_plus_age_sex",
    "ch1_0_2_baseline_current_z_plus_age_sex",
    "ch1_0_2_baseline_current_z_only",
    "ch1_0_2_decoder_only_sex_lambda0_z_only",
    "ch1_0_2_decoder_only_sex_lambda001_z_only",
    "ch1_0_2_decoder_only_age_sex_lambda0_z_only",
    "ch1_0_2_decoder_only_age_sex_lambda001_z_only",
    "ch1_0_2_decoder_only_age_sex_lambda001_z_plus_age_sex",
]


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--candidate", default="all", help="Candidate id from the preflight matrix, or all.")
    parser.add_argument("--preflight-root", type=Path, default=PREFLIGHT_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--big-disk-run-root", type=Path, default=BIG_DISK_RUN_ROOT)
    parser.add_argument("--python-executable", default=PYTHON)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print commands only; no training.")
    parser.add_argument("--confirm-training", action="store_true", help="Required for real Stage A/Stage B execution.")
    parser.add_argument("--resume", action="store_true", help="Skip complete candidates and continue incomplete ones.")
    parser.add_argument("--force-clean", action="store_true", help="Move existing candidate outputs to timestamped quarantine before running.")
    parser.add_argument("--no-external-symlink", action="store_true", help="Use normal local run directories instead of symlinks to big disk.")
    parser.add_argument("--skip-aggregation", action="store_true", help="Do not run the read-only aggregator after real execution.")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def read_matrix(preflight_root: Path) -> pd.DataFrame:
    path = preflight_root / "experiment_matrix.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing validated preflight matrix: {path}")
    df = pd.read_csv(path)
    missing = sorted(set(["candidate_id", "condition_id", "channels", "vae_conditioning_mode", "vae_conditioning_vars", "corr_lambda", "readout_feature_set"]) - set(df.columns))
    if missing:
        raise ValueError(f"Preflight matrix missing columns: {missing}")
    if df["candidate_id"].tolist() != EXPECTED_CANDIDATES:
        raise ValueError("Preflight matrix candidate order does not match the validated 14-candidate plan.")
    return df


def load_preflight_config(preflight_root: Path, candidate_id: str) -> Dict[str, Any]:
    path = preflight_root / "configs" / f"{candidate_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing preflight config: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def parse_channels(value: str) -> List[int]:
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, list) or not all(isinstance(x, int) for x in parsed):
        raise ValueError(f"Invalid channels value: {value!r}")
    return parsed


def selected_rows(matrix: pd.DataFrame, candidate: str) -> pd.DataFrame:
    if candidate == "all":
        return matrix.copy()
    rows = matrix[matrix["candidate_id"].eq(candidate)].copy()
    if rows.empty:
        valid = ", ".join(matrix["candidate_id"].tolist())
        raise ValueError(f"Unknown candidate={candidate!r}. Valid: all, {valid}")
    return rows


def command_values_after(tokens: Sequence[str], flag: str) -> List[str]:
    if flag not in tokens:
        return []
    values: List[str] = []
    for token in tokens[list(tokens).index(flag) + 1 :]:
        if token.startswith("--"):
            break
        values.append(token)
    return values


def command_value(tokens: Sequence[str], flag: str) -> Optional[str]:
    values = command_values_after(tokens, flag)
    return values[0] if values else None


def replace_flag_value(tokens: List[str], flag: str, new_values: Sequence[str]) -> List[str]:
    if flag not in tokens:
        return tokens + [flag, *list(new_values)]
    idx = tokens.index(flag)
    end = idx + 1
    while end < len(tokens) and not tokens[end].startswith("--"):
        end += 1
    return tokens[: idx + 1] + list(new_values) + tokens[end:]


def remove_token(tokens: Sequence[str], token: str) -> List[str]:
    out = list(tokens)
    while token in out:
        out.remove(token)
    return out


def ensure_qc_nuisance_cols(tokens: List[str]) -> List[str]:
    return replace_flag_value(tokens, "--qc_nuisance_cols", ["Age", "Sex", "Manufacturer"])


def candidate_dirs(output_root: Path, big_disk_run_root: Path, candidate_id: str, no_external_symlink: bool) -> Dict[str, Path]:
    local_run = output_root / "runs" / candidate_id
    big_run = local_run if no_external_symlink else big_disk_run_root / candidate_id
    readout = local_run / f"classifier_only_readout_{candidate_id.split('_')[-1] if False else 'unused'}"
    return {"local_run": local_run, "big_run": big_run, "readout_placeholder": readout}


def readout_dir_for(local_run: Path, readout_feature_set: str) -> Path:
    return local_run / f"classifier_only_readout_{readout_feature_set}"


def build_commands(row: pd.Series, preflight_root: Path, output_root: Path, big_disk_run_root: Path, python_executable: str, no_external_symlink: bool, force_clean: bool) -> Dict[str, Any]:
    candidate_id = str(row["candidate_id"])
    cfg = load_preflight_config(preflight_root, candidate_id)
    channels = parse_channels(str(row["channels"]))
    dirs = candidate_dirs(output_root, big_disk_run_root, candidate_id, no_external_symlink=no_external_symlink)
    local_run = dirs["local_run"]
    readout_dir = readout_dir_for(local_run, str(row["readout_feature_set"]))

    stage_a = list(cfg["stage_a_command"])
    stage_b = list(cfg["stage_b_command"])
    stage_a[0] = python_executable
    stage_b[0] = python_executable
    stage_a = replace_flag_value(stage_a, "--output_dir", [str(local_run)])
    stage_a = replace_flag_value(stage_a, "--channels_to_use", [str(x) for x in channels])
    stage_a = ensure_qc_nuisance_cols(stage_a)
    stage_b = replace_flag_value(stage_b, "--run-dir", [str(local_run)])
    stage_b = replace_flag_value(stage_b, "--output-dir", [str(readout_dir)])
    if force_clean and "--overwrite" not in stage_b:
        stage_b.append("--overwrite")
    return {
        "candidate_id": candidate_id,
        "condition_id": str(row["condition_id"]),
        "channels": channels,
        "channel_names": str(row["channel_names"]),
        "vae_conditioning_mode": str(row["vae_conditioning_mode"]),
        "vae_conditioning_vars": str(row["vae_conditioning_vars"]),
        "corr_lambda": float(row["corr_lambda"]),
        "readout_feature_set": str(row["readout_feature_set"]),
        "local_run_dir": local_run,
        "big_run_dir": dirs["big_run"],
        "readout_dir": readout_dir,
        "stage_a_dry_cmd": stage_a,
        "stage_a_real_cmd": remove_token(stage_a, "--dry-run"),
        "stage_b_cmd": stage_b,
    }


def validate_stage_a_command(spec: Dict[str, Any], real: bool) -> List[str]:
    errors: List[str] = []
    cmd = list(spec["stage_a_real_cmd"] if real else spec["stage_a_dry_cmd"])
    if real and "--dry-run" in cmd:
        errors.append(f"{spec['candidate_id']}: real Stage A command still contains --dry-run.")
    if not real and "--dry-run" not in cmd:
        errors.append(f"{spec['candidate_id']}: dry-run Stage A command lacks --dry-run.")
    expected_pairs = {
        "--classifier_types": ["logreg"],
        "--n_iter_logreg": ["1"],
        "--outer_folds": ["3"],
        "--inner_folds": ["3"],
        "--epochs_vae": ["960"],
        "--cyclical_beta_n_cycles": ["12"],
        "--lr_scheduler_T0": ["80"],
        "--latent_dim": ["256"],
        "--beta_vae": ["2.5"],
        "--dropout_rate_vae": ["0.15"],
        "--vae_dropout_scope": ["legacy_all"],
        "--vae_block_order": ["legacy_act_norm"],
        "--vae_final_activation": ["tanh"],
        "--recon_loss_mode": ["offdiag_channelmean_sum"],
        "--metadata_features": ["Age", "Sex"],
        "--vae_conditioning_mode": [spec["vae_conditioning_mode"]],
        "--vae_conditioning_vars": [spec["vae_conditioning_vars"]],
        "--vae_latent_covariate_corr_lambda": [str(spec["corr_lambda"])],
        "--qc_nuisance_cols": ["Age", "Sex", "Manufacturer"],
    }
    for flag, expected in expected_pairs.items():
        observed = command_values_after(cmd, flag)
        if observed != expected:
            errors.append(f"{spec['candidate_id']}: expected {flag} {expected}, got {observed}.")
    if "Manufacturer" not in command_values_after(cmd, "--classifier_stratify_cols"):
        errors.append(f"{spec['candidate_id']}: classifier_stratify_cols must include Manufacturer.")
    if "Manufacturer" not in command_values_after(cmd, "--vae_stratify_cols"):
        errors.append(f"{spec['candidate_id']}: vae_stratify_cols must include Manufacturer.")
    if "Sex" in command_values_after(cmd, "--classifier_stratify_cols") or "Sex" in command_values_after(cmd, "--vae_stratify_cols"):
        errors.append(f"{spec['candidate_id']}: Sex must remain metadata/covariate only, not stratifier.")
    forbidden = ["--n_iter_svm", "--n_iter_rf", "--n_iter_gb", "--n_iter_xgb", "--n_iter_mlp"]
    present = [flag for flag in forbidden if flag in cmd]
    if present:
        errors.append(f"{spec['candidate_id']}: Stage A contains forbidden classifier trial flags: {present}.")
    for idx, token in enumerate(cmd[:-1]):
        if token.startswith("--n_iter_") and cmd[idx + 1] == "0":
            errors.append(f"{spec['candidate_id']}: invalid zero Optuna trials: {token} 0.")
    tensor = command_value(cmd, "--global_tensor_path")
    metadata = command_value(cmd, "--metadata_path")
    if not tensor or not Path(tensor).exists():
        errors.append(f"{spec['candidate_id']}: tensor path missing: {tensor}")
    if not metadata or not Path(metadata).exists():
        errors.append(f"{spec['candidate_id']}: metadata path missing: {metadata}")
    return errors


def validate_stage_b_command(spec: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    cmd = list(spec["stage_b_cmd"])
    expected = {
        "--models": [PRIMARY_MODEL],
        "--readout-feature-sets": [spec["readout_feature_set"]],
        "--outer-folds": ["3"],
        "--inner-folds": ["3"],
    }
    for flag, values in expected.items():
        observed = command_values_after(cmd, flag)
        if observed != values:
            errors.append(f"{spec['candidate_id']}: expected Stage B {flag} {values}, got {observed}.")
    if "--reuse-latent-cache" not in cmd:
        errors.append(f"{spec['candidate_id']}: Stage B must include --reuse-latent-cache.")
    return errors


def validate_specs(specs: Sequence[Dict[str, Any]], real: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    try:
        py_compile.compile(str(Path(__file__).resolve()), doraise=True)
    except Exception as exc:
        rows.append({"candidate_id": "__launcher__", "status": "error", "message": f"py_compile failed: {exc}"})
    for path in [TRAIN_SCRIPT, STAGE_B_SCRIPT, AGGREGATOR_SCRIPT]:
        if not path.exists():
            rows.append({"candidate_id": "__common__", "status": "error", "message": f"Missing script: {path}"})
    for spec in specs:
        errors = validate_stage_a_command(spec, real=real)
        errors.extend(validate_stage_b_command(spec))
        status = "ok" if not errors else "error"
        rows.append({
            "candidate_id": spec["candidate_id"],
            "condition_id": spec["condition_id"],
            "channels": json.dumps(spec["channels"]),
            "vae_conditioning_mode": spec["vae_conditioning_mode"],
            "vae_conditioning_vars": spec["vae_conditioning_vars"],
            "corr_lambda": spec["corr_lambda"],
            "readout_feature_set": spec["readout_feature_set"],
            "local_run_dir": str(spec["local_run_dir"]),
            "big_run_dir": str(spec["big_run_dir"]),
            "readout_dir": str(spec["readout_dir"]),
            "status": status,
            "message": "; ".join(errors),
        })
    return rows


def stage_a_complete(run_dir: Path) -> bool:
    if not (run_dir / "run_config.json").exists():
        return False
    if not list(run_dir.glob("all_folds_metrics_MULTI*.csv")):
        return False
    for fold in range(1, 4):
        fold_dir = run_dir / f"fold_{fold}"
        required = [
            fold_dir / f"vae_model_fold_{fold}.pt",
            fold_dir / "vae_norm_params.joblib",
            fold_dir / "train_dev_subjects_fold.csv",
            fold_dir / "test_subjects_fold.csv",
        ]
        if not all(p.exists() for p in required):
            return False
    return True


def stage_b_complete(readout_dir: Path) -> bool:
    required = [
        readout_dir / "classifier_sweep_pooled_metrics.csv",
        readout_dir / "classifier_sweep_foldwise_metrics.csv",
        readout_dir / "classifier_sweep_predictions.csv",
        readout_dir / "classifier_sweep_thresholds_by_fold.csv",
    ]
    if not all(p.exists() for p in required):
        return False
    try:
        pooled = pd.read_csv(readout_dir / "classifier_sweep_pooled_metrics.csv")
    except Exception:
        return False
    if "readout_feature_set" not in pooled.columns:
        pooled["readout_feature_set"] = "z_plus_age_sex"
    mask = (
        pooled["model_name"].astype(str).eq(PRIMARY_MODEL)
        & pooled["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    )
    return bool(mask.any())


def timestamped_quarantine(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = path.parent / f"{path.name}_quarantine_{stamp}"
    idx = 1
    while candidate.exists() or candidate.is_symlink():
        idx += 1
        candidate = path.parent / f"{path.name}_quarantine_{stamp}_{idx}"
    return candidate


def symlink_target_matches(link_path: Path, target_path: Path) -> bool:
    if not link_path.is_symlink():
        return False
    raw = Path(os.readlink(link_path))
    if not raw.is_absolute():
        raw = link_path.parent / raw
    try:
        return raw.resolve() == target_path.resolve()
    except FileNotFoundError:
        return raw.absolute() == target_path.absolute()


def prepare_run_path(spec: Dict[str, Any], force_clean: bool, no_external_symlink: bool) -> List[str]:
    local_run = spec["local_run_dir"]
    big_run = spec["big_run_dir"]
    quarantines: List[str] = []
    local_run.parent.mkdir(parents=True, exist_ok=True)
    if no_external_symlink:
        if local_run.exists() and force_clean:
            q = timestamped_quarantine(local_run)
            shutil.move(str(local_run), str(q))
            quarantines.append(str(q))
        local_run.mkdir(parents=True, exist_ok=True)
        return quarantines

    big_run.mkdir(parents=True, exist_ok=True)
    if local_run.is_symlink():
        if not symlink_target_matches(local_run, big_run):
            if not force_clean:
                raise RuntimeError(f"{local_run} points to {os.readlink(local_run)}, expected {big_run}; use --force-clean.")
            q = timestamped_quarantine(local_run)
            shutil.move(str(local_run), str(q))
            quarantines.append(str(q))
    elif local_run.exists():
        if not force_clean:
            raise RuntimeError(f"Refusing existing non-symlink run dir: {local_run}; use --force-clean.")
        q = timestamped_quarantine(local_run)
        shutil.move(str(local_run), str(q))
        quarantines.append(str(q))

    if not local_run.exists() and not local_run.is_symlink():
        local_run.symlink_to(big_run, target_is_directory=True)
    if not symlink_target_matches(local_run, big_run):
        raise RuntimeError(f"Failed to prepare symlink {local_run} -> {big_run}")
    if force_clean and any(big_run.iterdir()):
        q = timestamped_quarantine(big_run)
        q.mkdir(parents=True)
        for child in list(big_run.iterdir()):
            shutil.move(str(child), str(q / child.name))
        quarantines.append(str(q))
    return quarantines


def verify_fresh_stage_a(run_dir: Path, started_at: float) -> None:
    stale: List[str] = []
    missing: List[str] = []
    for fold in range(1, 4):
        ckpt = run_dir / f"fold_{fold}" / f"vae_model_fold_{fold}.pt"
        if not ckpt.exists():
            missing.append(str(ckpt))
        elif ckpt.stat().st_mtime < started_at:
            stale.append(str(ckpt))
    if missing or stale:
        raise RuntimeError(f"Stage A checkpoint freshness failed. Missing={missing}; stale={stale}")


def run_logged(cmd: Sequence[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n" + "=" * 100 + "\n")
        log.write(f"UTC start: {now_utc()}\n")
        log.write("COMMAND: " + shlex.join(list(cmd)) + "\n")
        log.flush()
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        code = proc.wait()
        log.write(f"\nUTC end: {now_utc()}\nEXIT_CODE: {code}\n")
    return int(code)


def write_markdown_table(path: Path, df: pd.DataFrame) -> None:
    if df.empty:
        path.write_text("_No rows._\n", encoding="utf-8")
    else:
        path.write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")


def write_manifest(output_root: Path, rows: List[Dict[str, Any]], dry_run: bool) -> pd.DataFrame:
    output_root.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(output_root / "run_manifest.csv", index=False)
    write_markdown_table(output_root / "run_manifest.md", df)
    lines = [
        "# Conditional beta-VAE Age/Sex FAST 3x3",
        "",
        f"Generated UTC: {now_utc()}",
        f"Dry-run only: {dry_run}",
        "",
        "Stage A trains the FAST VAE with dummy canonical logreg (`n_iter_logreg=1`), ignored for ranking.",
        "Stage B is classifier-only `logreg_l2` with true inner-CV OOF thresholds.",
        "",
    ]
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return df


def main() -> int:
    args = parse_args()
    if args.confirm_training and args.dry_run:
        raise SystemExit("Use either --dry-run or --confirm-training, not both.")
    if not args.confirm_training:
        args.dry_run = True

    preflight_root = resolve(args.preflight_root)
    output_root = resolve(args.output_root)
    big_root = args.big_disk_run_root if args.big_disk_run_root.is_absolute() else resolve(args.big_disk_run_root)
    matrix = read_matrix(preflight_root)
    rows = selected_rows(matrix, args.candidate)
    specs = [
        build_commands(
            row,
            preflight_root=preflight_root,
            output_root=output_root,
            big_disk_run_root=big_root,
            python_executable=args.python_executable,
            no_external_symlink=args.no_external_symlink,
            force_clean=args.force_clean,
        )
        for _, row in rows.iterrows()
    ]
    validation_rows = validate_specs(specs, real=args.confirm_training)
    validation = pd.DataFrame(validation_rows)
    output_root.mkdir(parents=True, exist_ok=True)
    validation.to_csv(output_root / "launcher_validation.csv", index=False)
    write_markdown_table(output_root / "launcher_validation.md", validation)
    if validation["status"].eq("error").any():
        print(validation.to_string(index=False))
        raise SystemExit("Launcher validation failed; no training launched.")

    manifest_rows: List[Dict[str, Any]] = []
    for spec in specs:
        a_done = stage_a_complete(spec["local_run_dir"])
        b_done = stage_b_complete(spec["readout_dir"])
        if (a_done or b_done) and not args.dry_run and not (args.resume or args.force_clean):
            raise SystemExit(f"{spec['candidate_id']} already has outputs; use --resume or --force-clean.")
        manifest_rows.append({
            "candidate_id": spec["candidate_id"],
            "condition_id": spec["condition_id"],
            "channels": json.dumps(spec["channels"]),
            "channel_names": spec["channel_names"],
            "vae_conditioning_mode": spec["vae_conditioning_mode"],
            "vae_conditioning_vars": spec["vae_conditioning_vars"],
            "corr_lambda": spec["corr_lambda"],
            "readout_feature_set": spec["readout_feature_set"],
            "run_dir": str(spec["local_run_dir"]),
            "big_run_dir": str(spec["big_run_dir"]),
            "readout_dir": str(spec["readout_dir"]),
            "stage_a_completed_before": bool(a_done),
            "stage_b_completed_before": bool(b_done),
            "stage_a_command": shlex.join(spec["stage_a_real_cmd"]),
            "stage_b_command": shlex.join(spec["stage_b_cmd"]),
        })
    write_manifest(output_root, manifest_rows, dry_run=args.dry_run)

    command_log: Dict[str, Any] = {
        "created_utc": now_utc(),
        "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "preflight_root": str(preflight_root),
        "output_root": str(output_root),
        "big_disk_run_root": str(big_root),
        "candidate": args.candidate,
        "dry_run": bool(args.dry_run),
        "confirm_training": bool(args.confirm_training),
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "existing_model_outputs_modified": False,
        "candidates": [],
    }

    if args.dry_run:
        for spec in specs:
            print(f"\n## {spec['candidate_id']}")
            print("Stage A real command preview:")
            print(shlex.join(spec["stage_a_real_cmd"]))
            print("Stage B command preview:")
            print(shlex.join(spec["stage_b_cmd"]))
        (output_root / "command_log_launcher.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
        return 0

    command_log["training_launched"] = True
    for spec in specs:
        run_record: Dict[str, Any] = {"candidate_id": spec["candidate_id"], "started_utc": now_utc()}
        if args.resume and stage_b_complete(spec["readout_dir"]):
            run_record["status"] = "skipped_complete"
            command_log["candidates"].append(run_record)
            continue

        quarantines = prepare_run_path(spec, force_clean=args.force_clean, no_external_symlink=args.no_external_symlink)
        run_record["quarantines"] = quarantines
        log_path = output_root / "logs" / f"conditional_fast3x3_{spec['candidate_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        if not (args.resume and stage_a_complete(spec["local_run_dir"])):
            started = time.time()
            code = run_logged(spec["stage_a_real_cmd"], log_path)
            run_record["stage_a_exit_code"] = code
            if code != 0:
                run_record["status"] = "stage_a_failed"
                command_log["candidates"].append(run_record)
                (output_root / "command_log_launcher.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
                raise SystemExit(f"Stage A failed for {spec['candidate_id']} with code {code}")
            verify_fresh_stage_a(spec["local_run_dir"], started)
        else:
            run_record["stage_a_exit_code"] = "skipped_resume_complete"

        code = run_logged(spec["stage_b_cmd"], log_path)
        run_record["stage_b_exit_code"] = code
        if code != 0:
            run_record["status"] = "stage_b_failed"
            command_log["candidates"].append(run_record)
            (output_root / "command_log_launcher.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
            raise SystemExit(f"Stage B failed for {spec['candidate_id']} with code {code}")
        run_record["status"] = "completed"
        run_record["finished_utc"] = now_utc()
        command_log["candidates"].append(run_record)
        (output_root / "command_log_launcher.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")

    (output_root / "command_log_launcher.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
    if not args.skip_aggregation:
        aggregate_cmd = [
            args.python_executable,
            str(AGGREGATOR_SCRIPT),
            "--output-root",
            str(output_root),
            "--preflight-root",
            str(preflight_root),
        ]
        print("Running read-only aggregation:")
        print(shlex.join(aggregate_cmd))
        code = run_logged(aggregate_cmd, output_root / "logs" / f"conditional_fast3x3_aggregate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        if code != 0:
            raise SystemExit(f"Aggregation failed with code {code}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
