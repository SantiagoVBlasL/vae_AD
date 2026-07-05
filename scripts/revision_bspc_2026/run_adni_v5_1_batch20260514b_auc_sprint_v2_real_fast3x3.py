#!/usr/bin/env python3
"""Real launcher for AUC Sprint v2 representation FAST 3x3 candidates.

This launcher intentionally separates the executable training path from the
preparation wrapper.  It reads the previously validated Stage A dry-run command
for each candidate, removes only ``--dry-run`` for real execution, then runs the
classifier-only Stage B readout on saved fold latents.

Default behavior refuses to train.  Use ``--dry-run`` for command validation or
``--confirm-training`` for real training.
"""

from __future__ import annotations

import argparse
import json
import py_compile
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPRINT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_batch20260514b_auc_sprint_v2_representation_fast3x3"
)
PLANNED_RUNS = SPRINT_ROOT / "planned_runs.csv"
DRY_RUN_REPORT = SPRINT_ROOT / "dry_run_report.csv"
REAL_COMMAND_LOG = SPRINT_ROOT / "command_log_real_training.json"
LOG_DIR = SPRINT_ROOT / "logs"

STAGE_B_SCRIPT = (
    PROJECT_ROOT
    / "scripts"
    / "revision_bspc_2026"
    / "run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py"
)

PYTHON = "/home/diego/anaconda3/envs/vae_ad/bin/python"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
PRIMARY_MODEL = "logreg_l2"

CANDIDATES = [
    "control_current_fast3x3",
    "manufacturer_balanced_sampler_fast3x3",
    "diagnosis_manufacturer_balanced_sampler_fast3x3",
    "encoder_norm_fast3x3",
    "channel_dropout_fast3x3",
    "batch32_fast3x3",
]

LOCKED_FULL_REFERENCE = {
    "candidate_id": "locked_full_reference_not_fast",
    "run_status": "reference_only",
    "auc": 0.7788,
    "pr_auc": 0.5518,
    "balanced_accuracy": 0.7129,
    "sensitivity": 0.7292,
    "specificity": 0.6967,
    "f1": 0.5447,
    "philips_cn_fp_rate": 46 / 99,
    "ge_ad_fn_rate": 9 / 21,
    "note": "FULL 5x5 locked manuscript model; not the FAST control.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--candidate",
        choices=CANDIDATES + ["all"],
        required=True,
        help="Candidate to validate/run, or all candidates in planned order.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and print commands only; no training.")
    parser.add_argument("--confirm-training", action="store_true", help="Required for real Stage A/Stage B execution.")
    parser.add_argument("--resume", action="store_true", help="Skip completed stages and continue incomplete candidates.")
    parser.add_argument("--force", action="store_true", help="Allow rerun despite completed outputs; passes --overwrite to Stage B.")
    return parser.parse_args()


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
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.4f}" if np.isfinite(val) else "")
            elif pd.isna(val):
                vals.append("")
            else:
                vals.append(str(val).replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def write_table(stem: str, df: pd.DataFrame) -> None:
    df.to_csv(SPRINT_ROOT / f"{stem}.csv", index=False)
    (SPRINT_ROOT / f"{stem}.md").write_text(md_table(df), encoding="utf-8")


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return pd.read_csv(path)


def selected_candidates(choice: str) -> List[str]:
    return CANDIDATES if choice == "all" else [choice]


def command_arg(cmd: Sequence[str], flag: str) -> Optional[str]:
    if flag not in cmd:
        return None
    idx = list(cmd).index(flag)
    if idx + 1 >= len(cmd):
        return None
    return cmd[idx + 1]


def remove_one_token(cmd: Sequence[str], token: str) -> List[str]:
    out = list(cmd)
    if token in out:
        out.remove(token)
    return out


def load_candidate_specs() -> Dict[str, Dict[str, Any]]:
    planned = read_required_csv(PLANNED_RUNS)
    dry = read_required_csv(DRY_RUN_REPORT)
    if "candidate_id" not in planned.columns or "candidate_id" not in dry.columns:
        raise ValueError("planned_runs.csv and dry_run_report.csv must contain candidate_id.")
    dry_by_candidate = dry.set_index("candidate_id")
    specs: Dict[str, Dict[str, Any]] = {}
    for _, row in planned.iterrows():
        candidate = str(row["candidate_id"])
        if candidate not in CANDIDATES:
            continue
        if candidate not in dry_by_candidate.index:
            raise ValueError(f"Missing dry-run command for candidate {candidate}")
        dry_row = dry_by_candidate.loc[candidate]
        dry_cmd = shlex.split(str(dry_row["stage_a_command"]))
        real_stage_a = remove_one_token(dry_cmd, "--dry-run")
        stage_b = shlex.split(str(dry_row["stage_b_command_preview"]))
        specs[candidate] = {
            "planned": row.to_dict(),
            "dry_row": dry_row.to_dict(),
            "stage_a_cmd": real_stage_a,
            "stage_b_cmd": stage_b,
            "stage_a_output_dir": Path(str(row["stage_a_output_dir"])),
            "stage_b_output_dir": Path(str(row["stage_b_output_dir"])),
            "config_path": Path(str(row["config_path"])),
        }
    return specs


def validate_stage_a_command(candidate: str, cmd: Sequence[str]) -> List[str]:
    errors: List[str] = []
    joined = " ".join(cmd)
    if "--dry-run" in cmd:
        errors.append("Stage A real command still contains --dry-run.")
    for idx, token in enumerate(cmd[:-1]):
        if token.startswith("--n_iter_") and cmd[idx + 1] == "0":
            errors.append(f"Stage A command contains invalid {token}=0.")
    if "--classifier_types logreg" not in joined:
        errors.append("Stage A must run dummy canonical logreg only.")
    if "svm" in cmd:
        errors.append("Stage A command must not include svm.")
    required_pairs = {
        "--outer_folds": "3",
        "--inner_folds": "3",
        "--latent_dim": "128",
        "--epochs_vae": "960",
        "--cyclical_beta_n_cycles": "12",
        "--lr_scheduler_T0": "80",
        "--n_iter_logreg": "1",
        "--recon_loss_mode": "mse_sum_batchmean_current",
        "--vae_final_activation": "tanh",
    }
    for flag, expected in required_pairs.items():
        got = command_arg(cmd, flag)
        if got != expected:
            errors.append(f"{candidate}: expected {flag} {expected}, got {got!r}.")
    if "--classifier_stratify_cols Manufacturer" not in joined:
        errors.append("Stage A classifier split must include Manufacturer.")
    if "--vae_stratify_cols Manufacturer" not in joined:
        errors.append("Stage A VAE split must include Manufacturer.")
    if "--metadata_features Age Sex" not in joined:
        errors.append("Stage A metadata features must be Age Sex.")
    if "--classifier_stratify_cols Manufacturer Sex" in joined or "--vae_stratify_cols Manufacturer Sex" in joined:
        errors.append("Sex must not be used as a stratification column.")
    return errors


def validate_stage_b_command(candidate: str, cmd: Sequence[str]) -> List[str]:
    errors: List[str] = []
    joined = " ".join(cmd)
    if "--models logreg_l2" not in joined:
        errors.append(f"{candidate}: Stage B must use --models logreg_l2.")
    if "--outer-folds 3" not in joined or "--inner-folds 3" not in joined:
        errors.append(f"{candidate}: Stage B must use --outer-folds 3 --inner-folds 3.")
    if "--reuse-latent-cache" not in cmd:
        errors.append(f"{candidate}: Stage B must use --reuse-latent-cache.")
    return errors


def stage_a_completed(run_dir: Path, outer_folds: int = 3) -> bool:
    run_dir = run_dir.resolve() if run_dir.exists() else run_dir
    if not (run_dir / "run_config.json").exists():
        return False
    if not list(run_dir.glob("all_folds_metrics_MULTI*.csv")):
        return False
    for fold in range(1, outer_folds + 1):
        fold_dir = run_dir / f"fold_{fold}"
        required = [
            fold_dir / f"vae_model_fold_{fold}.pt",
            fold_dir / "train_dev_subjects_fold.csv",
            fold_dir / "test_subjects_fold.csv",
            fold_dir / "vae_norm_params.joblib",
        ]
        if not all(p.exists() for p in required):
            return False
    return True


def stage_b_completed(out_dir: Path) -> bool:
    required = [
        out_dir / "classifier_sweep_pooled_metrics.csv",
        out_dir / "classifier_sweep_foldwise_metrics.csv",
        out_dir / "classifier_sweep_predictions.csv",
        out_dir / "command_log.json",
    ]
    if not all(p.exists() for p in required):
        return False
    try:
        pooled = pd.read_csv(out_dir / "classifier_sweep_pooled_metrics.csv")
    except Exception:
        return False
    if pooled.empty:
        return False
    mask = pooled["model_name"].astype(str).eq(PRIMARY_MODEL) & pooled["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    return bool(mask.any())


def validate_common_preflight() -> List[str]:
    errors: List[str] = []
    try:
        py_compile.compile(str(Path(__file__).resolve()), doraise=True)
    except Exception as exc:
        errors.append(f"py_compile launcher failed: {exc}")
    for path in [PLANNED_RUNS, DRY_RUN_REPORT, STAGE_B_SCRIPT]:
        if not path.exists():
            errors.append(f"Missing required file: {path}")
    return errors


def validate_candidate_preflight(
    candidate: str,
    spec: Dict[str, Any],
    resume: bool,
    force: bool,
) -> Tuple[List[str], Dict[str, Any]]:
    errors: List[str] = []
    stage_a_cmd = list(spec["stage_a_cmd"])
    stage_b_cmd = list(spec["stage_b_cmd"])
    errors.extend(validate_stage_a_command(candidate, stage_a_cmd))
    errors.extend(validate_stage_b_command(candidate, stage_b_cmd))

    tensor = command_arg(stage_a_cmd, "--global_tensor_path")
    metadata = command_arg(stage_a_cmd, "--metadata_path")
    output_dir = command_arg(stage_a_cmd, "--output_dir")
    for label, value in [("tensor", tensor), ("metadata", metadata), ("stage_a_output_dir", output_dir)]:
        if not value:
            errors.append(f"{candidate}: missing {label} path in Stage A command.")
    if tensor and not Path(tensor).exists():
        errors.append(f"{candidate}: tensor path does not exist: {tensor}")
    if metadata and not Path(metadata).exists():
        errors.append(f"{candidate}: metadata path does not exist: {metadata}")
    if not spec["config_path"].exists():
        errors.append(f"{candidate}: config path does not exist: {spec['config_path']}")

    a_done = stage_a_completed(spec["stage_a_output_dir"])
    b_done = stage_b_completed(spec["stage_b_output_dir"])
    if (a_done or b_done) and not (resume or force):
        errors.append(
            f"{candidate}: completed outputs already exist; pass --resume to skip completed stages or --force to rerun."
        )
    status = {
        "stage_a_completed": bool(a_done),
        "stage_b_completed": bool(b_done),
        "tensor_path": tensor,
        "metadata_path": metadata,
        "stage_a_output_dir": str(spec["stage_a_output_dir"]),
        "stage_b_output_dir": str(spec["stage_b_output_dir"]),
        "config_path": str(spec["config_path"]),
    }
    return errors, status


def tee_run(cmd: Sequence[str], log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("\n" + "=" * 100 + "\n")
        log.write(f"UTC start: {datetime.now(timezone.utc).isoformat()}\n")
        log.write("COMMAND: " + shlex.join(list(cmd)) + "\n")
        log.write("=" * 100 + "\n")
        log.flush()
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        returncode = proc.wait()
        log.write(f"\nUTC end: {datetime.now(timezone.utc).isoformat()}\n")
        log.write(f"RETURN_CODE: {returncode}\n")
        log.flush()
    return int(returncode)


def stage_b_command(spec: Dict[str, Any], force: bool) -> List[str]:
    cmd = list(spec["stage_b_cmd"])
    if force and "--overwrite" not in cmd:
        cmd.append("--overwrite")
    return cmd


def load_existing_real_command_log() -> Dict[str, Any]:
    if not REAL_COMMAND_LOG.exists():
        return {"invocations": []}
    try:
        payload = json.loads(REAL_COMMAND_LOG.read_text(encoding="utf-8"))
    except Exception:
        return {"invocations": []}
    if "invocations" not in payload or not isinstance(payload["invocations"], list):
        payload = {"previous_payload": payload, "invocations": []}
    return payload


def write_real_command_log(entry: Dict[str, Any]) -> None:
    payload = load_existing_real_command_log()
    payload["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
    payload["invocations"].append(entry)
    REAL_COMMAND_LOG.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_candidate(candidate: str, spec: Dict[str, Any], resume: bool, force: bool) -> Dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"auc_sprint_v2_{candidate}_{timestamp}.log"
    result: Dict[str, Any] = {
        "candidate": candidate,
        "log_path": str(log_path),
        "stage_a": "not_started",
        "stage_b": "not_started",
        "stage_a_returncode": None,
        "stage_b_returncode": None,
    }

    if stage_a_completed(spec["stage_a_output_dir"]) and resume and not force:
        result["stage_a"] = "skipped_completed"
    else:
        rc = tee_run(spec["stage_a_cmd"], log_path, PROJECT_ROOT)
        result["stage_a_returncode"] = rc
        result["stage_a"] = "ok" if rc == 0 and stage_a_completed(spec["stage_a_output_dir"]) else "failed"
        if result["stage_a"] != "ok":
            return result

    if stage_b_completed(spec["stage_b_output_dir"]) and resume and not force:
        result["stage_b"] = "skipped_completed"
    else:
        cmd_b = stage_b_command(spec, force=force)
        rc = tee_run(cmd_b, log_path, PROJECT_ROOT)
        result["stage_b_returncode"] = rc
        result["stage_b"] = "ok" if rc == 0 and stage_b_completed(spec["stage_b_output_dir"]) else "failed"
    return result


def read_primary_pooled(candidate: str, out_dir: Path) -> Dict[str, Any]:
    if not stage_b_completed(out_dir):
        return {"candidate_id": candidate, "run_status": "not_run_or_incomplete", "note": "Missing completed Stage B readout."}
    pooled = pd.read_csv(out_dir / "classifier_sweep_pooled_metrics.csv")
    row = pooled[
        pooled["model_name"].astype(str).eq(PRIMARY_MODEL)
        & pooled["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].iloc[0]
    out = {"candidate_id": candidate, "run_status": "completed", "note": "Stage B primary readout."}
    for col in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "tn", "fp", "fn", "tp"]:
        out[col] = row.get(col, np.nan)
    return out


def aggregate_foldwise(candidate: str, out_dir: Path) -> pd.DataFrame:
    path = out_dir / "classifier_sweep_foldwise_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].copy()
    df.insert(0, "candidate_id", candidate)
    return df


def aggregate_predictions(candidate: str, out_dir: Path) -> pd.DataFrame:
    path = out_dir / "classifier_sweep_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].copy()
    df.insert(0, "candidate_id", candidate)
    return df


def binary_metrics(y_true: Sequence[int], y_score: Sequence[float], y_pred: Sequence[int]) -> Dict[str, Any]:
    from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

    y = np.asarray(y_true, dtype=int)
    score = np.asarray(y_score, dtype=float)
    pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    out = {
        "n": int(len(y)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float(np.nanmean([sensitivity, specificity])),
        "f1": float((2 * tp) / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else np.nan,
    }
    if len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, score))
        out["pr_auc"] = float(average_precision_score(y, score))
    else:
        out["auc"] = np.nan
        out["pr_auc"] = np.nan
    return out


def aggregate_manufacturer(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if pred.empty:
        return pd.DataFrame()
    for (candidate, manufacturer), sub in pred.groupby(["candidate_id", "Manufacturer"], dropna=False):
        row = {"candidate_id": candidate, "Manufacturer": manufacturer}
        row.update(binary_metrics(sub["y_true"], sub["y_score"], sub["y_pred"]))
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_error_rates(pred: pd.DataFrame, manufacturer: str, y_true: int, error_pred: int, stem: str) -> pd.DataFrame:
    rows = []
    if pred.empty:
        return pd.DataFrame()
    for candidate, sub in pred.groupby("candidate_id", dropna=False):
        target = sub[(sub["Manufacturer"].astype(str).eq(manufacturer)) & (sub["y_true"].astype(int).eq(y_true))]
        errors = target[target["y_pred"].astype(int).eq(error_pred)]
        rows.append(
            {
                "candidate_id": candidate,
                "Manufacturer": manufacturer,
                "target_class": "CN" if y_true == 0 else "AD",
                "error_type": stem,
                "n_target": int(len(target)),
                "n_errors": int(len(errors)),
                "error_rate": float(len(errors) / len(target)) if len(target) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def aggregate_vae_qc(candidate: str, run_dir: Path) -> Dict[str, Any]:
    row: Dict[str, Any] = {"candidate_id": candidate}
    latent_paths = sorted(run_dir.glob("fold_*/latent_qc_metrics.csv"))
    if latent_paths:
        frames = []
        for path in latent_paths:
            try:
                frames.append(pd.read_csv(path))
            except Exception:
                pass
        if frames:
            df = pd.concat(frames, ignore_index=True, sort=False)
            for col in ["silhouette_latent", "acc_site_latent", "acc_site_raw", "latent_dim", "beta_max"]:
                if col in df.columns:
                    row[f"{col}_mean"] = pd.to_numeric(df[col], errors="coerce").mean()
    rd_paths = sorted(run_dir.glob("fold_*/fold_*_rate_distortion.csv"))
    if rd_paths:
        frames = []
        for path in rd_paths:
            try:
                frames.append(pd.read_csv(path))
            except Exception:
                pass
        if frames:
            rd = pd.concat(frames, ignore_index=True, sort=False)
            for col in rd.columns:
                low = col.lower()
                if any(key in low for key in ["recon", "kld", "kl", "loss", "active"]):
                    vals = pd.to_numeric(rd[col], errors="coerce")
                    if vals.notna().any():
                        row[f"{col}_mean"] = vals.mean()
    row["stage_a_completed"] = stage_a_completed(run_dir)
    return row


def refresh_reports(specs: Dict[str, Dict[str, Any]]) -> None:
    main_rows = [dict(LOCKED_FULL_REFERENCE)]
    fold_frames = []
    pred_frames = []
    vae_rows = []
    for candidate in CANDIDATES:
        spec = specs[candidate]
        main = read_primary_pooled(candidate, spec["stage_b_output_dir"])
        pred = aggregate_predictions(candidate, spec["stage_b_output_dir"])
        if not pred.empty:
            philips = pred[(pred["Manufacturer"].astype(str).eq("Philips")) & (pred["y_true"].astype(int).eq(0))]
            ge_ad = pred[(pred["Manufacturer"].astype(str).eq("GE")) & (pred["y_true"].astype(int).eq(1))]
            main["philips_cn_fp_rate"] = float((philips["y_pred"].astype(int).eq(1)).mean()) if len(philips) else np.nan
            main["ge_ad_fn_rate"] = float((ge_ad["y_pred"].astype(int).eq(0)).mean()) if len(ge_ad) else np.nan
            pred_frames.append(pred)
        main_rows.append(main)
        fold = aggregate_foldwise(candidate, spec["stage_b_output_dir"])
        if not fold.empty:
            fold_frames.append(fold)
        vae_rows.append(aggregate_vae_qc(candidate, spec["stage_a_output_dir"]))

    main = pd.DataFrame(main_rows)
    write_table("main_comparison", main)

    foldwise = pd.concat(fold_frames, ignore_index=True, sort=False) if fold_frames else pd.DataFrame()
    write_table("foldwise_comparison", foldwise)

    predictions = pd.concat(pred_frames, ignore_index=True, sort=False) if pred_frames else pd.DataFrame()
    manufacturer = aggregate_manufacturer(predictions)
    write_table("manufacturer_subgroup_comparison", manufacturer)
    write_table("philips_cn_fp_comparison", aggregate_error_rates(predictions, "Philips", 0, 1, "false_positive"))
    write_table("ge_ad_fn_comparison", aggregate_error_rates(predictions, "GE", 1, 0, "false_negative"))
    write_table("vae_training_qc_comparison", pd.DataFrame(vae_rows))

    completed = main[main["run_status"].eq("completed")].copy() if "run_status" in main.columns else pd.DataFrame()
    lines = [
        "# AUC Sprint v2 Recommendation",
        "",
        "Real-training launcher refresh. Stage A canonical classifier metrics remain dummy/ignored; Stage B `logreg_l2` with true inner-CV OOF thresholding is the only ranking readout.",
        "",
    ]
    if completed.empty:
        lines.append("No FAST candidate has a completed Stage B readout yet. Do not promote any candidate.")
    else:
        best = completed.sort_values(["auc", "pr_auc"], ascending=False).iloc[0]
        lines.append(f"Best completed FAST candidate by AUC: `{best['candidate_id']}` with AUC={best['auc']:.4f}, PR-AUC={best['pr_auc']:.4f}.")
        lines.append("")
        lines.append("Apply the pre-registered promotion rule before any FULL 5x5 run: AUC delta >= +0.035 over FAST control, PR-AUC nondecreasing, improvement in at least 2/3 folds, Philips CN FP and GE AD FN not worse, and no VAE QC collapse/scanner leakage worsening.")
    (SPRINT_ROOT / "recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.dry_run and args.confirm_training:
        print("Use either --dry-run or --confirm-training, not both.", file=sys.stderr)
        return 2
    if not args.dry_run and not args.confirm_training:
        print("Refusing to train: pass --dry-run for validation or --confirm-training for real execution.", file=sys.stderr)
        return 2

    common_errors = validate_common_preflight()
    specs = load_candidate_specs()
    chosen = selected_candidates(args.candidate)
    entries: List[Dict[str, Any]] = []
    any_error = False
    for candidate in chosen:
        spec = specs[candidate]
        errors, status = validate_candidate_preflight(candidate, spec, resume=args.resume, force=args.force)
        errors = common_errors + errors
        stage_b_cmd = stage_b_command(spec, force=args.force)
        entry = {
            "candidate": candidate,
            "dry_run": bool(args.dry_run),
            "confirm_training": bool(args.confirm_training),
            "resume": bool(args.resume),
            "force": bool(args.force),
            "preflight_errors": errors,
            "status_before": status,
            "stage_a_command": shlex.join(spec["stage_a_cmd"]),
            "stage_b_command": shlex.join(stage_b_cmd),
            "tensor_modified": False,
            "metadata_modified": False,
            "ledger_modified": False,
        }
        entries.append(entry)
        print(f"\n## {candidate}")
        print(f"Stage A: {entry['stage_a_command']}")
        print(f"Stage B: {entry['stage_b_command']}")
        if errors:
            any_error = True
            print("Preflight errors:")
            for err in errors:
                print(f"- {err}")

    invocation = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "candidate_arg": args.candidate,
        "dry_run": bool(args.dry_run),
        "confirm_training": bool(args.confirm_training),
        "resume": bool(args.resume),
        "force": bool(args.force),
        "entries": entries,
    }
    if args.dry_run:
        write_real_command_log(invocation)
        if any_error:
            return 1
        print("\nDry-run validation OK. No real training launched.")
        return 0
    if any_error:
        write_real_command_log(invocation)
        return 1

    run_results = []
    for candidate in chosen:
        result = run_candidate(candidate, specs[candidate], resume=args.resume, force=args.force)
        run_results.append(result)
        if result.get("stage_a") == "failed" or result.get("stage_b") == "failed":
            break
    invocation["run_results"] = run_results
    refresh_reports(specs)
    write_real_command_log(invocation)
    if any(r.get("stage_a") == "failed" or r.get("stage_b") == "failed" for r in run_results):
        return 1
    print("\nSelected candidates completed. Refreshed comparison reports.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
