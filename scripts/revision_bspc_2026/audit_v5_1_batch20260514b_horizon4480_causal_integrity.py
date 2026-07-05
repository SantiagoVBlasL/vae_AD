#!/usr/bin/env python3
"""Read-only causal integrity audit for horizon4480_cycles56.

The goal is to distinguish stale artifact reuse from expected bit-identical
deterministic retraining when early stopping occurs before the original 3840
epoch horizon.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_LOCKED_RUN = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate"
)
DEFAULT_LOCKED_READOUT = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep"
)
DEFAULT_CANDIDATE_RUN = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5"
)
DEFAULT_CANDIDATE_READOUT = DEFAULT_CANDIDATE_RUN / "classifier_only_readout"
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_batch20260514b_horizon4480_causal_integrity_audit"
)

PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
ORIGINAL_MAX_EPOCHS = 3840
EXTENDED_MAX_EPOCHS = 4480


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Causal integrity audit for horizon4480_cycles56.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--locked-run", type=Path, default=DEFAULT_LOCKED_RUN)
    parser.add_argument("--locked-readout", type=Path, default=DEFAULT_LOCKED_READOUT)
    parser.add_argument("--candidate-run", type=Path, default=DEFAULT_CANDIDATE_RUN)
    parser.add_argument("--candidate-readout", type=Path, default=DEFAULT_CANDIDATE_READOUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path)


def clean(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null"} else text


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def parse_time(value: Any) -> Optional[datetime]:
    text = clean(value)
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_mtime_utc(path: Path) -> Optional[datetime]:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def load_history(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    obj = joblib.load(path)
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, pd.DataFrame):
        return {col: obj[col].tolist() for col in obj.columns}
    return {}


def as_float_array(values: Any) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=float)
    return np.asarray(values, dtype=float)


def history_summary(run_dir: Path, fold: int, max_epochs: int) -> Dict[str, Any]:
    path = run_dir / f"fold_{fold}" / f"vae_train_history_fold_{fold}.joblib"
    hist = load_history(path)
    val_modelsel = as_float_array(hist.get("val_loss_modelsel", hist.get("val_loss")))
    val_loss = as_float_array(hist.get("val_loss"))
    n_epochs = int(len(val_modelsel))
    if n_epochs:
        best_epoch = int(np.nanargmin(val_modelsel) + 1)
        best_val_modelsel = float(np.nanmin(val_modelsel))
    else:
        best_epoch = ""
        best_val_modelsel = np.nan
    return {
        "history_path": rel(path),
        "history_exists": path.exists(),
        "history_sha256": file_sha256(path) if path.exists() else "",
        "history_n_epochs": n_epochs,
        "best_epoch": best_epoch,
        "final_or_stop_epoch": n_epochs,
        "best_val_loss_modelsel": best_val_modelsel,
        "best_val_loss": float(val_loss[best_epoch - 1]) if n_epochs and best_epoch else np.nan,
        "stopped_before_max_epoch": bool(n_epochs and n_epochs < max_epochs),
        "reached_max_epoch": bool(n_epochs == max_epochs),
        "missing_history": not path.exists() or n_epochs == 0,
    }


def checkpoint_info(run_dir: Path, fold: int, run_start: Optional[datetime]) -> Dict[str, Any]:
    path = run_dir / f"fold_{fold}" / f"vae_model_fold_{fold}.pt"
    mtime = file_mtime_utc(path)
    return {
        "checkpoint_path": rel(path),
        "checkpoint_exists": path.exists(),
        "checkpoint_sha256": file_sha256(path) if path.exists() else "",
        "checkpoint_mtime_utc": mtime.isoformat() if mtime else "",
        "checkpoint_mtime_after_candidate_start": (
            bool(mtime and run_start and mtime >= run_start) if run_start else ""
        ),
    }


def dataframe_hash_and_summary(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {
            "path": rel(path),
            "exists": False,
            "sha256": "",
            "n_rows": "",
            "n_cols": "",
            "numeric_mean": "",
            "numeric_std": "",
        }
    df = pd.read_csv(path)
    num = df.select_dtypes(include=[np.number])
    arr = num.to_numpy(dtype=float) if not num.empty else np.asarray([], dtype=float)
    return {
        "path": rel(path),
        "exists": True,
        "sha256": file_sha256(path),
        "n_rows": int(len(df)),
        "n_cols": int(df.shape[1]),
        "numeric_mean": float(np.nanmean(arr)) if arr.size else np.nan,
        "numeric_std": float(np.nanstd(arr)) if arr.size else np.nan,
    }


def csv_numeric_equal(path_a: Path, path_b: Path) -> Tuple[bool, float, float]:
    if not path_a.exists() or not path_b.exists():
        return False, np.nan, np.nan
    a = pd.read_csv(path_a)
    b = pd.read_csv(path_b)
    if a.shape != b.shape:
        return False, np.nan, np.nan
    num_cols = [col for col in a.columns if col in b.columns and pd.api.types.is_numeric_dtype(a[col])]
    if not num_cols:
        return file_sha256(path_a) == file_sha256(path_b), 0.0, 0.0
    av = a[num_cols].to_numpy(dtype=float)
    bv = b[num_cols].to_numpy(dtype=float)
    diff = np.abs(av - bv)
    return bool(np.array_equal(av, bv)), float(np.nanmax(diff)), float(np.nanmean(diff))


def latent_identity_rows(locked_readout: Path, candidate_readout: Path, fold: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split in ["trainDev", "test"]:
        locked_path = locked_readout / "latent_cache" / f"fold_{fold}_{split}_latent_mu.csv"
        cand_path = candidate_readout / "latent_cache" / f"fold_{fold}_{split}_latent_mu.csv"
        locked_summary = dataframe_hash_and_summary(locked_path)
        cand_summary = dataframe_hash_and_summary(cand_path)
        numeric_equal, max_abs_diff, mean_abs_diff = csv_numeric_equal(locked_path, cand_path)
        rows.append(
            {
                "fold": fold,
                "split": split,
                "locked_path": locked_summary["path"],
                "candidate_path": cand_summary["path"],
                "locked_exists": locked_summary["exists"],
                "candidate_exists": cand_summary["exists"],
                "locked_sha256": locked_summary["sha256"],
                "candidate_sha256": cand_summary["sha256"],
                "sha_equal": locked_summary["sha256"] == cand_summary["sha256"],
                "numeric_equal": numeric_equal,
                "max_abs_diff": max_abs_diff,
                "mean_abs_diff": mean_abs_diff,
                "locked_n_rows": locked_summary["n_rows"],
                "candidate_n_rows": cand_summary["n_rows"],
                "locked_n_cols": locked_summary["n_cols"],
                "candidate_n_cols": cand_summary["n_cols"],
            }
        )
    return rows


def load_primary_predictions(readout_dir: Path) -> pd.DataFrame:
    path = readout_dir / "classifier_sweep_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "threshold_strategy" in df.columns:
        df = df[df["threshold_strategy"] == PRIMARY_THRESHOLD].copy()
    if "model_name" in df.columns:
        df = df[df["model_name"] == "logreg_l2"].copy()
    return df


def prediction_identity_rows(locked_readout: Path, candidate_readout: Path) -> List[Dict[str, Any]]:
    locked = load_primary_predictions(locked_readout)
    cand = load_primary_predictions(candidate_readout)
    rows: List[Dict[str, Any]] = []
    if locked.empty or cand.empty:
        for fold in range(1, 6):
            rows.append(
                {
                    "fold": fold,
                    "locked_predictions_present": not locked.empty,
                    "candidate_predictions_present": not cand.empty,
                    "n_joined": 0,
                    "score_equal": False,
                    "prediction_equal": False,
                    "threshold_equal": False,
                    "max_abs_score_diff": np.nan,
                    "mean_abs_score_diff": np.nan,
                }
            )
        return rows

    join_cols = ["SubjectID", "fold"]
    merged = locked.merge(
        cand,
        on=join_cols,
        suffixes=("_locked", "_candidate"),
        how="outer",
        indicator=True,
    )
    for fold in range(1, 6):
        sub = merged[merged["fold"] == fold].copy()
        both = sub[sub["_merge"] == "both"].copy()
        if both.empty:
            rows.append(
                {
                    "fold": fold,
                    "locked_predictions_present": True,
                    "candidate_predictions_present": True,
                    "n_joined": 0,
                    "score_equal": False,
                    "prediction_equal": False,
                    "threshold_equal": False,
                    "max_abs_score_diff": np.nan,
                    "mean_abs_score_diff": np.nan,
                }
            )
            continue
        score_diff = np.abs(both["y_score_locked"].astype(float) - both["y_score_candidate"].astype(float))
        pred_equal = bool((both["y_pred_locked"].astype(str) == both["y_pred_candidate"].astype(str)).all())
        threshold_equal = bool(
            np.array_equal(
                both["threshold_locked"].astype(float).to_numpy(),
                both["threshold_candidate"].astype(float).to_numpy(),
            )
        )
        rows.append(
            {
                "fold": fold,
                "locked_predictions_present": True,
                "candidate_predictions_present": True,
                "n_joined": int(len(both)),
                "score_equal": bool(np.all(score_diff.to_numpy() == 0.0)),
                "prediction_equal": pred_equal,
                "threshold_equal": threshold_equal,
                "locked_threshold": float(both["threshold_locked"].iloc[0]),
                "candidate_threshold": float(both["threshold_candidate"].iloc[0]),
                "max_abs_score_diff": float(score_diff.max()),
                "mean_abs_score_diff": float(score_diff.mean()),
                "locked_only_rows": int((sub["_merge"] == "left_only").sum()),
                "candidate_only_rows": int((sub["_merge"] == "right_only").sum()),
            }
        )
    return rows


def read_manifest_paths(candidate_readout: Path) -> Dict[int, Dict[str, str]]:
    manifest = read_json(candidate_readout / "latent_feature_manifest.json")
    by_fold: Dict[int, Dict[str, str]] = {}
    for item in manifest.get("folds", []):
        fold = int(item.get("fold"))
        by_fold[fold] = {
            "manifest_checkpoint": clean(item.get("checkpoint", "")),
            "manifest_trainDev_path": clean(item.get("trainDev_path", "")),
            "manifest_test_path": clean(item.get("test_path", "")),
        }
    return by_fold


def classify_fold(row: Dict[str, Any]) -> Tuple[str, str]:
    stale_reasons = []
    if row["candidate_training_launched"] is False:
        stale_reasons.append("candidate command log does not show training_launched=true")
    if row["candidate_stage_a_returncode"] not in {"", 0, "0"}:
        stale_reasons.append(f"candidate Stage A returncode is {row['candidate_stage_a_returncode']}")
    if row["candidate_stage_b_returncode"] not in {"", 0, "0"}:
        stale_reasons.append(f"candidate Stage B returncode is {row['candidate_stage_b_returncode']}")
    if not row["candidate_history_exists"]:
        stale_reasons.append("candidate history missing")
    if not row["candidate_checkpoint_exists"]:
        stale_reasons.append("candidate checkpoint missing")
    if row["checkpoint_mtime_after_candidate_start"] is False:
        stale_reasons.append("candidate checkpoint mtime older than candidate run start")
    if row["manifest_paths_point_to_locked"]:
        stale_reasons.append("latent manifest path points to locked run/readout")
    if stale_reasons:
        return "suspicious_stale_artifact", "; ".join(stale_reasons)

    if (
        row["locked_stopped_before_epoch_3840_by_early_stopping"]
        and row["candidate_stopped_same_epoch_as_locked"]
        and row["checkpoint_sha_equal"]
        and row["latent_mu_equal_all_splits"]
        and row["prediction_score_equal"]
    ):
        return (
            "expected_identical_due_to_early_stopping_before_original_horizon",
            "locked and candidate stopped at the same pre-3840 epoch with fresh candidate artifacts and identical outputs",
        )

    if (
        not row["locked_stopped_before_epoch_3840_by_early_stopping"]
        and row["candidate_stop_or_final_epoch"] > ORIGINAL_MAX_EPOCHS
        and row["checkpoint_sha_equal"]
        and row["latent_mu_equal_all_splits"]
        and row["prediction_score_equal"]
    ):
        return (
            "genuinely_changed_due_to_extended_horizon",
            "extended-horizon training executed past 3840 with fresh artifacts, but model selection kept the same best checkpoint and downstream outputs",
        )

    if (
        row["locked_stopped_before_epoch_3840_by_early_stopping"]
        and not row["candidate_stopped_same_epoch_as_locked"]
        and not row["latent_mu_equal_all_splits"]
    ):
        return (
            "genuinely_changed_due_to_extended_horizon",
            "fresh candidate rerun produced different selected checkpoint/latents/scores despite stopping before 3840; this is genuine non-stale output divergence under the candidate run",
        )

    return (
        "genuinely_changed_due_to_extended_horizon",
        "candidate artifacts are fresh and complete, and at least one of epoch horizon, checkpoint, latent mu, or prediction scores differs",
    )


def write_md_table(df: pd.DataFrame, path: Path) -> None:
    path.write_text(df.to_markdown(index=False) + "\n")


def build_final_decision(fold_df: pd.DataFrame, output_path: Path) -> None:
    n_stale = int((fold_df["causal_integrity_classification"] == "suspicious_stale_artifact").sum())
    n_expected = int(
        (
            fold_df["causal_integrity_classification"]
            == "expected_identical_due_to_early_stopping_before_original_horizon"
        ).sum()
    )
    n_changed = int((fold_df["causal_integrity_classification"] == "genuinely_changed_due_to_extended_horizon").sum())

    lines = [
        "# Final Causal Integrity Decision",
        "",
        "This is a read-only audit. It did not train models or modify tensors, metadata, ledger, configs, or model outputs.",
        "",
        f"- Folds classified as expected deterministic identity: {n_expected}",
        f"- Folds classified as genuinely changed: {n_changed}",
        f"- Folds classified as suspicious stale artifacts: {n_stale}",
        "",
    ]
    if n_stale == 0:
        lines.append(
            "No fold meets the predefined stale-artifact criteria. Candidate checkpoints exist, candidate histories are present, "
            "checkpoint mtimes are newer than the candidate run start, and the latent-feature manifest points to the candidate run/readout."
        )
    else:
        lines.append(
            "At least one fold meets stale-artifact criteria. Do not promote or interpret the candidate until those folds are rerun cleanly."
        )
    lines.extend(
        [
            "",
            "Folds can be SHA-identical without being stale when early stopping stops both runs before the original 3840-epoch limit. "
            "Under deterministic training, extending only the maximum horizon/cycle count does not alter the optimization trajectory before the stop epoch.",
            "",
            "Fold-level interpretation:",
            "",
        ]
    )
    for _, row in fold_df.iterrows():
        lines.append(
            f"- Fold {int(row['fold'])}: {row['causal_integrity_classification']} ({row['classification_reason']})."
        )
    lines.append("")
    output_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)

    candidate_manifest = read_json(args.candidate_run / "run_manifest.json")
    candidate_command_log = read_json(args.candidate_run / "command_log.json")
    run_start = parse_time(candidate_manifest.get("created_utc"))
    if run_start is None:
        run_start = parse_time(candidate_command_log.get("created_utc"))
    manifest_paths = read_manifest_paths(args.candidate_readout)

    latent_rows: List[Dict[str, Any]] = []
    for fold in range(1, 6):
        latent_rows.extend(latent_identity_rows(args.locked_readout, args.candidate_readout, fold))
    latent_df = pd.DataFrame(latent_rows)

    prediction_df = pd.DataFrame(prediction_identity_rows(args.locked_readout, args.candidate_readout))

    fold_rows: List[Dict[str, Any]] = []
    mtime_rows: List[Dict[str, Any]] = []
    early_rows: List[Dict[str, Any]] = []
    for fold in range(1, 6):
        locked_hist = history_summary(args.locked_run, fold, ORIGINAL_MAX_EPOCHS)
        cand_hist = history_summary(args.candidate_run, fold, EXTENDED_MAX_EPOCHS)
        locked_ckpt = checkpoint_info(args.locked_run, fold, run_start)
        cand_ckpt = checkpoint_info(args.candidate_run, fold, run_start)

        fold_latents = latent_df[latent_df["fold"] == fold]
        latent_equal_all = bool(
            not fold_latents.empty
            and fold_latents["numeric_equal"].all()
            and fold_latents["sha_equal"].all()
        )
        pred_row = prediction_df[prediction_df["fold"] == fold].iloc[0].to_dict()

        paths = manifest_paths.get(fold, {})
        manifest_text = " ".join(paths.values())
        manifest_points_to_locked = str(args.locked_run.resolve()) in manifest_text or str(args.locked_readout.resolve()) in manifest_text

        row: Dict[str, Any] = {
            "fold": fold,
            "candidate_training_launched": candidate_command_log.get("training_launched", ""),
            "candidate_stage_a_returncode": candidate_command_log.get("stage_a_returncode", ""),
            "candidate_stage_b_returncode": candidate_command_log.get("stage_b_returncode", ""),
            "locked_best_epoch": locked_hist["best_epoch"],
            "locked_stop_or_final_epoch": locked_hist["final_or_stop_epoch"],
            "candidate_best_epoch": cand_hist["best_epoch"],
            "candidate_stop_or_final_epoch": cand_hist["final_or_stop_epoch"],
            "locked_stopped_before_epoch_3840_by_early_stopping": locked_hist["stopped_before_max_epoch"],
            "candidate_stopped_same_epoch_as_locked": cand_hist["final_or_stop_epoch"] == locked_hist["final_or_stop_epoch"],
            "candidate_history_exists": cand_hist["history_exists"],
            "candidate_checkpoint_exists": cand_ckpt["checkpoint_exists"],
            "checkpoint_mtime_after_candidate_start": cand_ckpt["checkpoint_mtime_after_candidate_start"],
            "checkpoint_sha_equal": locked_ckpt["checkpoint_sha256"] == cand_ckpt["checkpoint_sha256"],
            "latent_mu_equal_all_splits": latent_equal_all,
            "prediction_score_equal": bool(pred_row.get("score_equal", False)),
            "prediction_threshold_equal": bool(pred_row.get("threshold_equal", False)),
            "manifest_paths_point_to_locked": manifest_points_to_locked,
            "candidate_manifest_checkpoint": paths.get("manifest_checkpoint", ""),
            "candidate_manifest_trainDev_path": paths.get("manifest_trainDev_path", ""),
            "candidate_manifest_test_path": paths.get("manifest_test_path", ""),
            "locked_checkpoint_sha256": locked_ckpt["checkpoint_sha256"],
            "candidate_checkpoint_sha256": cand_ckpt["checkpoint_sha256"],
            "locked_history_sha256": locked_hist["history_sha256"],
            "candidate_history_sha256": cand_hist["history_sha256"],
            "locked_best_val_loss_modelsel": locked_hist["best_val_loss_modelsel"],
            "candidate_best_val_loss_modelsel": cand_hist["best_val_loss_modelsel"],
            "locked_checkpoint_mtime_utc": locked_ckpt["checkpoint_mtime_utc"],
            "candidate_checkpoint_mtime_utc": cand_ckpt["checkpoint_mtime_utc"],
            "candidate_run_start_utc": run_start.isoformat() if run_start else "",
        }
        classification, reason = classify_fold(row)
        row["causal_integrity_classification"] = classification
        row["classification_reason"] = reason
        fold_rows.append(row)

        mtime_rows.append(
            {
                "fold": fold,
                "candidate_run_start_utc": row["candidate_run_start_utc"],
                "locked_checkpoint_path": locked_ckpt["checkpoint_path"],
                "locked_checkpoint_mtime_utc": locked_ckpt["checkpoint_mtime_utc"],
                "candidate_checkpoint_path": cand_ckpt["checkpoint_path"],
                "candidate_checkpoint_mtime_utc": cand_ckpt["checkpoint_mtime_utc"],
                "candidate_checkpoint_mtime_after_candidate_start": cand_ckpt["checkpoint_mtime_after_candidate_start"],
                "locked_checkpoint_sha256": locked_ckpt["checkpoint_sha256"],
                "candidate_checkpoint_sha256": cand_ckpt["checkpoint_sha256"],
                "checkpoint_sha_equal": row["checkpoint_sha_equal"],
            }
        )
        early_rows.append(
            {
                "fold": fold,
                "locked_best_epoch": locked_hist["best_epoch"],
                "locked_stop_or_final_epoch": locked_hist["final_or_stop_epoch"],
                "locked_stopped_before_epoch_3840_by_early_stopping": locked_hist["stopped_before_max_epoch"],
                "candidate_best_epoch": cand_hist["best_epoch"],
                "candidate_stop_or_final_epoch": cand_hist["final_or_stop_epoch"],
                "candidate_stopped_before_epoch_4480_by_early_stopping": cand_hist["stopped_before_max_epoch"],
                "candidate_stopped_same_epoch_as_locked": row["candidate_stopped_same_epoch_as_locked"],
                "locked_best_val_loss_modelsel": locked_hist["best_val_loss_modelsel"],
                "candidate_best_val_loss_modelsel": cand_hist["best_val_loss_modelsel"],
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    mtime_df = pd.DataFrame(mtime_rows)
    early_df = pd.DataFrame(early_rows)

    outputs = {
        "fold_causal_integrity_table": args.output_dir / "fold_causal_integrity_table.csv",
        "checkpoint_mtime_table": args.output_dir / "checkpoint_mtime_table.csv",
        "early_stopping_comparison": args.output_dir / "early_stopping_comparison.csv",
        "latent_prediction_identity_table": args.output_dir / "latent_prediction_identity_table.csv",
    }
    fold_df.to_csv(outputs["fold_causal_integrity_table"], index=False)
    mtime_df.to_csv(outputs["checkpoint_mtime_table"], index=False)
    early_df.to_csv(outputs["early_stopping_comparison"], index=False)
    latent_prediction_df = latent_df.merge(prediction_df, on="fold", how="left")
    latent_prediction_df.to_csv(outputs["latent_prediction_identity_table"], index=False)

    for key, csv_path in outputs.items():
        write_md_table(pd.read_csv(csv_path), csv_path.with_suffix(".md"))

    final_decision = args.output_dir / "final_causal_integrity_decision.md"
    build_final_decision(fold_df, final_decision)

    command_log = {
        "script": rel(Path(__file__)),
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "locked_run": rel(args.locked_run),
        "locked_readout": rel(args.locked_readout),
        "candidate_run": rel(args.candidate_run),
        "candidate_readout": rel(args.candidate_readout),
        "output_dir": rel(args.output_dir),
        "primary_threshold_strategy": PRIMARY_THRESHOLD,
        "read_only": True,
        "training_launched": False,
        "modified_tensor_metadata_ledger_config_or_model_outputs": False,
        "candidate_run_start_utc_used": run_start.isoformat() if run_start else "",
        "outputs": [rel(p) for p in list(outputs.values())]
        + [rel(p.with_suffix(".md")) for p in outputs.values()]
        + [rel(final_decision), rel(args.output_dir / "command_log.json")],
        "classification_counts": fold_df["causal_integrity_classification"].value_counts().to_dict(),
    }
    (args.output_dir / "command_log.json").write_text(json.dumps(command_log, indent=2))
    print(json.dumps(command_log, indent=2))


if __name__ == "__main__":
    main()
