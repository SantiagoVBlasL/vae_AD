#!/usr/bin/env python3
"""Read-only integrity audit for v5.1c horizon10000/cycles125 FULL run.

The audit distinguishes stale artifact reuse from deterministic identity. A
fold is suspicious only if artifacts are missing, paths point outside the
candidate run, mtimes predate run start, Stage B logs reference another run, or
candidate/reference realpaths are identical. SHA identity is reported but is not
alone considered stale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"
CANDIDATE_RUN = RESULTS / "adni_v5_1c_recover035_ch1_0_2_horizon10000_cycles125_full_5x5"
CANDIDATE_READOUT = CANDIDATE_RUN / "classifier_only_readout"
V51C4480_RUN = RESULTS / "adni_v5_1c_recover035_ch1_0_2_horizon4480_cycles56_full_5x5"
V51C4480_READOUT = V51C4480_RUN / "classifier_only_readout"
V51B4480_RUN = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5"
V51B4480_READOUT = V51B4480_RUN / "classifier_only_readout"
OUT_DIR = RESULTS / "adni_v5_1c_horizon10000_cycles125_integrity_audit"
FOLDS = [1, 2, 3, 4, 5]
PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--candidate-run", type=Path, default=CANDIDATE_RUN)
    parser.add_argument("--candidate-readout", type=Path, default=CANDIDATE_READOUT)
    parser.add_argument("--v51c4480-run", type=Path, default=V51C4480_RUN)
    parser.add_argument("--v51c4480-readout", type=Path, default=V51C4480_READOUT)
    parser.add_argument("--v51b4480-run", type=Path, default=V51B4480_RUN)
    parser.add_argument("--v51b4480-readout", type=Path, default=V51B4480_READOUT)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_time(payload: dict[str, Any]) -> float | None:
    raw = payload.get("created_utc") or payload.get("run_start_utc")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def file_row(path: Path, artifact: str, fold: int | None, run_start: float | None) -> dict[str, Any]:
    exists = path.exists()
    mtime = path.stat().st_mtime if exists else np.nan
    return {
        "fold": fold,
        "artifact": artifact,
        "path": str(path),
        "realpath": str(path.resolve()) if exists else "",
        "exists": exists,
        "size_bytes": int(path.stat().st_size) if exists and path.is_file() else np.nan,
        "mtime_epoch": mtime,
        "mtime_iso": datetime.fromtimestamp(mtime).isoformat() if exists else "",
        "mtime_after_run_start": bool(exists and run_start and mtime > run_start),
        "sha256": sha256_file(path) if exists and path.is_file() else "",
    }


def candidate_artifacts(run: Path, readout: Path) -> list[tuple[int | None, str, Path]]:
    items: list[tuple[int | None, str, Path]] = [
        (None, "run_manifest", run / "run_manifest.json"),
        (None, "command_log", run / "command_log.json"),
        (None, "fresh_checkpoint_validation", run / "fresh_checkpoint_validation.json"),
        (None, "stageb_command_log", readout / "command_log.json"),
        (None, "stageb_predictions", readout / "classifier_sweep_predictions.csv"),
        (None, "stageb_foldwise_metrics", readout / "classifier_sweep_foldwise_metrics.csv"),
        (None, "stageb_pooled_metrics", readout / "classifier_sweep_pooled_metrics.csv"),
        (None, "stageb_thresholds", readout / "classifier_sweep_thresholds_by_fold.csv"),
        (None, "stageb_model_status", readout / "classifier_sweep_model_status.csv"),
    ]
    for fold in FOLDS:
        fdir = run / f"fold_{fold}"
        items.extend(
            [
                (fold, "vae_model", fdir / f"vae_model_fold_{fold}.pt"),
                (fold, "vae_history", fdir / f"vae_train_history_fold_{fold}.joblib"),
                (fold, "vae_norm_params", fdir / "vae_norm_params.joblib"),
                (fold, "test_subjects", fdir / "test_subjects_fold.csv"),
                (fold, "train_dev_subjects", fdir / "train_dev_subjects_fold.csv"),
                (fold, "stagea_logreg_predictions", fdir / "test_predictions_logreg.csv"),
                (fold, "stagea_svm_predictions", fdir / "test_predictions_svm.csv"),
                (fold, "readout_test_latent_mu", readout / "latent_cache" / f"fold_{fold}_test_latent_mu.csv"),
                (fold, "readout_trainDev_latent_mu", readout / "latent_cache" / f"fold_{fold}_trainDev_latent_mu.csv"),
            ]
        )
    return items


def ref_path_for(artifact: str, fold: int | None, ref_run: Path, ref_readout: Path) -> Path:
    if fold is None:
        mapping = {
            "run_manifest": ref_run / "run_manifest.json",
            "command_log": ref_run / "command_log.json",
            "stageb_command_log": ref_readout / "command_log.json",
            "stageb_predictions": ref_readout / "classifier_sweep_predictions.csv",
            "stageb_foldwise_metrics": ref_readout / "classifier_sweep_foldwise_metrics.csv",
            "stageb_pooled_metrics": ref_readout / "classifier_sweep_pooled_metrics.csv",
            "stageb_thresholds": ref_readout / "classifier_sweep_thresholds_by_fold.csv",
            "stageb_model_status": ref_readout / "classifier_sweep_model_status.csv",
        }
        return mapping.get(artifact, ref_run / "__missing__")
    fdir = ref_run / f"fold_{fold}"
    mapping = {
        "vae_model": fdir / f"vae_model_fold_{fold}.pt",
        "vae_history": fdir / f"vae_train_history_fold_{fold}.joblib",
        "vae_norm_params": fdir / "vae_norm_params.joblib",
        "test_subjects": fdir / "test_subjects_fold.csv",
        "train_dev_subjects": fdir / "train_dev_subjects_fold.csv",
        "stagea_logreg_predictions": fdir / "test_predictions_logreg.csv",
        "stagea_svm_predictions": fdir / "test_predictions_svm.csv",
        "readout_test_latent_mu": ref_readout / "latent_cache" / f"fold_{fold}_test_latent_mu.csv",
        "readout_trainDev_latent_mu": ref_readout / "latent_cache" / f"fold_{fold}_trainDev_latent_mu.csv",
    }
    return mapping.get(artifact, ref_run / "__missing__")


def build_inventory(candidate_run: Path, candidate_readout: Path, ref_specs: list[tuple[str, Path, Path]], run_start: float | None) -> pd.DataFrame:
    rows = []
    for fold, artifact, path in candidate_artifacts(candidate_run, candidate_readout):
        row = file_row(path, artifact, fold, run_start)
        for ref_label, ref_run, ref_readout in ref_specs:
            rpath = ref_path_for(artifact, fold, ref_run, ref_readout)
            rexists = rpath.exists()
            row[f"{ref_label}_path"] = str(rpath)
            row[f"{ref_label}_realpath"] = str(rpath.resolve()) if rexists else ""
            row[f"{ref_label}_realpath_same"] = bool(row["realpath"] and rexists and row["realpath"] == str(rpath.resolve()))
            row[f"{ref_label}_sha256_same"] = bool(row["sha256"] and rexists and rpath.is_file() and row["sha256"] == sha256_file(rpath))
        rows.append(row)
    return pd.DataFrame(rows)


def read_primary_predictions(readout: Path) -> pd.DataFrame:
    df = pd.read_csv(readout / "classifier_sweep_predictions.csv")
    return df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].copy()


def prediction_identity(candidate_readout: Path, ref_specs: list[tuple[str, Path, Path]]) -> pd.DataFrame:
    cpred = read_primary_predictions(candidate_readout)
    rows = []
    for fold in FOLDS:
        cfold = cpred[cpred["fold"].astype(int).eq(fold)].sort_values("SubjectID").reset_index(drop=True)
        for ref_label, _, ref_readout in ref_specs:
            if not (ref_readout / "classifier_sweep_predictions.csv").exists():
                continue
            rpred = read_primary_predictions(ref_readout)
            rfold = rpred[rpred["fold"].astype(int).eq(fold)].sort_values("SubjectID").reset_index(drop=True)
            same_subjects = cfold["SubjectID"].astype(str).tolist() == rfold["SubjectID"].astype(str).tolist()
            if same_subjects and len(cfold):
                score_diff = cfold["y_score"].to_numpy(float) - rfold["y_score"].to_numpy(float)
                pred_same = bool(np.array_equal(cfold["y_pred"].to_numpy(int), rfold["y_pred"].to_numpy(int)))
                score_exact = bool(np.array_equal(cfold["y_score"].to_numpy(float), rfold["y_score"].to_numpy(float)))
                max_abs = float(np.max(np.abs(score_diff)))
                mean_abs = float(np.mean(np.abs(score_diff)))
            else:
                pred_same = False
                score_exact = False
                max_abs = np.nan
                mean_abs = np.nan
            rows.append(
                {
                    "fold": fold,
                    "reference": ref_label,
                    "same_subject_order_after_sort": same_subjects,
                    "score_exact_equal": score_exact,
                    "prediction_label_exact_equal": pred_same,
                    "max_abs_score_diff": max_abs,
                    "mean_abs_score_diff": mean_abs,
                    "n_candidate": int(len(cfold)),
                    "n_reference": int(len(rfold)),
                }
            )
    return pd.DataFrame(rows)


def validate_stageb_log(candidate_run: Path, candidate_readout: Path) -> dict[str, Any]:
    log = read_json(candidate_readout / "command_log.json")
    run_dir = str(Path(log.get("run_dir", "")).resolve()) if log.get("run_dir") else ""
    out_dir = str(Path(log.get("output_dir", "")).resolve()) if log.get("output_dir") else ""
    expected_run = str(candidate_run.resolve())
    expected_out = str(candidate_readout.resolve())
    return {
        "stageb_command_log_exists": bool(log),
        "stageb_run_dir": run_dir,
        "stageb_output_dir": out_dir,
        "stageb_run_dir_matches_candidate": run_dir == expected_run,
        "stageb_output_dir_matches_candidate": out_dir == expected_out,
        "threshold_selection": log.get("threshold_selection", ""),
        "latent_source": log.get("latent_source", ""),
        "models": ",".join(log.get("classifiers_requested", [])) if isinstance(log.get("classifiers_requested"), list) else "",
    }


def classify_decision(inventory: pd.DataFrame, stageb: dict[str, Any]) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not bool(inventory["exists"].all()):
        reasons.append("missing candidate artifact")
    stale_required = inventory[
        inventory["artifact"].isin(
            [
                "vae_model",
                "vae_history",
                "readout_test_latent_mu",
                "readout_trainDev_latent_mu",
                "stageb_predictions",
                "stageb_foldwise_metrics",
                "stageb_pooled_metrics",
            ]
        )
    ]
    if not bool(stale_required["mtime_after_run_start"].all()):
        reasons.append("one or more required artifacts predate candidate run start")
    for col in [c for c in inventory.columns if c.endswith("_realpath_same")]:
        if bool(inventory[col].any()):
            reasons.append(f"candidate realpath matches reference in {col}")
            break
    if not stageb.get("stageb_run_dir_matches_candidate"):
        reasons.append("Stage B command_log run_dir does not match candidate")
    if not stageb.get("stageb_output_dir_matches_candidate"):
        reasons.append("Stage B command_log output_dir does not match candidate")
    if stageb.get("models") != PRIMARY_MODEL:
        reasons.append("Stage B models are not restricted to logreg_l2")
    if "true_inner_cv_oof" not in str(stageb.get("threshold_selection")):
        reasons.append("Stage B threshold selection does not report true inner-CV OOF")
    return ("PASS" if not reasons else "FAIL", reasons)


def write_table(outdir: Path, stem: str, df: pd.DataFrame) -> None:
    df.to_csv(outdir / f"{stem}.csv", index=False)
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    (outdir / f"{stem}.md").write_text(view.to_markdown(index=False) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    candidate_run = resolve(args.candidate_run)
    candidate_readout = resolve(args.candidate_readout)
    v51c4480_run = resolve(args.v51c4480_run)
    v51c4480_readout = resolve(args.v51c4480_readout)
    v51b4480_run = resolve(args.v51b4480_run)
    v51b4480_readout = resolve(args.v51b4480_readout)
    outdir = resolve(args.output_dir)
    ref_specs = [
        ("v51c4480", v51c4480_run, v51c4480_readout),
        ("v51b4480", v51b4480_run, v51b4480_readout),
    ]
    candidate_available = (candidate_run / "run_manifest.json").exists() and (candidate_readout / "command_log.json").exists()
    if args.dry_run:
        print("Dry-run integrity preflight.")
        print(f"Candidate run    : {candidate_run}")
        print(f"Candidate readout: {candidate_readout}")
        print(f"Candidate available: {candidate_available}")
        print(f"Reference v5.1c  : {v51c4480_run}")
        print(f"Reference v5.1b  : {v51b4480_run}")
        print(f"Output dir       : {outdir}")
        print("No files were written.")
        return 0

    run_manifest = read_json(candidate_run / "run_manifest.json")
    run_start = parse_time(run_manifest)
    if run_start is None:
        raise RuntimeError("Could not parse candidate run start from run_manifest.json")
    inventory = build_inventory(candidate_run, candidate_readout, ref_specs, run_start)
    stageb = validate_stageb_log(candidate_run, candidate_readout)
    pred_identity = prediction_identity(candidate_readout, ref_specs)
    decision, reasons = classify_decision(inventory, stageb)

    outdir.mkdir(parents=True, exist_ok=True)
    write_table(outdir, "foldwise_file_inventory", inventory)
    write_table(outdir, "prediction_score_identity", pred_identity)
    stageb_df = pd.DataFrame([stageb])
    write_table(outdir, "stageb_readout_integrity", stageb_df)

    final_text = f"""# v5.1c Horizon10000/Cycles125 Integrity Decision

Decision: **{decision}**

Reasons:

{chr(10).join(f'- {r}' for r in reasons) if reasons else '- none'}

Important rule:

SHA identity with a reference run is reported but is not sufficient to call a fold stale. A fold is called stale only when artifacts are missing, mtimes predate the candidate run start, candidate realpaths point to reference outputs, or Stage B logs point outside the candidate run.

Candidate run:

`{candidate_run}`

Candidate readout:

`{candidate_readout}`
"""
    (outdir / "final_integrity_decision.md").write_text(final_text, encoding="utf-8")
    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "candidate_run": str(candidate_run),
        "candidate_readout": str(candidate_readout),
        "run_start_epoch": run_start,
        "decision": decision,
        "reasons": reasons,
        "read_only": True,
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "config_modified": False,
        "existing_model_output_modified": False,
    }
    (outdir / "command_log.json").write_text(json.dumps(command_log, indent=2) + "\n", encoding="utf-8")
    print(final_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
