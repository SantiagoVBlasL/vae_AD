#!/usr/bin/env python3
"""Read-only integrity audit for v5.1b [1] offdiag_channelmean FULL run."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"
CANDIDATE_RUN = RESULTS / "adni_v5_1b_ch1_only_offdiag_channelmean_horizon4480_cycles56_full_5x5"
CANDIDATE_READOUT = CANDIDATE_RUN / "classifier_only_readout"
REFERENCE_RUN = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5"
REFERENCE_READOUT = REFERENCE_RUN / "classifier_only_readout"
OUT_DIR = RESULTS / "adni_v5_1b_ch1_only_offdiag_channelmean_integrity_audit"
FOLDS = [1, 2, 3, 4, 5]
PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--candidate-run", type=Path, default=CANDIDATE_RUN)
    parser.add_argument("--candidate-readout", type=Path, default=CANDIDATE_READOUT)
    parser.add_argument("--reference-run", type=Path, default=REFERENCE_RUN)
    parser.add_argument("--reference-readout", type=Path, default=REFERENCE_READOUT)
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


def artifact_paths(run: Path, readout: Path) -> list[tuple[int | None, str, Path]]:
    rows: list[tuple[int | None, str, Path]] = [
        (None, "run_manifest", run / "run_manifest.json"),
        (None, "command_log", run / "command_log.json"),
        (None, "fresh_checkpoint_validation", run / "fresh_checkpoint_validation.json"),
        (None, "stageb_command_log", readout / "command_log.json"),
        (None, "stageb_predictions", readout / "classifier_sweep_predictions.csv"),
        (None, "stageb_foldwise_metrics", readout / "classifier_sweep_foldwise_metrics.csv"),
        (None, "stageb_pooled_metrics", readout / "classifier_sweep_pooled_metrics.csv"),
        (None, "stageb_thresholds", readout / "classifier_sweep_thresholds_by_fold.csv"),
    ]
    for fold in FOLDS:
        fdir = run / f"fold_{fold}"
        rows.extend(
            [
                (fold, "vae_model", fdir / f"vae_model_fold_{fold}.pt"),
                (fold, "vae_history", fdir / f"vae_train_history_fold_{fold}.joblib"),
                (fold, "vae_norm_params", fdir / "vae_norm_params.joblib"),
                (fold, "test_subjects", fdir / "test_subjects_fold.csv"),
                (fold, "train_dev_subjects", fdir / "train_dev_subjects_fold.csv"),
                (fold, "readout_test_latent_mu", readout / "latent_cache" / f"fold_{fold}_test_latent_mu.csv"),
                (fold, "readout_trainDev_latent_mu", readout / "latent_cache" / f"fold_{fold}_trainDev_latent_mu.csv"),
            ]
        )
    return rows


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
        }
        return mapping.get(artifact, ref_run / "__missing__")
    fdir = ref_run / f"fold_{fold}"
    mapping = {
        "vae_model": fdir / f"vae_model_fold_{fold}.pt",
        "vae_history": fdir / f"vae_train_history_fold_{fold}.joblib",
        "vae_norm_params": fdir / "vae_norm_params.joblib",
        "test_subjects": fdir / "test_subjects_fold.csv",
        "train_dev_subjects": fdir / "train_dev_subjects_fold.csv",
        "readout_test_latent_mu": ref_readout / "latent_cache" / f"fold_{fold}_test_latent_mu.csv",
        "readout_trainDev_latent_mu": ref_readout / "latent_cache" / f"fold_{fold}_trainDev_latent_mu.csv",
    }
    return mapping.get(artifact, ref_run / "__missing__")


def inventory(candidate_run: Path, candidate_readout: Path, reference_run: Path, reference_readout: Path, run_start: float | None) -> pd.DataFrame:
    rows = []
    for fold, artifact, path in artifact_paths(candidate_run, candidate_readout):
        exists = path.exists()
        mtime = path.stat().st_mtime if exists else None
        ref = ref_path_for(artifact, fold, reference_run, reference_readout)
        ref_exists = ref.exists()
        cand_sha = sha256_file(path) if exists and path.is_file() else ""
        ref_sha = sha256_file(ref) if ref_exists and ref.is_file() else ""
        rows.append(
            {
                "fold": fold,
                "artifact": artifact,
                "path": str(path),
                "realpath": str(path.resolve()) if exists else "",
                "exists": exists,
                "size_bytes": int(path.stat().st_size) if exists and path.is_file() else None,
                "mtime_iso": datetime.fromtimestamp(mtime).isoformat() if exists and mtime else "",
                "mtime_after_run_start": bool(exists and run_start and mtime and mtime > run_start),
                "sha256": cand_sha,
                "reference_path": str(ref),
                "reference_realpath": str(ref.resolve()) if ref_exists else "",
                "reference_realpath_same": bool(exists and ref_exists and path.resolve() == ref.resolve()),
                "reference_sha256_same": bool(cand_sha and ref_sha and cand_sha == ref_sha),
            }
        )
    return pd.DataFrame(rows)


def primary_predictions(readout: Path) -> pd.DataFrame:
    df = pd.read_csv(readout / "classifier_sweep_predictions.csv")
    return df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].copy()


def prediction_comparison(candidate: Path, reference: Path) -> pd.DataFrame:
    cand = primary_predictions(candidate)
    ref = primary_predictions(reference)
    keys = ["SubjectID", "fold"]
    score_col = "y_score"
    pred_col = "y_pred"
    merged = cand[keys + [score_col, pred_col]].merge(ref[keys + [score_col, pred_col]], on=keys, suffixes=("_candidate", "_reference"))
    rows = []
    for fold, sub in merged.groupby("fold"):
        diff = sub[f"{score_col}_candidate"].astype(float) - sub[f"{score_col}_reference"].astype(float)
        rows.append(
            {
                "fold": int(fold),
                "n_merged": int(len(sub)),
                "scores_exact_equal": bool((diff == 0).all()),
                "max_abs_score_diff": float(diff.abs().max()),
                "mean_abs_score_diff": float(diff.abs().mean()),
                "predictions_exact_equal": bool((sub[f"{pred_col}_candidate"] == sub[f"{pred_col}_reference"]).all()),
            }
        )
    return pd.DataFrame(rows)


def write_table(outdir: Path, stem: str, df: pd.DataFrame) -> None:
    df.to_csv(outdir / f"{stem}.csv", index=False)
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    (outdir / f"{stem}.md").write_text(view.to_markdown(index=False) + "\n", encoding="utf-8")


def final_decision(inv: pd.DataFrame, pred: pd.DataFrame) -> str:
    missing = int((~inv["exists"]).sum())
    shared_realpath = int(inv["reference_realpath_same"].sum())
    stale = int((inv["artifact"].eq("vae_model") & ~inv["mtime_after_run_start"]).sum())
    all_scores_equal = bool(not pred.empty and pred["scores_exact_equal"].all())
    ok = missing == 0 and shared_realpath == 0 and stale == 0
    status = "PASS" if ok else "FAIL"
    return f"""# v5.1b [1] Offdiag-Channelmean Integrity Decision

Integrity status: **{status}**.

- Missing candidate artifacts: `{missing}`
- Candidate/reference realpath collisions: `{shared_realpath}`
- VAE checkpoints not newer than candidate run start: `{stale}`
- Prediction scores exactly equal to reference across all folds: `{all_scores_equal}`

This audit is read-only. SHA identity is reported as diagnostic context; it is not sufficient by itself to declare stale reuse unless paths, mtimes, or run manifests indicate reuse.
"""


def main() -> int:
    args = parse_args()
    candidate_run = resolve(args.candidate_run)
    candidate_readout = resolve(args.candidate_readout)
    reference_run = resolve(args.reference_run)
    reference_readout = resolve(args.reference_readout)
    outdir = resolve(args.output_dir)
    if args.dry_run:
        print("Dry-run integrity preflight.")
        print(f"Candidate run    : {candidate_run}")
        print(f"Candidate readout: {candidate_readout}")
        print(f"Reference run    : {reference_run}")
        print(f"Reference readout: {reference_readout}")
        print(f"Output dir       : {outdir}")
        print("No files were written.")
        return 0

    manifest = read_json(candidate_run / "run_manifest.json")
    run_start = parse_time(manifest)
    outdir.mkdir(parents=True, exist_ok=True)
    inv = inventory(candidate_run, candidate_readout, reference_run, reference_readout, run_start)
    pred = prediction_comparison(candidate_readout, reference_readout) if (candidate_readout / "classifier_sweep_predictions.csv").exists() else pd.DataFrame()
    write_table(outdir, "file_inventory", inv)
    write_table(outdir, "prediction_score_comparison", pred)
    (outdir / "final_integrity_decision.md").write_text(final_decision(inv, pred), encoding="utf-8")
    command_log = {
        "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_run": str(candidate_run),
        "candidate_readout": str(candidate_readout),
        "reference_run": str(reference_run),
        "reference_readout": str(reference_readout),
        "output_dir": str(outdir),
        "read_only": True,
        "modified_tensor_metadata_ledger_config_or_model_outputs": False,
    }
    (outdir / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
    print(json.dumps(command_log, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
