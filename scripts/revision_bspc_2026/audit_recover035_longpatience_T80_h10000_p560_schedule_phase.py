#!/usr/bin/env python3
"""Schedule-phase audit for the recover035_longpatience_T80_h10000_p560 experiment.

Pre-training (--dry-run or when training history is absent):
  - Verifies beta cycle length = 10000/125 = 80 epochs from config.
  - Verifies lr_scheduler_T0 = 80 from config (same as recover035 and locked).
  - Verifies beta and LR cycles are phase-aligned at epoch 0 (both start at 0.0).
  - Verifies patience = 560 = 7 × 80-epoch cycles.
  - Reports complete cycle structure: ramp epochs, stable epochs, restart points.
  - Compares cycle structure against locked (4480/56=80) and recover035 (4480/56=80):
    cycle length is IDENTICAL; only horizon and patience change.

Post-training (when VAE training histories are available):
  - Loads val_loss_modelsel history per fold.
  - Identifies best epoch (argmin of val_loss_modelsel).
  - Computes beta phase fraction and LR phase fraction at best epoch.
  - Reports beta region (ramping_beta / stable_beta) at best epoch.
  - Compares fold-level best_epoch and pct_horizon_used vs locked and recover035.
  - Key question: does longpatience use more of the 10000-epoch horizon?

Post-Stage-B (when Stage B readout is available):
  - Reports foldwise Stage B AUC/PR-AUC.
  - Compares against locked and recover035 references.

Does not train, score, or modify any existing files.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"

DEFAULT_CONFIG = PROJECT_ROOT / "configs/runs/adni_v5_1c_recover035_longpatience_T80_h10000_p560_full5x5.json"
DEFAULT_RUN_DIR = RESULTS / "recover035_longpatience_T80_h10000_p560_full5x5"
DEFAULT_READOUT_DIR = DEFAULT_RUN_DIR / "classifier_only_readout"
DEFAULT_LOCKED_RUN = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5"
DEFAULT_LOCKED_READOUT = DEFAULT_LOCKED_RUN / "classifier_only_readout"
DEFAULT_RECOVER035_RUN = RESULTS / "recover035_full5x5"
DEFAULT_RECOVER035_READOUT = DEFAULT_RECOVER035_RUN / "classifier_only_readout"
DEFAULT_OUTPUT = RESULTS / "recover035_longpatience_T80_h10000_p560_schedule_phase_audit"

PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
FOLDS = [1, 2, 3, 4, 5]

EXPECTED_EPOCHS_VAE = 10000
EXPECTED_N_CYCLES = 125
EXPECTED_CYCLE_LEN = 80
EXPECTED_T0 = 80
EXPECTED_PATIENCE = 560
CYCLICAL_BETA_RATIO_INCREASE = 0.4   # fraction of cycle spent ramping beta up

# Reference schedules (for comparison)
LOCKED_EPOCHS_VAE = 4480
LOCKED_N_CYCLES = 56
LOCKED_CYCLE_LEN = 80
LOCKED_T0 = 80
LOCKED_PATIENCE = 320

RECOVER035_EPOCHS_VAE = 4480
RECOVER035_N_CYCLES = 56
RECOVER035_CYCLE_LEN = 80
RECOVER035_T0 = 80
RECOVER035_PATIENCE = 320


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--readout-dir", type=Path, default=DEFAULT_READOUT_DIR)
    parser.add_argument("--locked-run-dir", type=Path, default=DEFAULT_LOCKED_RUN)
    parser.add_argument("--locked-readout-dir", type=Path, default=DEFAULT_LOCKED_READOUT)
    parser.add_argument("--recover035-run-dir", type=Path, default=DEFAULT_RECOVER035_RUN)
    parser.add_argument("--recover035-readout-dir", type=Path, default=DEFAULT_RECOVER035_READOUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Config-level checks only; no artifact files are read and no output files are written.",
    )
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
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


def write_table(outdir: Path, stem: str, df: pd.DataFrame, max_rows: int = 20) -> None:
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def cycle_structure(epochs: int, n_cycles: int, ratio_increase: float) -> pd.DataFrame:
    cycle_len = epochs / n_cycles
    ramp_epochs = int(round(cycle_len * ratio_increase))
    stable_epochs = int(round(cycle_len * (1 - ratio_increase)))
    rows: List[Dict[str, Any]] = []
    for i in range(n_cycles):
        start = int(round(i * cycle_len))
        ramp_end = start + ramp_epochs
        end = int(round((i + 1) * cycle_len))
        rows.append({
            "cycle": i + 1,
            "start_epoch": start,
            "ramp_end_epoch": ramp_end,
            "end_epoch": end,
            "cycle_len": end - start,
            "ramp_len": ramp_epochs,
            "stable_len": stable_epochs,
        })
    return pd.DataFrame(rows)


def validate_config_schedule(config: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    params = config["parameters"]
    epochs = params.get("epochs_vae")
    n_cycles = params.get("cyclical_beta_n_cycles")
    t0 = params.get("lr_scheduler_T0")
    patience = params.get("early_stopping_patience_vae")

    if epochs != EXPECTED_EPOCHS_VAE:
        errors.append(f"epochs_vae={epochs}, expected {EXPECTED_EPOCHS_VAE}")
    if n_cycles != EXPECTED_N_CYCLES:
        errors.append(f"cyclical_beta_n_cycles={n_cycles}, expected {EXPECTED_N_CYCLES}")
    if t0 != EXPECTED_T0:
        errors.append(f"lr_scheduler_T0={t0}, expected {EXPECTED_T0}")
    if patience != EXPECTED_PATIENCE:
        errors.append(f"early_stopping_patience_vae={patience}, expected {EXPECTED_PATIENCE}")
    if epochs is not None and n_cycles is not None:
        cycle_len = epochs / n_cycles
        if cycle_len != EXPECTED_CYCLE_LEN:
            errors.append(f"cycle_len={cycle_len}, expected {EXPECTED_CYCLE_LEN}")
        if t0 is not None and cycle_len != t0:
            errors.append(f"cycle_len ({cycle_len}) != lr_scheduler_T0 ({t0}) — phases not aligned")
    if patience is not None:
        if patience % EXPECTED_CYCLE_LEN != 0:
            errors.append(f"patience={patience} not a multiple of cycle_len={EXPECTED_CYCLE_LEN}")
        else:
            n_pc = patience // EXPECTED_CYCLE_LEN
            if n_pc != 7:
                errors.append(f"patience={patience} = {n_pc} cycles, expected 7")
    return errors


def beta_phase_at_epoch(epoch: int, cycle_len: int, ratio_increase: float) -> Dict[str, Any]:
    cycle_idx = epoch // cycle_len
    pos_in_cycle = epoch % cycle_len
    ramp_end = int(round(cycle_len * ratio_increase))
    frac = pos_in_cycle / cycle_len
    region = "ramping_beta" if pos_in_cycle < ramp_end else "stable_beta"
    return {
        "epoch": epoch,
        "cycle_idx": cycle_idx,
        "pos_in_cycle": pos_in_cycle,
        "cycle_frac": frac,
        "beta_region": region,
    }


def load_training_history(fold_dir: Path, fold: int) -> Optional[pd.DataFrame]:
    candidates = sorted(fold_dir.glob(f"vae_training_history_fold_{fold}*.csv"))
    if not candidates:
        return None
    return pd.read_csv(candidates[0])


def find_best_epoch(hist: pd.DataFrame) -> Optional[int]:
    for col in ["val_loss_modelsel", "val_loss", "val_total_loss"]:
        if col in hist.columns:
            valid = hist[col].dropna()
            if len(valid) > 0:
                return int(valid.idxmin())
    return None


def per_fold_maturity(run_dir: Path, epochs_vae: int, n_cycles: int, ratio_increase: float) -> pd.DataFrame:
    cycle_len = epochs_vae / n_cycles
    rows: List[Dict[str, Any]] = []
    for fold in FOLDS:
        fold_dir = run_dir / f"fold_{fold}"
        hist = load_training_history(fold_dir, fold)
        row: Dict[str, Any] = {
            "fold": fold, "best_epoch": None, "pct_horizon": None,
            "cycle_at_best": None, "beta_region_at_best": None, "note": "",
        }
        if hist is None:
            row["note"] = "training history not found"
        else:
            best_ep = find_best_epoch(hist)
            if best_ep is None:
                row["note"] = "no valid val_loss column"
            else:
                row["best_epoch"] = best_ep
                row["pct_horizon"] = float(best_ep / epochs_vae)
                phase = beta_phase_at_epoch(best_ep, int(cycle_len), ratio_increase)
                row["cycle_at_best"] = phase["cycle_idx"] + 1
                row["beta_region_at_best"] = phase["beta_region"]
                epochs_after_best = epochs_vae - best_ep
                row["epochs_after_best"] = int(epochs_after_best)
                row["patience_margin"] = int(epochs_after_best - EXPECTED_PATIENCE)
        rows.append(row)
    return pd.DataFrame(rows)


def foldwise_from_readout(readout: Path, run_id: str) -> Optional[pd.DataFrame]:
    path = readout / "classifier_sweep_foldwise_metrics.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    ].copy()
    df.insert(0, "run_id", run_id)
    return df if not df.empty else None


def schedule_comparison_table() -> pd.DataFrame:
    rows = [
        {
            "run": "locked_horizon4480_cycles56",
            "epochs_vae": LOCKED_EPOCHS_VAE,
            "n_cycles": LOCKED_N_CYCLES,
            "cycle_len": LOCKED_CYCLE_LEN,
            "lr_T0": LOCKED_T0,
            "patience": LOCKED_PATIENCE,
            "patience_cycles": LOCKED_PATIENCE // LOCKED_CYCLE_LEN,
            "phase_aligned": LOCKED_CYCLE_LEN == LOCKED_T0,
        },
        {
            "run": "recover035_full5x5",
            "epochs_vae": RECOVER035_EPOCHS_VAE,
            "n_cycles": RECOVER035_N_CYCLES,
            "cycle_len": RECOVER035_CYCLE_LEN,
            "lr_T0": RECOVER035_T0,
            "patience": RECOVER035_PATIENCE,
            "patience_cycles": RECOVER035_PATIENCE // RECOVER035_CYCLE_LEN,
            "phase_aligned": RECOVER035_CYCLE_LEN == RECOVER035_T0,
        },
        {
            "run": "recover035_longpatience_T80_h10000_p560",
            "epochs_vae": EXPECTED_EPOCHS_VAE,
            "n_cycles": EXPECTED_N_CYCLES,
            "cycle_len": EXPECTED_CYCLE_LEN,
            "lr_T0": EXPECTED_T0,
            "patience": EXPECTED_PATIENCE,
            "patience_cycles": EXPECTED_PATIENCE // EXPECTED_CYCLE_LEN,
            "phase_aligned": EXPECTED_CYCLE_LEN == EXPECTED_T0,
        },
    ]
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    config_path = resolve(args.config)
    run_dir = resolve(args.run_dir)
    readout_dir = resolve(args.readout_dir)
    locked_run_dir = resolve(args.locked_run_dir)
    locked_readout = resolve(args.locked_readout_dir)
    recover035_run = resolve(args.recover035_run_dir)
    recover035_readout = resolve(args.recover035_readout_dir)
    outdir = resolve(args.output_dir)

    config = load_json(config_path)
    errors = validate_config_schedule(config)

    params = config["parameters"]
    cycle_len = params["epochs_vae"] / params["cyclical_beta_n_cycles"]
    patience = params["early_stopping_patience_vae"]

    print(f"Config           : {config_path}")
    print(f"Run dir          : {run_dir}")
    print(f"Mode             : {'DRY-RUN (config checks only)' if args.dry_run else 'FULL ARTIFACT AUDIT'}")
    print()
    print("Schedule parameters:")
    print(f"  epochs_vae              = {params['epochs_vae']}")
    print(f"  cyclical_beta_n_cycles  = {params['cyclical_beta_n_cycles']}")
    print(f"  cycle_len               = {cycle_len:.0f} epochs")
    print(f"  lr_scheduler_T0         = {params['lr_scheduler_T0']} (SAME as recover035 and locked)")
    print(f"  early_stopping_patience = {patience} = {patience // int(cycle_len)} × {int(cycle_len)}-epoch cycles")
    print(f"  phase_aligned           = {cycle_len == params['lr_scheduler_T0']}")
    print()
    print("Schedule comparison (cycle length is IDENTICAL across all three runs):")
    print(schedule_comparison_table().to_string(index=False))
    print()
    ramp_epochs = int(round(cycle_len * CYCLICAL_BETA_RATIO_INCREASE))
    stable_epochs = int(round(cycle_len * (1 - CYCLICAL_BETA_RATIO_INCREASE)))
    print(f"Per-cycle structure: ramp={ramp_epochs} epochs (beta 0→max), stable={stable_epochs} epochs (beta=max)")
    print(f"LR cosine restart every {params['lr_scheduler_T0']} epochs — coincides with beta cycle start")
    print()
    if errors:
        for e in errors:
            print(f"[CONFIG ERROR] {e}")
        return 1
    else:
        print("Config schedule checks: PASS")

    if args.dry_run:
        print("\nDry-run complete. No artifact files read, no output files written.")
        return 0

    if outdir.exists() and not args.overwrite:
        raise FileExistsError(f"{outdir} exists; pass --overwrite")
    outdir.mkdir(parents=True, exist_ok=True)

    sched_cmp = schedule_comparison_table()
    write_table(outdir, "schedule_comparison", sched_cmp)

    struct_df = cycle_structure(params["epochs_vae"], params["cyclical_beta_n_cycles"], CYCLICAL_BETA_RATIO_INCREASE)
    write_table(outdir, "longpatience_cycle_structure", struct_df, max_rows=len(struct_df))

    longpatience_maturity = per_fold_maturity(run_dir, params["epochs_vae"], params["cyclical_beta_n_cycles"], CYCLICAL_BETA_RATIO_INCREASE)
    write_table(outdir, "longpatience_maturity_by_fold", longpatience_maturity)

    locked_maturity = per_fold_maturity(locked_run_dir, LOCKED_EPOCHS_VAE, LOCKED_N_CYCLES, CYCLICAL_BETA_RATIO_INCREASE)
    locked_maturity.insert(0, "run_id", "locked")
    recover035_maturity = per_fold_maturity(recover035_run, RECOVER035_EPOCHS_VAE, RECOVER035_N_CYCLES, CYCLICAL_BETA_RATIO_INCREASE)
    recover035_maturity.insert(0, "run_id", "recover035")
    longpatience_labeled = longpatience_maturity.copy()
    longpatience_labeled.insert(0, "run_id", "longpatience")
    maturity_all = pd.concat([locked_maturity, recover035_maturity, longpatience_labeled], ignore_index=True)
    write_table(outdir, "maturity_comparison_all", maturity_all, max_rows=30)

    fw_locked = foldwise_from_readout(locked_readout, "locked")
    fw_recover035 = foldwise_from_readout(recover035_readout, "recover035")
    fw_cand = foldwise_from_readout(readout_dir, "longpatience")
    fw_parts = [df for df in [fw_locked, fw_recover035, fw_cand] if df is not None]
    if fw_parts:
        fw_all = pd.concat(fw_parts, ignore_index=True)
        write_table(outdir, "foldwise_auc_comparison", fw_all, max_rows=30)

    cl = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "dry_run": False,
        "training_launched": False,
        "tensor_modified": False,
        "original_metadata_modified": False,
        "ledger_modified": False,
        "locked_model_outputs_modified": False,
        "recover035_outputs_modified": False,
    }
    (outdir / "command_log.json").write_text(json.dumps(cl, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nSchedule-phase audit complete. Output: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
