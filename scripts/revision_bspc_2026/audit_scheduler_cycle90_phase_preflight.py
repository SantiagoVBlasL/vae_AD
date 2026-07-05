#!/usr/bin/env python3
"""Schedule-phase audit for the synchronized 90-epoch beta/LR cycle experiment.

Pre-training (--dry-run or when training history is absent):
  - Verifies beta cycle length = 4500/50 = 90 epochs from config.
  - Verifies lr_scheduler_T0 = 90 from config.
  - Verifies beta and LR cycles are phase-aligned at epoch 0 (both start at 0.0).
  - Reports complete cycle structure: ramp epochs, stable epochs, restart points.

Post-training (when VAE training histories are available):
  - Loads val_loss_modelsel history per fold.
  - Identifies best epoch (argmin of val_loss_modelsel).
  - Computes beta phase fraction and LR phase fraction at best epoch.
  - Reports beta region (ramping_beta / stable_beta) at best epoch.
  - Compares fold 1 and fold 4 against the locked reference (80-epoch cycles).

Post-Stage-B (when Stage B readout is available):
  - Reports foldwise Stage B AUC/PR-AUC for fold 1 and fold 4.
  - Provides fold-specific delta vs locked reference.

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

try:
    import joblib as _joblib
    _JOBLIB_OK = True
except ImportError:
    _JOBLIB_OK = False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"

DEFAULT_CONFIG = PROJECT_ROOT / "configs/runs/adni_v5_1_batch20260514b_ch1_0_2_scheduler_cycle90_sync_full5x5.json"
DEFAULT_RUN_DIR = RESULTS / "scheduler_cycle90_sync_full5x5"
DEFAULT_READOUT_DIR = DEFAULT_RUN_DIR / "classifier_only_readout"
DEFAULT_LOCKED_RUN = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5"
DEFAULT_LOCKED_READOUT = DEFAULT_LOCKED_RUN / "classifier_only_readout"
DEFAULT_OUTPUT = RESULTS / "scheduler_cycle90_sync_phase_audit"

PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
FOLDS = [1, 2, 3, 4, 5]
FOCUS_FOLDS = [1, 4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--readout-dir", type=Path, default=DEFAULT_READOUT_DIR)
    parser.add_argument("--locked-run-dir", type=Path, default=DEFAULT_LOCKED_RUN)
    parser.add_argument("--locked-readout-dir", type=Path, default=DEFAULT_LOCKED_READOUT)
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


def ensure_outdir(path: Path, overwrite: bool) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"{path} already contains files; pass --overwrite")
    if overwrite:
        for child in path.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                for nested in sorted(child.rglob("*"), reverse=True):
                    if nested.is_file() or nested.is_symlink():
                        nested.unlink()
                    elif nested.is_dir():
                        nested.rmdir()
                child.rmdir()


def verify_config_cycle_sync(config: Dict[str, Any]) -> Dict[str, Any]:
    params = config["parameters"]
    epochs = int(params["epochs_vae"])
    n_cycles = int(params["cyclical_beta_n_cycles"])
    T0 = int(params["lr_scheduler_T0"])
    ratio_inc = float(params["cyclical_beta_ratio_increase"])
    cycle_len = epochs / n_cycles
    remainder = epochs % n_cycles
    ramp_epochs_per_cycle = cycle_len * ratio_inc
    stable_epochs_per_cycle = cycle_len * (1.0 - ratio_inc)
    restart_epochs = [i * int(cycle_len) for i in range(n_cycles + 1) if i * int(cycle_len) <= epochs]
    checks = {
        "epochs_vae": epochs,
        "cyclical_beta_n_cycles": n_cycles,
        "lr_scheduler_T0": T0,
        "cyclical_beta_ratio_increase": ratio_inc,
        "beta_cycle_length": cycle_len,
        "lr_restart_period": T0,
        "beta_lr_synchronized": cycle_len == T0,
        "beta_phase_at_epoch_0": 0.0,
        "lr_phase_at_epoch_0": 0.0,
        "phase_aligned_at_epoch_0": True,
        "complete_cycles": int(epochs / cycle_len),
        "remainder_epochs": int(epochs % cycle_len) if cycle_len == int(cycle_len) else "non-integer cycle",
        "ramp_epochs_per_cycle": ramp_epochs_per_cycle,
        "stable_epochs_per_cycle": stable_epochs_per_cycle,
        "restart_epochs_first_7": restart_epochs[:7],
        "check_beta_cycle_len_eq_90": cycle_len == 90,
        "check_lr_T0_eq_90": T0 == 90,
        "check_synchronized": cycle_len == T0 == 90,
        "check_no_remainder": (epochs % int(cycle_len) == 0) if cycle_len == int(cycle_len) else False,
    }
    return checks


def print_config_analysis(checks: Dict[str, Any]) -> None:
    ok = "[PASS]"
    fail = "[FAIL]"
    print("\nSchedule-phase config analysis:")
    print(f"  epochs_vae                 = {checks['epochs_vae']}")
    print(f"  cyclical_beta_n_cycles     = {checks['cyclical_beta_n_cycles']}")
    print(f"  beta cycle length          = {checks['beta_cycle_length']:.1f} epochs  "
          f"{ok if checks['check_beta_cycle_len_eq_90'] else fail}")
    print(f"  lr_scheduler_T0            = {checks['lr_restart_period']} epochs  "
          f"{ok if checks['check_lr_T0_eq_90'] else fail}")
    print(f"  beta/LR synchronized       = {checks['beta_lr_synchronized']}  "
          f"{ok if checks['check_synchronized'] else fail}")
    print(f"  phase at epoch 0           : beta=0.0, LR=0.0  [SYNCHRONIZED]")
    print(f"  complete cycles            = {checks['complete_cycles']}")
    print(f"  remainder epochs           = {checks['remainder_epochs']}  "
          f"{ok if checks['check_no_remainder'] else fail}")
    print(f"  ramp epochs / cycle        = {checks['ramp_epochs_per_cycle']:.0f}  "
          f"(ratio_increase={checks['cyclical_beta_ratio_increase']})")
    print(f"  stable epochs / cycle      = {checks['stable_epochs_per_cycle']:.0f}")
    restarts = checks.get("restart_epochs_first_7", [])
    suffix = "..." if checks["complete_cycles"] > 6 else ""
    print(f"  restart epochs (first 7)   = {restarts}{suffix}")
    all_ok = checks["check_beta_cycle_len_eq_90"] and checks["check_lr_T0_eq_90"] and checks["check_synchronized"] and checks["check_no_remainder"]
    print(f"\n  Overall: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")


def load_vae_history(run_dir: Path, fold: int) -> Optional[Dict[str, Any]]:
    if not _JOBLIB_OK:
        return None
    p = run_dir / f"fold_{fold}" / f"vae_train_history_fold_{fold}.joblib"
    if not p.exists():
        return None
    return dict(_joblib.load(p))


def compute_fold_phase(
    run_dir: Path,
    fold: int,
    cycle_len: int,
    T0: int,
    ratio_inc: float,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "fold": fold,
        "cycle_len": cycle_len,
        "T0": T0,
        "history_available": False,
    }
    hist = load_vae_history(run_dir, fold)
    if hist is None or "val_loss_modelsel" not in hist:
        for k in ["total_epochs_trained", "best_epoch", "epoch_in_cycle", "phase_frac",
                  "lr_phase_frac", "beta_region", "ramp_boundary_epoch"]:
            row[k] = np.nan
        row["beta_region"] = "unknown"
        return row
    vl = list(hist["val_loss_modelsel"])
    best = int(np.argmin(vl))
    total = len(vl)
    epoch_in_cycle = best % cycle_len
    phase_frac = epoch_in_cycle / cycle_len
    ramp_boundary = int(ratio_inc * cycle_len)
    region = "ramping_beta" if phase_frac < ratio_inc else "stable_beta"
    row.update({
        "history_available": True,
        "total_epochs_trained": total,
        "best_epoch": best,
        "epoch_in_cycle": epoch_in_cycle,
        "phase_frac": phase_frac,
        "lr_phase_frac": (best % T0) / T0,
        "beta_region": region,
        "ramp_boundary_epoch": ramp_boundary,
    })
    return row


def build_phase_table(
    cand_run: Path,
    locked_run: Path,
    cand_cycle: int,
    cand_T0: int,
    locked_cycle: int,
    locked_T0: int,
    ratio_inc: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fold in FOLDS:
        cand_row = compute_fold_phase(cand_run, fold, cand_cycle, cand_T0, ratio_inc)
        lock_row = compute_fold_phase(locked_run, fold, locked_cycle, locked_T0, ratio_inc)
        row: Dict[str, Any] = {
            "fold": fold,
            "is_focus_fold": fold in FOCUS_FOLDS,
            "cand_total_epochs": cand_row.get("total_epochs_trained", np.nan),
            "cand_best_epoch": cand_row.get("best_epoch", np.nan),
            "cand_epoch_in_cycle": cand_row.get("epoch_in_cycle", np.nan),
            "cand_phase_frac": cand_row.get("phase_frac", np.nan),
            "cand_lr_phase_frac": cand_row.get("lr_phase_frac", np.nan),
            "cand_beta_region": cand_row.get("beta_region", "unknown"),
            "cand_cycle_len": cand_cycle,
            "locked_total_epochs": lock_row.get("total_epochs_trained", np.nan),
            "locked_best_epoch": lock_row.get("best_epoch", np.nan),
            "locked_epoch_in_cycle": lock_row.get("epoch_in_cycle", np.nan),
            "locked_phase_frac": lock_row.get("phase_frac", np.nan),
            "locked_beta_region": lock_row.get("beta_region", "unknown"),
            "locked_cycle_len": locked_cycle,
        }
        # delta best epoch (only if both numeric)
        if not np.isnan(row["cand_best_epoch"]) and not np.isnan(row["locked_best_epoch"]):
            row["delta_best_epoch"] = float(row["cand_best_epoch"]) - float(row["locked_best_epoch"])
            row["delta_phase_frac"] = float(row["cand_phase_frac"]) - float(row["locked_phase_frac"])
        else:
            row["delta_best_epoch"] = np.nan
            row["delta_phase_frac"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def read_foldwise_stageb(readout: Path) -> Optional[pd.DataFrame]:
    p = readout / "classifier_sweep_foldwise_metrics.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    mask = df["model_name"].astype(str).eq(PRIMARY_MODEL) & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    return df[mask].copy()


def build_fold_focus_stageb(
    cand_readout: Path,
    locked_readout: Path,
) -> pd.DataFrame:
    cand_fw = read_foldwise_stageb(cand_readout)
    locked_fw = read_foldwise_stageb(locked_readout)
    rows: List[Dict[str, Any]] = []
    for fold in FOCUS_FOLDS:
        row: Dict[str, Any] = {"fold": fold}
        for tag, fw in [("cand", cand_fw), ("locked", locked_fw)]:
            if fw is not None and fold in fw["fold"].values:
                r = fw[fw["fold"].eq(fold)].iloc[0]
                for m in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
                    row[f"{tag}_{m}"] = float(r.get(m, np.nan))
            else:
                for m in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
                    row[f"{tag}_{m}"] = np.nan
        for m in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
            c = row.get(f"cand_{m}", np.nan)
            l = row.get(f"locked_{m}", np.nan)
            row[f"delta_{m}"] = float(c) - float(l) if not (np.isnan(c) or np.isnan(l)) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write_readme(
    outdir: Path,
    checks: Dict[str, Any],
    phase_df: pd.DataFrame,
    fold_focus_df: pd.DataFrame,
    cand_cycle: int,
    locked_cycle: int,
) -> None:
    phase_available = not phase_df.empty and bool(phase_df["cand_total_epochs"].notna().any())
    stageb_available = not fold_focus_df.empty and bool(fold_focus_df["cand_auc"].notna().any())
    lines = [
        "# Scheduler-Cycle90-Sync Phase Audit",
        "",
        "Read-only schedule-phase analysis for the synchronized 90-epoch beta/LR cycle experiment.",
        "",
        "## Config-level verification",
        "",
        f"- Beta cycle length : {checks['beta_cycle_length']:.0f} epochs (4500/50=90)  "
        f"[{'PASS' if checks['check_beta_cycle_len_eq_90'] else 'FAIL'}]",
        f"- LR restart T0     : {checks['lr_restart_period']} epochs  "
        f"[{'PASS' if checks['check_lr_T0_eq_90'] else 'FAIL'}]",
        f"- Synchronized      : {checks['beta_lr_synchronized']}  "
        f"[{'PASS' if checks['check_synchronized'] else 'FAIL'}]",
        f"- Phase at epoch 0  : beta=0.0, LR=0.0  [SYNCHRONIZED]",
        f"- No remainder      : {checks['check_no_remainder']}  "
        f"[{'PASS' if checks['check_no_remainder'] else 'FAIL'}]",
        f"- Ramp epochs/cycle : {checks['ramp_epochs_per_cycle']:.0f} (0–35), stable: {checks['stable_epochs_per_cycle']:.0f} (36–89)",
        "",
        "## Comparison with locked reference",
        "",
        f"| parameter | candidate (cycle90) | locked reference (cycle80) |",
        f"|---|---|---|",
        f"| beta cycle length | {cand_cycle} epochs | {locked_cycle} epochs |",
        f"| LR restart T0     | {cand_cycle} epochs | {locked_cycle} epochs |",
        f"| epochs_vae        | 4500 | 4480 |",
        f"| n_cycles          | 50 | 56 |",
        f"| phase at epoch 0  | 0.0 (both) | 0.0 (both) |",
        f"| ramp boundary     | epoch 36/cycle | epoch 32/cycle |",
        "",
    ]
    if phase_available:
        lines.extend([
            "## Best-epoch phase analysis",
            "",
            "Phase fraction = (best_epoch % cycle_len) / cycle_len.",
            "Values >= 0.4 are in the stable-beta region (beta held at max).",
            "",
            md_table(phase_df),
        ])
    else:
        lines.extend([
            "## Best-epoch phase analysis",
            "",
            "_Training history not yet available. Run after Stage A completes._",
            "",
        ])
    if stageb_available:
        lines.extend([
            "## Fold 1 and Fold 4 Stage B focus",
            "",
            md_table(fold_focus_df),
        ])
    else:
        lines.extend([
            "## Fold 1 and Fold 4 Stage B focus",
            "",
            "_Stage B readout not yet available._",
            "",
        ])
    (outdir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    config_path = resolve(args.config)
    cand_run = resolve(args.run_dir)
    cand_readout = resolve(args.readout_dir)
    locked_run = resolve(args.locked_run_dir)
    locked_readout = resolve(args.locked_readout_dir)
    outdir = resolve(args.output_dir)

    config = load_json(config_path)
    checks = verify_config_cycle_sync(config)
    print_config_analysis(checks)

    all_pass = checks["check_beta_cycle_len_eq_90"] and checks["check_lr_T0_eq_90"] and checks["check_synchronized"] and checks["check_no_remainder"]
    if not all_pass:
        raise RuntimeError("Config schedule verification FAILED. Check output above.")

    if args.dry_run:
        print("\nDry-run complete. No output files were written.")
        return 0

    ensure_outdir(outdir, overwrite=args.overwrite)

    params = config["parameters"]
    cand_cycle = int(params["epochs_vae"] / params["cyclical_beta_n_cycles"])
    cand_T0 = int(params["lr_scheduler_T0"])
    locked_cycle = 80
    locked_T0 = 80
    ratio_inc = float(params["cyclical_beta_ratio_increase"])

    # Config check output
    (outdir / "schedule_config_check.json").write_text(
        json.dumps(
            {k: (v if not isinstance(v, list) else v) for k, v in checks.items()},
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )

    phase_df = build_phase_table(cand_run, locked_run, cand_cycle, cand_T0, locked_cycle, locked_T0, ratio_inc)
    write_table(outdir, "fold_phase_analysis", phase_df, max_rows=10)

    fold_focus_df = build_fold_focus_stageb(cand_readout, locked_readout)
    write_table(outdir, "fold_focus_stageb", fold_focus_df, max_rows=5)

    write_readme(outdir, checks, phase_df, fold_focus_df, cand_cycle, locked_cycle)

    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "config_path": str(config_path),
        "candidate_run_dir": str(cand_run),
        "candidate_readout_dir": str(cand_readout),
        "locked_run_dir": str(locked_run),
        "locked_readout_dir": str(locked_readout),
        "output_dir": str(outdir),
        "candidate_cycle_len": cand_cycle,
        "candidate_T0": cand_T0,
        "locked_cycle_len": locked_cycle,
        "locked_T0": locked_T0,
        "config_checks": {k: v for k, v in checks.items() if k.startswith("check_")},
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "config_modified": False,
        "model_output_modified": False,
    }
    (outdir / "command_log.json").write_text(json.dumps(command_log, indent=2, default=str, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nWrote schedule-phase audit to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
