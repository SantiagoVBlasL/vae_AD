#!/usr/bin/env python3
"""Read-only training-dynamics and classifier-boundary audit for v5.1c.

This script reads completed VAE/classifier outputs and writes compact audit
tables. It does not modify run folders, tensors, metadata, ledgers, configs, or
model outputs.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


DEFAULT_SOURCE_RUN = Path(
    "results/revision_bspc_2026/"
    "adni_v5_1c_recover035_ch1_0_2_horizon4480_cycles56_full_5x5"
)
DEFAULT_REFERENCE_RUN = Path(
    "results/revision_bspc_2026/"
    "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5"
)
DEFAULT_ULTRA_REG_DIR = Path(
    "results/revision_bspc_2026/"
    "adni_v5_1_batch20260514b_ultra_regularized_logreg_readout_audit"
)
DEFAULT_OUT_DIR = Path(
    "results/revision_bspc_2026/"
    "adni_v5_1c_training_dynamics_classifier_boundary_audit"
)

PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"

STAGE_A_BOUNDS = {
    "logreg_C": (1e-5, 1.0),
    "svm_C": (1e-1, 1e4),
    "svm_gamma": (1e-7, 1e-1),
}
STAGE_B_LOGREG_L2_GRID = [0.001, 0.01, 0.1, 1.0]


def _resolve(path: Path) -> Path:
    return path.expanduser().resolve()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _find_one(base: Path, pattern: str) -> Path | None:
    matches = sorted(base.glob(pattern))
    return matches[0] if matches else None


def _to_md(df: pd.DataFrame, path: Path, title: str | None = None) -> None:
    with path.open("w", encoding="utf-8") as f:
        if title:
            f.write(f"# {title}\n\n")
        if df.empty:
            f.write("_No rows._\n")
            return
        try:
            f.write(df.to_markdown(index=False))
        except Exception:
            f.write(df.to_csv(index=False))
        f.write("\n")


def _as_float(x: Any) -> float | None:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def _parse_params(raw: Any) -> dict[str, Any]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    if isinstance(raw, dict):
        return raw
    text = str(raw)
    for parser in (json.loads, ast.literal_eval):
        try:
            value = parser(text)
            if isinstance(value, dict):
                return value
        except Exception:
            continue
    return {}


def _log_edge_status(value: float | None, lower: float, upper: float) -> str:
    if value is None or value <= 0:
        return "unknown"
    logv = math.log10(value)
    loglo = math.log10(lower)
    loghi = math.log10(upper)
    pos = (logv - loglo) / (loghi - loglo)
    if pos <= 0.05:
        return "near_lower_bound"
    if pos >= 0.95:
        return "near_upper_bound"
    return "interior"


def _grid_edge_status(value: float | None, grid: list[float]) -> str:
    if value is None:
        return "unknown"
    if np.isclose(value, min(grid)):
        return "at_lower_bound"
    if np.isclose(value, max(grid)):
        return "at_upper_bound"
    return "interior"


def _phase_label(epoch_1based: int, cycle_len: int, ramp_ratio: float) -> tuple[int, int, str]:
    phase_epoch0 = (epoch_1based - 1) % cycle_len
    cycle_idx = (epoch_1based - 1) // cycle_len + 1
    ramp_len = int(round(cycle_len * ramp_ratio))
    phase = "beta_ramp" if phase_epoch0 < ramp_len else "beta_plateau"
    return cycle_idx, phase_epoch0 + 1, phase


def _cosine_lr(epoch_1based: int, lr_max: float, eta_min: float, t0: int) -> float:
    # Approximation for CosineAnnealingWarmRestarts with constant T_0/T_mult=1.
    phase_epoch0 = (epoch_1based - 1) % t0
    return eta_min + 0.5 * (lr_max - eta_min) * (1.0 + math.cos(math.pi * phase_epoch0 / t0))


def _lr_phase(epoch_1based: int, t0: int) -> str:
    phase_epoch0 = (epoch_1based - 1) % t0
    if phase_epoch0 <= 2:
        return "near_lr_restart"
    if phase_epoch0 >= t0 - 3:
        return "near_lr_trough"
    if phase_epoch0 < t0 / 2:
        return "lr_descending_early"
    return "lr_descending_late"


def _validation_slope(values: list[float], window: int = 100) -> float | None:
    clean = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(clean) < 5:
        return None
    tail = clean[-min(window, len(clean)) :]
    x = np.arange(len(tail), dtype=float)
    return float(np.polyfit(x, tail, deg=1)[0])


def extract_training_dynamics(run_dir: Path, label: str) -> pd.DataFrame:
    run_config = _read_json(run_dir / "run_config.json")
    args = run_config.get("args", {})
    max_epoch = int(args.get("epochs_vae", 4480))
    cycles = int(args.get("cyclical_beta_n_cycles", 56))
    cycle_len = int(round(max_epoch / cycles)) if cycles else int(args.get("lr_scheduler_T0", 80))
    t0 = int(args.get("lr_scheduler_T0", cycle_len))
    ramp_ratio = float(args.get("cyclical_beta_ratio_increase", 0.4))
    patience = int(args.get("early_stopping_patience_vae", 320))
    lr_max = float(args.get("lr_vae", 1e-4))
    eta_min = float(args.get("lr_scheduler_eta_min", 5e-7))

    rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        hist_path = run_dir / f"fold_{fold}" / f"vae_train_history_fold_{fold}.joblib"
        if not hist_path.exists():
            rows.append({"run_label": label, "fold": fold, "history_missing": True})
            continue
        hist = joblib.load(hist_path)
        modelsel = list(hist.get("val_loss_modelsel") or hist.get("val_loss") or [])
        final_epoch = len(modelsel)
        if final_epoch == 0:
            rows.append({"run_label": label, "fold": fold, "history_missing": True})
            continue
        best_idx = int(np.nanargmin(np.asarray(modelsel, dtype=float)))
        best_epoch = best_idx + 1
        epochs_after_best = final_epoch - best_epoch
        reached_max = final_epoch == max_epoch
        early_stopped = final_epoch < max_epoch
        slope = _validation_slope(modelsel)
        best_cycle, best_cycle_epoch, beta_phase = _phase_label(best_epoch, cycle_len, ramp_ratio)
        best_beta = _as_float(hist.get("beta", [None] * final_epoch)[best_idx])
        lr_at_best = _cosine_lr(best_epoch, lr_max, eta_min, t0)

        close_to_final = epochs_after_best <= min(patience, 80)
        still_improving = bool(slope is not None and slope < -1e-4)
        right_censored = bool(reached_max and (close_to_final or still_improving))

        def hist_value(key: str, idx: int) -> float | None:
            values = hist.get(key)
            if not values or idx >= len(values):
                return None
            return _as_float(values[idx])

        rows.append(
            {
                "run_label": label,
                "fold": fold,
                "history_missing": False,
                "best_epoch": best_epoch,
                "final_epoch": final_epoch,
                "early_stop_epoch": final_epoch if early_stopped else np.nan,
                "reached_max_epoch_4480": reached_max,
                "early_stopped_before_max": early_stopped,
                "epochs_after_best": epochs_after_best,
                "best_valL_beta_max": hist_value("val_loss_modelsel", best_idx),
                "last_valL_beta_max": _as_float(modelsel[-1]),
                "last100_valL_beta_max_slope": slope,
                "validation_still_improving_last100": still_improving,
                "best_epoch_close_to_final": close_to_final,
                "right_censored_by_max_epoch": right_censored,
                "train_recon_at_best": hist_value("train_recon", best_idx),
                "val_recon_at_best": hist_value("val_recon", best_idx),
                "train_kld_at_best": hist_value("train_kld", best_idx),
                "val_kld_at_best": hist_value("val_kld", best_idx),
                "val_kld_over_recon_at_best": hist_value("val_kld_over_recon", best_idx),
                "val_beta_kld_over_recon_at_best": hist_value("val_beta_kld_over_recon", best_idx),
                "beta_at_best_epoch": best_beta,
                "beta_cycle_index_at_best": best_cycle,
                "beta_cycle_epoch_at_best": best_cycle_epoch,
                "beta_phase_at_best_epoch": beta_phase,
                "lr_at_best_epoch_estimated": lr_at_best,
                "lr_phase_at_best_epoch": _lr_phase(best_epoch, t0),
                "max_epoch": max_epoch,
                "n_cycles": cycles,
                "cycle_len": cycle_len,
                "lr_scheduler_T0": t0,
            }
        )
    return pd.DataFrame(rows)


def extract_stage_metrics(run_dir: Path, label: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metrics_path = _find_one(run_dir, "all_folds_metrics*.csv")
    if metrics_path:
        df = pd.read_csv(metrics_path)
        for _, r in df.iterrows():
            rows.append(
                {
                    "run_label": label,
                    "stage": "Stage A",
                    "fold": int(r["fold"]),
                    "model_name": str(r["actual_classifier_type"]),
                    "threshold_strategy": "original_fixed_0p5",
                    "auc": r.get("auc"),
                    "pr_auc": r.get("pr_auc"),
                    "balanced_accuracy": r.get("balanced_accuracy"),
                    "sensitivity": r.get("sensitivity"),
                    "specificity": r.get("specificity"),
                    "f1": r.get("f1_score"),
                }
            )

    stageb_path = run_dir / "classifier_only_readout" / "classifier_sweep_foldwise_metrics.csv"
    if stageb_path.exists():
        df = pd.read_csv(stageb_path)
        primary = df[
            (df["model_name"] == "logreg_l2")
            & (df["threshold_strategy"] == PRIMARY_THRESHOLD)
        ].copy()
        for _, r in primary.iterrows():
            rows.append(
                {
                    "run_label": label,
                    "stage": "Stage B",
                    "fold": int(r["fold"]),
                    "model_name": "logreg_l2",
                    "threshold_strategy": PRIMARY_THRESHOLD,
                    "auc": r.get("auc"),
                    "pr_auc": r.get("pr_auc"),
                    "balanced_accuracy": r.get("balanced_accuracy"),
                    "sensitivity": r.get("sensitivity"),
                    "specificity": r.get("specificity"),
                    "f1": r.get("f1"),
                }
            )
    return pd.DataFrame(rows)


def _extract_stage_a_params(run_dir: Path, fold: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for clf in ("logreg", "svm"):
        path = run_dir / f"fold_{fold}" / f"optuna_best_trial_{clf}_fold_{fold}.json"
        if not path.exists():
            continue
        data = _read_json(path)
        params = data.get("best_params", {})
        if clf == "logreg":
            c = _as_float(params.get("model__C"))
            out.update(
                {
                    "stagea_logreg_C": c,
                    "stagea_logreg_C_boundary": _log_edge_status(
                        c, *STAGE_A_BOUNDS["logreg_C"]
                    ),
                    "stagea_logreg_inner_auc": data.get("best_value"),
                }
            )
        if clf == "svm":
            c = _as_float(params.get("model__C"))
            gamma = _as_float(params.get("model__gamma"))
            out.update(
                {
                    "stagea_svm_C": c,
                    "stagea_svm_gamma": gamma,
                    "stagea_svm_C_boundary": _log_edge_status(
                        c, *STAGE_A_BOUNDS["svm_C"]
                    ),
                    "stagea_svm_gamma_boundary": _log_edge_status(
                        gamma, *STAGE_A_BOUNDS["svm_gamma"]
                    ),
                    "stagea_svm_inner_auc": data.get("best_value"),
                }
            )
    return out


def extract_classifier_boundaries(run_dir: Path, label: str) -> pd.DataFrame:
    status_path = run_dir / "classifier_only_readout" / "classifier_sweep_model_status.csv"
    status_df = _safe_read_csv(status_path)
    rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        row: dict[str, Any] = {"run_label": label, "fold": fold}
        row.update(_extract_stage_a_params(run_dir, fold))
        if not status_df.empty:
            match = status_df[
                (status_df["fold"] == fold) & (status_df["model_name"] == "logreg_l2")
            ]
            if not match.empty:
                params = _parse_params(match.iloc[0].get("best_params"))
                c = _as_float(params.get("model__C"))
                row.update(
                    {
                        "stageb_logreg_l2_C": c,
                        "stageb_logreg_l2_C_boundary": _grid_edge_status(
                            c, STAGE_B_LOGREG_L2_GRID
                        ),
                        "stageb_best_inner_auc": match.iloc[0].get("best_inner_auc"),
                    }
                )
        rows.append(row)
    return pd.DataFrame(rows)


def load_ultra_regularized_context(ultra_dir: Path) -> tuple[pd.DataFrame, str]:
    pieces: list[str] = []
    selected_path = ultra_dir / "selected_C_by_fold.csv"
    primary_path = ultra_dir / "primary_comparison.csv"
    selected_df = _safe_read_csv(selected_path)
    primary_df = _safe_read_csv(primary_path)
    if not selected_df.empty:
        pieces.append("Ultra-regularized selected C by fold was available.")
    if not primary_df.empty:
        pieces.append("Ultra-regularized primary comparison was available.")
    if selected_df.empty and primary_df.empty:
        pieces.append("Ultra-regularized audit tables were not found.")
    context = "\n".join(pieces)
    if not primary_df.empty:
        primary_df = primary_df.copy()
        if "source" not in primary_df.columns:
            primary_df.insert(0, "source", "previous_ultra_regularized_audit")
        else:
            primary_df["source"] = primary_df["source"].fillna(
                "previous_ultra_regularized_audit"
            )
    return primary_df, context


def make_readme(
    out_dir: Path,
    training: pd.DataFrame,
    metrics: pd.DataFrame,
    boundaries: pd.DataFrame,
    ultra_primary: pd.DataFrame,
) -> str:
    v5c_train = training[training["run_label"] == "v5.1c"]
    v5c_stageb = metrics[
        (metrics["run_label"] == "v5.1c")
        & (metrics["stage"] == "Stage B")
        & (metrics["model_name"] == "logreg_l2")
    ]
    right = v5c_train[v5c_train["right_censored_by_max_epoch"] == True]
    lower_bound_stageb = boundaries[
        (boundaries["run_label"] == "v5.1c")
        & (boundaries["stageb_logreg_l2_C_boundary"] == "at_lower_bound")
    ]

    mean_auc = float(v5c_stageb["auc"].mean()) if not v5c_stageb.empty else float("nan")
    mean_pr = float(v5c_stageb["pr_auc"].mean()) if not v5c_stageb.empty else float("nan")

    fold1 = v5c_train[v5c_train["fold"] == 1]
    fold1_right = bool(
        not fold1.empty and bool(fold1.iloc[0].get("right_censored_by_max_epoch"))
    )
    any_right = not right.empty
    many_right = len(right) >= 2
    full_more_epochs = (
        "not justified"
        if not many_right
        else "diagnostic-only; multiple folds are right-censored, but confirm fold-specific behavior first"
    )
    fold1_extension = (
        "justified as a fold-only diagnostic"
        if fold1_right
        else "not justified from this audit"
    )

    ultra_summary = "No prior ultra-regularized table was readable."
    if not ultra_primary.empty:
        cols = [c for c in ["model_name", "threshold_strategy", "auc", "pr_auc", "balanced_accuracy", "f1"] if c in ultra_primary.columns]
        if cols:
            best = ultra_primary.sort_values("auc", ascending=False).head(3)
            ultra_summary = best[cols].to_string(index=False)

    grid_decision = (
        "A Stage B C-grid expansion is not recommended as a new optimization path. "
        "v5.1c still selects the lower Stage B C bound in several folds, but the prior "
        "ultra-regularized audit found that pushing below the original grid hurt test "
        "ranking. If needed, it should remain a read-only sensitivity analysis, not a "
        "promoted model-search branch."
        if not lower_bound_stageb.empty
        else "No Stage B lower-bound pressure was detected; no grid expansion is justified."
    )

    text = f"""# v5.1c Training-Dynamics And Classifier-Boundary Audit

Read-only audit of:

- Source: `{DEFAULT_SOURCE_RUN}`
- Reference: `{DEFAULT_REFERENCE_RUN}`

No training was launched. No tensor, metadata, ledger, config, or existing model-output files were modified.

## v5.1c Stage B Summary

Primary readout: `logreg_l2` with `{PRIMARY_THRESHOLD}`.

- Mean foldwise ROC-AUC: {mean_auc:.6f}
- Mean foldwise PR-AUC: {mean_pr:.6f}
- Right-censored folds by max epoch: {len(right)} / 5
- Stage B lower-bound C selections: {len(lower_bound_stageb)} / 5

## Horizon Decision

- Fold 1-only horizon5120/cycles64: **{fold1_extension}**.
- Full v5.1c rerun with more epochs: **{full_more_epochs}**.

The fold-only decision is based on whether Fold 1 reached `max_epoch=4480` without early stopping and had either a best epoch close to the final epoch or an improving final validation-loss slope. A full rerun is not recommended unless multiple folds show the same right-censoring pattern and the fold-only diagnostic is positive.

## Classifier Boundary Decision

{grid_decision}

Previous ultra-regularized context:

```text
{ultra_summary}
```

## Files

- `fold_training_dynamics.csv/.md`
- `fold_stagea_stageb_metrics.csv/.md`
- `classifier_hyperparameter_boundary.csv/.md`
- `fold1_horizon_extension_recommendation.md`
- `command_log.json`
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")
    return text


def write_fold1_recommendation(
    out_dir: Path,
    training: pd.DataFrame,
    metrics: pd.DataFrame,
    boundaries: pd.DataFrame,
) -> None:
    v5c_train = training[training["run_label"] == "v5.1c"].copy()
    v5c_stageb = metrics[
        (metrics["run_label"] == "v5.1c")
        & (metrics["stage"] == "Stage B")
        & (metrics["model_name"] == "logreg_l2")
    ].copy()
    fold1_train = v5c_train[v5c_train["fold"] == 1]
    fold1_metric = v5c_stageb[v5c_stageb["fold"] == 1]
    right_folds = v5c_train[v5c_train["right_censored_by_max_epoch"] == True]["fold"].tolist()

    if not fold1_train.empty and bool(fold1_train.iloc[0]["right_censored_by_max_epoch"]):
        fold1_decision = "Fold 1-only horizon5120/cycles64 is justified as a diagnostic."
    else:
        fold1_decision = "Fold 1-only horizon5120/cycles64 is not justified by the v5.1c training dynamics."

    if len(right_folds) >= 2:
        full_decision = (
            "Do not launch a full v5.1c longer-horizon rerun yet. Multiple folds show "
            "right-censoring, but a fold-specific diagnostic should precede any full run."
        )
    else:
        full_decision = (
            "A full v5.1c longer-horizon rerun is not justified from this audit."
        )

    lower_bound_count = int(
        (
            (boundaries["run_label"] == "v5.1c")
            & (boundaries["stageb_logreg_l2_C_boundary"] == "at_lower_bound")
        ).sum()
    )
    grid_decision = (
        "Classifier grid boundary expansion is not recommended for promotion. "
        f"Stage B selected the lower C bound in {lower_bound_count}/5 folds, but the prior "
        "ultra-regularized audit already showed that stronger L2 regularization reduced "
        "test AUC/PR-AUC."
        if lower_bound_count
        else "Classifier grid boundary expansion is not justified; no lower-bound pressure was detected."
    )

    fold1_auc = (
        float(fold1_metric.iloc[0]["auc"]) if not fold1_metric.empty else float("nan")
    )
    fold1_pr = (
        float(fold1_metric.iloc[0]["pr_auc"]) if not fold1_metric.empty else float("nan")
    )

    text = f"""# Fold 1 Horizon / Classifier Boundary Recommendation

## Fold 1 Horizon5120/Cycles64

Decision: **{fold1_decision}**

Fold 1 Stage B primary readout:

- ROC-AUC: {fold1_auc:.6f}
- PR-AUC: {fold1_pr:.6f}

Right-censored v5.1c folds: `{right_folds}`.

Criteria used:

- `final_epoch == 4480`
- no early stopping
- best epoch close to final epoch or final validation slope still improving

## Classifier Grid Boundary

Decision: **{grid_decision}**

## Full v5.1c Longer-Horizon Rerun

Decision: **{full_decision}**

This is a read-only audit. No new training was launched.
"""
    (out_dir / "fold1_horizon_extension_recommendation.md").write_text(
        text, encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--reference-run", type=Path, default=DEFAULT_REFERENCE_RUN)
    parser.add_argument("--ultra-regularized-dir", type=Path, default=DEFAULT_ULTRA_REG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    source_run = _resolve(args.source_run)
    reference_run = _resolve(args.reference_run)
    ultra_dir = _resolve(args.ultra_regularized_dir)
    out_dir = _resolve(args.output_dir)

    for required in [source_run, reference_run]:
        if not required.exists():
            raise FileNotFoundError(required)

    out_dir.mkdir(parents=True, exist_ok=True)

    source_training = extract_training_dynamics(source_run, "v5.1c")
    ref_training = extract_training_dynamics(reference_run, "v5.1b_horizon4480")
    training = pd.concat([source_training, ref_training], ignore_index=True)

    source_metrics = extract_stage_metrics(source_run, "v5.1c")
    ref_metrics = extract_stage_metrics(reference_run, "v5.1b_horizon4480")
    metrics = pd.concat([source_metrics, ref_metrics], ignore_index=True)

    source_boundaries = extract_classifier_boundaries(source_run, "v5.1c")
    ref_boundaries = extract_classifier_boundaries(reference_run, "v5.1b_horizon4480")
    boundaries = pd.concat([source_boundaries, ref_boundaries], ignore_index=True)

    ultra_primary, ultra_context = load_ultra_regularized_context(ultra_dir)
    if not ultra_primary.empty:
        ultra_primary.to_csv(out_dir / "previous_ultra_regularized_primary_comparison.csv", index=False)
        _to_md(
            ultra_primary,
            out_dir / "previous_ultra_regularized_primary_comparison.md",
            "Previous Ultra-Regularized Primary Comparison",
        )

    training.to_csv(out_dir / "fold_training_dynamics.csv", index=False)
    _to_md(training, out_dir / "fold_training_dynamics.md", "Fold Training Dynamics")

    metrics.to_csv(out_dir / "fold_stagea_stageb_metrics.csv", index=False)
    _to_md(
        metrics,
        out_dir / "fold_stagea_stageb_metrics.md",
        "Fold Stage A / Stage B Metrics",
    )

    boundaries.to_csv(out_dir / "classifier_hyperparameter_boundary.csv", index=False)
    _to_md(
        boundaries,
        out_dir / "classifier_hyperparameter_boundary.md",
        "Classifier Hyperparameter Boundary Audit",
    )

    write_fold1_recommendation(out_dir, training, metrics, boundaries)
    readme_text = make_readme(out_dir, training, metrics, boundaries, ultra_primary)

    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "source_run": str(source_run),
        "reference_run": str(reference_run),
        "ultra_regularized_dir": str(ultra_dir),
        "output_dir": str(out_dir),
        "read_only": True,
        "no_training": True,
        "no_tensor_metadata_ledger_config_or_model_output_modification": True,
        "primary_threshold_strategy": PRIMARY_THRESHOLD,
        "stage_a_bounds": STAGE_A_BOUNDS,
        "stage_b_logreg_l2_grid": STAGE_B_LOGREG_L2_GRID,
        "ultra_regularized_context": ultra_context,
        "outputs": [
            "README.md",
            "fold_training_dynamics.csv",
            "fold_training_dynamics.md",
            "fold_stagea_stageb_metrics.csv",
            "fold_stagea_stageb_metrics.md",
            "classifier_hyperparameter_boundary.csv",
            "classifier_hyperparameter_boundary.md",
            "fold1_horizon_extension_recommendation.md",
            "command_log.json",
        ],
    }
    (out_dir / "command_log.json").write_text(
        json.dumps(command_log, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(readme_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
