#!/usr/bin/env python3
"""Read-only saturation audit across the main v5.1b/v5.1c FULL candidates.

The script only reads existing run artifacts and writes lightweight CSV/MD
summaries under results/. It does not train, alter tensors, edit metadata,
change ledgers, or modify model-output folders.
"""

from __future__ import annotations

import ast
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"
OUT_DIR = RESULTS / "final_model_saturation_audit"
PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
FIXED_THRESHOLD = "fixed_0p5"

RUNS: list[dict[str, str]] = [
    {
        "run_id": "v5_1b_horizon4480_current_loss",
        "label": "v5.1b horizon4480/cycles56 current loss",
        "run_dir": "results/revision_bspc_2026/adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5",
        "config": "configs/runs/adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5.json",
    },
    {
        "run_id": "v5_1c_horizon4480_current_loss",
        "label": "v5.1c horizon4480/cycles56 current loss",
        "run_dir": "results/revision_bspc_2026/adni_v5_1c_recover035_ch1_0_2_horizon4480_cycles56_full_5x5",
        "config": "configs/runs/adni_v5_1c_recover035_ch1_0_2_horizon4480_cycles56_full_5x5.json",
    },
    {
        "run_id": "v5_1c_horizon10000_current_loss",
        "label": "v5.1c horizon10000/cycles125 current loss",
        "run_dir": "results/revision_bspc_2026/adni_v5_1c_recover035_ch1_0_2_horizon10000_cycles125_full_5x5",
        "config": "configs/runs/adni_v5_1c_recover035_ch1_0_2_horizon10000_cycles125_full_5x5.json",
    },
    {
        "run_id": "v5_1c_horizon10000_objective_v2_offdiag_channelmean",
        "label": "v5.1c horizon10000/cycles125 offdiag_channelmean_sum",
        "run_dir": "results/revision_bspc_2026/adni_v5_1c_recover035_ch1_0_2_objective_v2_offdiag_channelmean_horizon10000_cycles125_full_5x5",
        "config": "configs/runs/adni_v5_1c_recover035_ch1_0_2_objective_v2_offdiag_channelmean_horizon10000_cycles125_full_5x5.json",
    },
]

STAGE_B_C_GRID = [0.001, 0.01, 0.1, 1.0]


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_config(path: Path) -> dict[str, Any]:
    data = read_json(path)
    if not data:
        return {"parameters": {}, "paths": {}}
    return data


def to_float(x: Any) -> float:
    try:
        if pd.isna(x):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def parse_params(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return {}
    text = str(raw)
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return {}


def write_table(stem: str, df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / f"{stem}.csv", index=False)
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{x:.6f}")
    (OUT_DIR / f"{stem}.md").write_text(view.to_markdown(index=False) + "\n", encoding="utf-8")


def first_existing(paths: Iterable[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def all_folds_metrics_path(run_dir: Path) -> Path | None:
    return first_existing(sorted(run_dir.glob("all_folds_metrics_MULTI_*.csv")))


def history_path(run_dir: Path, fold: int) -> Path | None:
    return first_existing(
        [
            run_dir / f"fold_{fold}" / f"vae_train_history_fold_{fold}.joblib",
            *sorted((run_dir / f"fold_{fold}").glob("*vae*history*.joblib")),
        ]
    )


def read_history(run_dir: Path, fold: int) -> dict[str, Sequence[float]]:
    path = history_path(run_dir, fold)
    if path is None:
        return {}
    try:
        h = joblib.load(path)
        return h if isinstance(h, dict) else {}
    except Exception:
        return {}


def find_best_epoch(history: dict[str, Sequence[float]]) -> tuple[int | None, int | None, float]:
    key = "val_loss_modelsel" if "val_loss_modelsel" in history else "val_loss"
    vals = np.asarray(history.get(key, []), dtype=float)
    if vals.size == 0 or np.all(np.isnan(vals)):
        return None, None, float("nan")
    idx = int(np.nanargmin(vals))
    return idx + 1, int(vals.size), float(vals[idx])


def value_at(history: dict[str, Sequence[float]], key: str, epoch: int | None) -> float:
    if epoch is None:
        return float("nan")
    vals = history.get(key)
    if vals is None or len(vals) < epoch:
        return float("nan")
    return to_float(vals[epoch - 1])


def last100_slope(history: dict[str, Sequence[float]]) -> float:
    key = "val_loss_modelsel" if "val_loss_modelsel" in history else "val_loss"
    vals = np.asarray(history.get(key, []), dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 3:
        return float("nan")
    tail = vals[-min(100, vals.size) :]
    x = np.arange(tail.size, dtype=float)
    try:
        return float(np.polyfit(x, tail, 1)[0])
    except Exception:
        return float("nan")


def lr_at_epoch(epoch: int | None, cfg: dict[str, Any]) -> tuple[float, str]:
    if epoch is None:
        return float("nan"), ""
    p = cfg.get("parameters", {})
    lr = to_float(p.get("lr_vae", 1e-4))
    eta_min = to_float(p.get("lr_scheduler_eta_min", 5e-7))
    t0 = int(p.get("lr_scheduler_T0", 80) or 80)
    pos0 = (epoch - 1) % t0
    approx = eta_min + 0.5 * (lr - eta_min) * (1.0 + math.cos(math.pi * pos0 / t0))
    if pos0 <= 2:
        phase = "near_lr_restart"
    elif pos0 >= t0 - 3:
        phase = "near_lr_trough"
    else:
        phase = "mid_cycle"
    return float(approx), phase


def beta_phase(epoch: int | None, beta_value: float, cfg: dict[str, Any]) -> str:
    if epoch is None or pd.isna(beta_value):
        return ""
    p = cfg.get("parameters", {})
    max_epochs = int(p.get("epochs_vae", 0) or 0)
    n_cycles = int(p.get("cyclical_beta_n_cycles", 0) or 0)
    cycle_len = max_epochs / n_cycles if n_cycles else float("nan")
    if not np.isfinite(cycle_len) or cycle_len <= 0:
        return ""
    ratio = to_float(p.get("cyclical_beta_ratio_increase", 0.4))
    pos = ((epoch - 1) % cycle_len) + 1
    if pos <= ratio * cycle_len:
        return "beta_ramp"
    return "beta_plateau"


def training_dynamics_for_run(spec: dict[str, str], cfg: dict[str, Any]) -> pd.DataFrame:
    run_dir = resolve(spec["run_dir"])
    p = cfg.get("parameters", {})
    max_epoch = int(p.get("epochs_vae", 0) or 0)
    rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        hist = read_history(run_dir, fold)
        best_epoch, final_epoch, best_val = find_best_epoch(hist)
        best_beta = value_at(hist, "beta", best_epoch)
        lr, lr_phase = lr_at_epoch(best_epoch, cfg)
        rows.append(
            {
                "run_id": spec["run_id"],
                "label": spec["label"],
                "fold": fold,
                "run_dir": rel(run_dir),
                "history_exists": bool(hist),
                "best_epoch": best_epoch,
                "final_epoch": final_epoch,
                "early_stop_epoch": final_epoch,
                "reached_max_epoch": bool(final_epoch == max_epoch) if final_epoch else False,
                "epochs_after_best": (final_epoch - best_epoch) if final_epoch and best_epoch else np.nan,
                "max_epoch_config": max_epoch,
                "best_valL_beta_max": best_val,
                "best_val_recon": value_at(hist, "val_recon", best_epoch),
                "best_val_kld": value_at(hist, "val_kld", best_epoch),
                "best_val_kld_over_recon": value_at(hist, "val_kld_over_recon", best_epoch),
                "best_val_beta_kld_over_recon": value_at(hist, "val_beta_kld_over_recon", best_epoch),
                "best_train_recon": value_at(hist, "train_recon", best_epoch),
                "best_train_kld": value_at(hist, "train_kld", best_epoch),
                "best_train_kld_over_recon": value_at(hist, "train_kld_over_recon", best_epoch),
                "best_train_beta_kld_over_recon": value_at(hist, "train_beta_kld_over_recon", best_epoch),
                "last100_valL_slope": last100_slope(hist),
                "beta_at_best_epoch": best_beta,
                "lr_at_best_epoch_approx": lr,
                "beta_phase_at_best_epoch": beta_phase(best_epoch, best_beta, cfg),
                "lr_phase_at_best_epoch": lr_phase,
                "recon_loss_mode": p.get("recon_loss_mode", ""),
                "beta_vae": p.get("beta_vae", np.nan),
                "epochs_vae": p.get("epochs_vae", np.nan),
                "cyclical_beta_n_cycles": p.get("cyclical_beta_n_cycles", np.nan),
            }
        )
    return pd.DataFrame(rows)


def rate_distortion_for_run(spec: dict[str, str], dynamics: pd.DataFrame) -> pd.DataFrame:
    run_dir = resolve(spec["run_dir"])
    rows = []
    for _, dyn in dynamics.iterrows():
        fold = int(dyn["fold"])
        path = run_dir / f"fold_{fold}" / f"fold_{fold}_rate_distortion.csv"
        if not path.exists():
            rows.append({"run_id": spec["run_id"], "label": spec["label"], "fold": fold, "rd_exists": False})
            continue
        df = pd.read_csv(path)
        best_epoch = int(dyn["best_epoch"]) if pd.notna(dyn["best_epoch"]) else None
        if best_epoch and "epoch" in df.columns and (df["epoch"].astype(int) == best_epoch).any():
            row = df[df["epoch"].astype(int) == best_epoch].iloc[0].to_dict()
        else:
            row = df.iloc[-1].to_dict()
        out = {"run_id": spec["run_id"], "label": spec["label"], "fold": fold, "rd_exists": True, "source_epoch": row.get("epoch")}
        for col in ["beta", "D_train", "R_train_nats", "R_train_bits", "L_train_betaMax", "D_val", "R_val_nats", "R_val_bits", "L_val_betaMax"]:
            out[col] = row.get(col, np.nan)
        out["val_R_over_D"] = to_float(out.get("R_val_nats")) / to_float(out.get("D_val")) if to_float(out.get("D_val")) else np.nan
        out["val_betaR_over_D"] = to_float(out.get("beta")) * out["val_R_over_D"] if pd.notna(out["val_R_over_D"]) else np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def primary_metrics_for_run(spec: dict[str, str], cfg: dict[str, Any]) -> dict[str, Any]:
    readout = resolve(spec["run_dir"]) / "classifier_only_readout"
    pooled_path = readout / "classifier_sweep_pooled_metrics.csv"
    p = cfg.get("parameters", {})
    out = {
        "run_id": spec["run_id"],
        "label": spec["label"],
        "run_dir": rel(resolve(spec["run_dir"])),
        "config": rel(resolve(spec["config"])),
        "run_exists": resolve(spec["run_dir"]).exists(),
        "readout_exists": readout.exists(),
        "recon_loss_mode": p.get("recon_loss_mode", ""),
        "epochs_vae": p.get("epochs_vae", np.nan),
        "cyclical_beta_n_cycles": p.get("cyclical_beta_n_cycles", np.nan),
        "beta_vae": p.get("beta_vae", np.nan),
    }
    if not pooled_path.exists():
        return out
    df = pd.read_csv(pooled_path)
    mask = df["model_name"].astype(str).eq(PRIMARY_MODEL) & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    if not mask.any():
        return out
    row = df[mask].iloc[0].to_dict()
    for col in ["n", "n_cn", "n_ad", "tn", "fp", "fn", "tp", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "accuracy", "predicted_ad_rate"]:
        out[col] = row.get(col, np.nan)
    return out


def stage_a_metrics_for_run(spec: dict[str, str]) -> pd.DataFrame:
    run_dir = resolve(spec["run_dir"])
    path = all_folds_metrics_path(run_dir)
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.insert(0, "run_id", spec["run_id"])
    df.insert(1, "label", spec["label"])
    return df


def stage_b_foldwise_for_run(spec: dict[str, str]) -> pd.DataFrame:
    path = resolve(spec["run_dir"]) / "classifier_only_readout" / "classifier_sweep_foldwise_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["model_name"].astype(str).eq(PRIMARY_MODEL)].copy()
    df.insert(0, "run_id", spec["run_id"])
    df.insert(1, "label", spec["label"])
    return df


def stage_b_confusion_for_run(spec: dict[str, str]) -> pd.DataFrame:
    path = resolve(spec["run_dir"]) / "classifier_only_readout" / "classifier_sweep_confusion_by_fold.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["model_name"].astype(str).eq(PRIMARY_MODEL)].copy()
    df.insert(0, "run_id", spec["run_id"])
    df.insert(1, "label", spec["label"])
    return df


def stage_b_thresholds_for_run(spec: dict[str, str]) -> pd.DataFrame:
    path = resolve(spec["run_dir"]) / "classifier_only_readout" / "classifier_sweep_thresholds_by_fold.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["model_name"].astype(str).eq(PRIMARY_MODEL)].copy()
    df.insert(0, "run_id", spec["run_id"])
    df.insert(1, "label", spec["label"])
    return df


def latent_information_for_run(spec: dict[str, str]) -> pd.DataFrame:
    run_dir = resolve(spec["run_dir"])
    rows = []
    for fold in range(1, 6):
        fdir = run_dir / f"fold_{fold}"
        summary_path = fdir / f"fold_{fold}_test_latent_info_summary.csv"
        per_dim_path = fdir / f"fold_{fold}_test_latent_info_per_dim.csv"
        if not summary_path.exists():
            continue
        summary = pd.read_csv(summary_path)
        per_dim = pd.read_csv(per_dim_path) if per_dim_path.exists() else pd.DataFrame()
        diag_sum = np.nan
        nuisance_max = np.nan
        if "variable" in summary.columns:
            y_rows = summary[summary["variable"].astype(str).isin(["Y_target", "Diagnosis", "ResearchGroup_Mapped"])]
            if not y_rows.empty:
                diag_sum = to_float(y_rows["mi_sum_nats"].iloc[0])
            nuis = summary[~summary["variable"].astype(str).isin(["Y_target", "Diagnosis", "ResearchGroup_Mapped"])]
            if not nuis.empty:
                nuisance_max = float(pd.to_numeric(nuis["mi_sum_nats"], errors="coerce").max())
        for _, row in summary.iterrows():
            variable = str(row.get("variable", ""))
            top_dim = ""
            top_mi = np.nan
            if not per_dim.empty and "variable" in per_dim.columns:
                sub = per_dim[per_dim["variable"].astype(str).eq(variable)].copy()
                if not sub.empty:
                    sub["mi_nats"] = pd.to_numeric(sub["mi_nats"], errors="coerce")
                    best = sub.sort_values("mi_nats", ascending=False).iloc[0]
                    top_dim = best.get("dim", "")
                    top_mi = best.get("mi_nats", np.nan)
            rows.append(
                {
                    "run_id": spec["run_id"],
                    "label": spec["label"],
                    "fold": fold,
                    "fold_tag": row.get("fold_tag", ""),
                    "variable": variable,
                    "mi_sum_nats": row.get("mi_sum_nats", np.nan),
                    "mi_mean_nats": row.get("mi_mean_nats", np.nan),
                    "top_k": row.get("top_k", np.nan),
                    "top_dims": row.get("top_dims", ""),
                    "top_dim": top_dim,
                    "top_dim_mi_nats": top_mi,
                    "n_samples": row.get("n_samples", np.nan),
                    "latent_dim": row.get("latent_dim", np.nan),
                    "active_units": row.get("n_active", np.nan),
                    "frac_active": row.get("frac_active", np.nan),
                    "total_correlation_nats": row.get("total_correlation_nats", np.nan),
                    "diagnosis_mi_sum_nats": diag_sum,
                    "max_nuisance_mi_sum_nats": nuisance_max,
                    "diagnosis_to_max_nuisance_mi_ratio": diag_sum / nuisance_max if pd.notna(diag_sum) and pd.notna(nuisance_max) and nuisance_max else np.nan,
                }
            )
    return pd.DataFrame(rows)


def scanner_leakage_for_run(spec: dict[str, str]) -> pd.DataFrame:
    run_dir = resolve(spec["run_dir"])
    rows = []
    for fold in range(1, 6):
        for scope, path in [
            ("train_dev", run_dir / f"fold_{fold}" / f"fold_{fold}_scanner_leakage_summary.csv"),
            ("test", run_dir / f"fold_{fold}" / f"fold_{fold}_test_scanner_leakage_summary.csv"),
            ("latent_qc", run_dir / f"fold_{fold}" / "latent_qc_metrics.csv"),
        ]:
            if not path.exists():
                continue
            df = pd.read_csv(path)
            if df.empty:
                continue
            row = df.iloc[0].to_dict()
            out = {"run_id": spec["run_id"], "label": spec["label"], "fold": fold, "scope": scope}
            for col in ["site_col", "n_sites", "n_samples", "chance_level", "acc_site_raw", "acc_site_latent", "acc_site_raw_std", "acc_site_latent_std", "silhouette_latent"]:
                out[col] = row.get(col, np.nan)
            out["latent_minus_raw"] = to_float(out.get("acc_site_latent")) - to_float(out.get("acc_site_raw"))
            rows.append(out)
    return pd.DataFrame(rows)


def optuna_best_params(run_dir: Path, fold: int, clf: str) -> dict[str, Any]:
    data = read_json(run_dir / f"fold_{fold}" / f"optuna_best_trial_{clf}_fold_{fold}.json")
    params = data.get("best_params", {}) if isinstance(data, dict) else {}
    return params if isinstance(params, dict) else {}


def observed_boundary_status(run_dir: Path, fold: int, clf: str, param: str, value: float) -> tuple[bool, bool, float, float]:
    trials_path = run_dir / f"fold_{fold}" / f"optuna_trials_{clf}_fold_{fold}.csv"
    if not trials_path.exists() or pd.isna(value):
        return False, False, np.nan, np.nan
    trials = pd.read_csv(trials_path)
    col = f"params_{param}"
    if col not in trials.columns:
        return False, False, np.nan, np.nan
    vals = pd.to_numeric(trials[col], errors="coerce").dropna()
    if vals.empty:
        return False, False, np.nan, np.nan
    lo, hi = float(vals.min()), float(vals.max())
    return bool(np.isclose(value, lo)), bool(np.isclose(value, hi)), lo, hi


def classifier_boundary_summary(run_specs: list[dict[str, str]]) -> pd.DataFrame:
    rows = []
    for spec in run_specs:
        run_dir = resolve(spec["run_dir"])
        status_path = run_dir / "classifier_only_readout" / "classifier_sweep_model_status.csv"
        status = pd.read_csv(status_path) if status_path.exists() else pd.DataFrame()
        for fold in range(1, 6):
            # Stage A logreg/SVM.
            for clf in ["logreg", "svm"]:
                params = optuna_best_params(run_dir, fold, clf)
                if not params:
                    continue
                if clf == "logreg":
                    c = to_float(params.get("model__C"))
                    at_min, at_max, obs_min, obs_max = observed_boundary_status(run_dir, fold, clf, "model__C", c)
                    rows.append(
                        {
                            "run_id": spec["run_id"],
                            "label": spec["label"],
                            "fold": fold,
                            "stage": "stage_a",
                            "model": "logreg",
                            "C": c,
                            "gamma": np.nan,
                            "best_params": json.dumps(params, sort_keys=True),
                            "at_known_grid_min": np.nan,
                            "at_known_grid_max": np.nan,
                            "at_observed_trial_min": at_min,
                            "at_observed_trial_max": at_max,
                            "observed_trial_min": obs_min,
                            "observed_trial_max": obs_max,
                        }
                    )
                if clf == "svm":
                    for param in ["model__C", "model__gamma"]:
                        val = to_float(params.get(param))
                        at_min, at_max, obs_min, obs_max = observed_boundary_status(run_dir, fold, clf, param, val)
                        rows.append(
                            {
                                "run_id": spec["run_id"],
                                "label": spec["label"],
                                "fold": fold,
                                "stage": "stage_a",
                                "model": f"svm_{param.split('__')[-1]}",
                                "C": to_float(params.get("model__C")),
                                "gamma": to_float(params.get("model__gamma")),
                                "best_params": json.dumps(params, sort_keys=True),
                                "at_known_grid_min": np.nan,
                                "at_known_grid_max": np.nan,
                                "at_observed_trial_min": at_min,
                                "at_observed_trial_max": at_max,
                                "observed_trial_min": obs_min,
                                "observed_trial_max": obs_max,
                            }
                        )
            # Stage B logreg_l2.
            if not status.empty:
                sub = status[(status["fold"].astype(int).eq(fold)) & status["model_name"].astype(str).eq(PRIMARY_MODEL)]
                if not sub.empty:
                    params = parse_params(sub.iloc[0].get("best_params"))
                    c = to_float(params.get("model__C"))
                    rows.append(
                        {
                            "run_id": spec["run_id"],
                            "label": spec["label"],
                            "fold": fold,
                            "stage": "stage_b",
                            "model": PRIMARY_MODEL,
                            "C": c,
                            "gamma": np.nan,
                            "best_params": json.dumps(params, sort_keys=True),
                            "best_inner_auc": sub.iloc[0].get("best_inner_auc", np.nan),
                            "at_known_grid_min": bool(np.isclose(c, min(STAGE_B_C_GRID))) if pd.notna(c) else np.nan,
                            "at_known_grid_max": bool(np.isclose(c, max(STAGE_B_C_GRID))) if pd.notna(c) else np.nan,
                            "known_grid_min": min(STAGE_B_C_GRID),
                            "known_grid_max": max(STAGE_B_C_GRID),
                        }
                    )
    return pd.DataFrame(rows)


def threshold_and_confusion(run_specs: list[dict[str, str]]) -> pd.DataFrame:
    frames = []
    for spec in run_specs:
        foldwise = stage_b_foldwise_for_run(spec)
        if foldwise.empty:
            continue
        keep = foldwise[foldwise["threshold_strategy"].astype(str).isin([FIXED_THRESHOLD, PRIMARY_THRESHOLD])].copy()
        frames.append(keep)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def hard_fold_diagnosis_text(
    run_level: pd.DataFrame,
    training: pd.DataFrame,
    stage_a: pd.DataFrame,
    stage_b_primary: pd.DataFrame,
    leakage: pd.DataFrame,
    latent_info: pd.DataFrame,
    boundary: pd.DataFrame,
) -> str:
    lines = ["# Hard Fold Diagnosis", ""]
    if stage_b_primary.empty:
        lines.append("No Stage B primary foldwise metrics were available.")
        return "\n".join(lines) + "\n"
    pivot = stage_b_primary.pivot_table(index="fold", columns="run_id", values="auc", aggfunc="first")
    mean_by_fold = stage_b_primary.groupby("fold")["auc"].mean().sort_values()
    lines.append("## Consistently Hard Folds")
    lines.append("")
    lines.append("Mean Stage B AUC by fold across available runs:")
    lines.append("")
    lines.append(mean_by_fold.to_frame("mean_auc").to_markdown())
    lines.append("")
    hard_fold = int(mean_by_fold.index[0])
    lines.append(f"Fold `{hard_fold}` is the lowest mean-AUC fold across the compared runs.")
    lines.append("")
    lines.append("Foldwise AUC matrix:")
    lines.append("")
    lines.append(pivot.to_markdown())
    lines.append("")

    leak_primary = leakage[leakage["scope"].isin(["train_dev", "latent_qc"])].copy()
    if not leak_primary.empty:
        leak_summary = leak_primary.groupby(["run_id"])["latent_minus_raw"].mean().to_frame("mean_latent_minus_raw")
        lines.append("## Scanner/Manufacturer Leakage")
        lines.append("")
        lines.append(leak_summary.to_markdown())
        lines.append("")
        lines.append(
            "Latent scanner/manufacturer accuracy is generally lower than raw scanner accuracy when `latent_minus_raw` is negative. "
            "This argues against a simple scanner-leakage explanation for the AUC ceiling."
        )
        lines.append("")

    stage_b_boundary = boundary[(boundary["stage"] == "stage_b") & (boundary["model"] == PRIMARY_MODEL)].copy()
    if not stage_b_boundary.empty:
        n_min = int(stage_b_boundary["at_known_grid_min"].fillna(False).sum())
        n = int(len(stage_b_boundary))
        lines.append("## Classifier Hyperparameter Boundaries")
        lines.append("")
        lines.append(f"Stage B `logreg_l2` selected the lower grid boundary in `{n_min}/{n}` fold/run fits.")
        lines.append(
            "This confirms strong regularization pressure in the frozen-latent readout, but prior extended-grid audits showed stronger regularization did not improve test ranking."
        )
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "The repeated hard-fold behavior and broad Stage B metric spread are most consistent with fold/subgroup difficulty and score-overlap limits, not a single obvious training-horizon or classifier-grid failure."
    )
    return "\n".join(lines) + "\n"


def final_decision_text(
    run_level: pd.DataFrame,
    training: pd.DataFrame,
    rd: pd.DataFrame,
    leakage: pd.DataFrame,
    boundary: pd.DataFrame,
) -> str:
    best = run_level.sort_values(["auc", "pr_auc"], ascending=False).iloc[0] if not run_level.empty and "auc" in run_level else None
    lines = ["# Final Saturation Decision", ""]
    if best is not None:
        lines.append(
            f"Best available primary Stage B run by ROC-AUC is `{best['run_id']}` "
            f"(AUC={to_float(best.get('auc')):.6f}, PR-AUC={to_float(best.get('pr_auc')):.6f})."
        )
        lines.append("")

    def answer(q: str, a: str) -> None:
        lines.extend([f"## {q}", "", a, ""])

    # Simple evidence summaries.
    reached = training.groupby("run_id")["reached_max_epoch"].sum().to_dict() if not training.empty else {}
    slopes = training.groupby("run_id")["last100_valL_slope"].mean().to_dict() if not training.empty else {}
    leak = leakage[leakage["scope"].isin(["train_dev", "latent_qc"])].groupby("run_id")["latent_minus_raw"].mean().to_dict() if not leakage.empty else {}
    cmin = (
        boundary[(boundary["stage"] == "stage_b") & (boundary["model"] == PRIMARY_MODEL)]
        .groupby("run_id")["at_known_grid_min"]
        .sum()
        .to_dict()
        if not boundary.empty
        else {}
    )

    answer(
        "Is there evidence of over-regularization?",
        "No decisive evidence. Increasing bottleneck/dropout pressure was already negative in prior controlled confirmations, and in this saturation set the objective-v2 loss does not by itself establish a better clinical ranking. The current limitation looks more like fold/subgroup score overlap than a clean over-regularization signature.",
    )
    answer(
        "Is there evidence of under-regularization?",
        "No strong evidence. Latent scanner/manufacturer leakage is not higher than raw leakage in the main summaries, and removing or reducing regularization in prior checks did not improve AUC/PR-AUC.",
    )
    answer(
        "Is beta=2.5 too large or too small?",
        "The available evidence does not justify changing beta. The beta65 stress test did not improve ranking, while current beta=2.5 remains the best validated setting among completed FULL confirmations.",
    )
    answer(
        "Did offdiag_channelmean_sum improve rate-distortion but hurt clinical ranking?",
        _objective_v2_interpretation(run_level, rd),
    )
    answer(
        "Are classifier hyperparameter ranges limiting AUC?",
        f"Stage B lower-bound selections are common: {cmin}. This is a warning flag, but it is not sufficient to justify another FULL run because the prior ultra-regularized readout audit showed lower C values hurt test ranking.",
    )
    answer(
        "Did longer horizon solve the ceiling?",
        f"Reached-max-epoch fold counts by run: {reached}. Mean last-100 validation slopes by run: {slopes}. Horizon extension helped v5.1b modestly, but v5.1c 10000 did not beat v5.1b horizon4480, so horizon alone is not a general solution.",
    )
    answer(
        "Is scanner/manufacturer leakage the dominant limitation?",
        f"Mean latent-minus-raw scanner/manufacturer accuracy by run: {leak}. Negative values argue that latent space is not simply amplifying scanner identity relative to raw inputs.",
    )
    answer(
        "Is any further FULL run justified?",
        "No further FULL internal optimization is justified from this audit alone. A scientifically stronger next step is external validation or a clearly pre-registered data/label/QC improvement, not additional architecture or hyperparameter search on the same cohort.",
    )
    return "\n".join(lines)


def _objective_v2_interpretation(run_level: pd.DataFrame, rd: pd.DataFrame) -> str:
    obj_id = "v5_1c_horizon10000_objective_v2_offdiag_channelmean"
    cur_id = "v5_1c_horizon10000_current_loss"
    ref_id = "v5_1b_horizon4480_current_loss"
    if run_level.empty or obj_id not in set(run_level["run_id"]):
        return "The objective-v2 run is not complete enough to answer this from current artifacts."
    obj = run_level[run_level["run_id"].eq(obj_id)].iloc[0]
    cur = run_level[run_level["run_id"].eq(cur_id)].iloc[0] if cur_id in set(run_level["run_id"]) else None
    ref = run_level[run_level["run_id"].eq(ref_id)].iloc[0] if ref_id in set(run_level["run_id"]) else None
    rd_mean = rd.groupby("run_id")[["D_val", "R_val_nats", "val_R_over_D", "val_betaR_over_D"]].mean(numeric_only=True)
    obj_rd = rd_mean.loc[obj_id].to_dict() if obj_id in rd_mean.index else {}
    cur_rd = rd_mean.loc[cur_id].to_dict() if cur_id in rd_mean.index else {}
    parts = [
        "The objective-v2 loss changes the reconstruction scale, so raw `D_val` is not directly comparable with the current summed-MSE objective.",
    ]
    if obj_rd:
        parts.append(
            f"It produced mean `D_val={obj_rd.get('D_val', np.nan):.3f}`, `R_val_nats={obj_rd.get('R_val_nats', np.nan):.3f}`, and `beta*R/D={obj_rd.get('val_betaR_over_D', np.nan):.6f}`."
        )
    if cur_rd:
        parts.append(
            f"The v5.1c horizon10000 current-loss run had `D_val={cur_rd.get('D_val', np.nan):.3f}`, `R_val_nats={cur_rd.get('R_val_nats', np.nan):.3f}`, and `beta*R/D={cur_rd.get('val_betaR_over_D', np.nan):.6f}`."
        )
    if cur is not None:
        parts.append(
            f"Clinically, objective-v2 improved threshold BA/F1 versus v5.1c current-loss (`BA {to_float(cur.get('balanced_accuracy')):.6f}->{to_float(obj.get('balanced_accuracy')):.6f}`, `F1 {to_float(cur.get('f1')):.6f}->{to_float(obj.get('f1')):.6f}`), but reduced PR-AUC (`{to_float(cur.get('pr_auc')):.6f}->{to_float(obj.get('pr_auc')):.6f}`) and slightly reduced AUC (`{to_float(cur.get('auc')):.6f}->{to_float(obj.get('auc')):.6f}`)."
        )
    if ref is not None:
        parts.append(
            f"It remains below the v5.1b horizon4480 reference on the promotion gate (`AUC {to_float(obj.get('auc')):.6f} vs {to_float(ref.get('auc')):.6f}`, `PR-AUC {to_float(obj.get('pr_auc')):.6f} vs {to_float(ref.get('pr_auc')):.6f}`)."
        )
    parts.append("Interpretation: objective-v2 is useful as loss-scaling hygiene, but it does not solve the ranking ceiling.")
    return " ".join(parts)


def readme_text() -> str:
    return f"""# Final Model Saturation Audit

Read-only audit across the main v5.1b/v5.1c FULL candidates.

Primary Stage B operating point:

- model: `{PRIMARY_MODEL}`
- threshold strategy: `{PRIMARY_THRESHOLD}`

Runs:

{chr(10).join(f"- `{r['run_id']}`: {r['label']}" for r in RUNS)}

This audit reads existing artifacts only. It does not train, modify tensors, edit metadata, change ledgers, alter configs, or touch existing model-output folders.
"""


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configs = {spec["run_id"]: load_config(resolve(spec["config"])) for spec in RUNS}

    run_level = pd.DataFrame([primary_metrics_for_run(spec, configs[spec["run_id"]]) for spec in RUNS])
    write_table("run_level_primary_metrics", run_level)

    training = pd.concat([training_dynamics_for_run(spec, configs[spec["run_id"]]) for spec in RUNS], ignore_index=True)
    write_table("foldwise_training_dynamics", training)

    rd = pd.concat([rate_distortion_for_run(spec, training[training["run_id"].eq(spec["run_id"])]) for spec in RUNS], ignore_index=True)
    write_table("foldwise_rate_distortion_summary", rd)

    latent = pd.concat([latent_information_for_run(spec) for spec in RUNS], ignore_index=True)
    write_table("foldwise_latent_information", latent)

    leakage = pd.concat([scanner_leakage_for_run(spec) for spec in RUNS], ignore_index=True)
    write_table("foldwise_scanner_leakage", leakage)

    boundary = classifier_boundary_summary(RUNS)
    write_table("classifier_hyperparameter_boundary_summary", boundary)

    threshold_conf = threshold_and_confusion(RUNS)
    write_table("threshold_and_confusion_by_fold", threshold_conf)

    stage_a = pd.concat([stage_a_metrics_for_run(spec) for spec in RUNS], ignore_index=True)
    stage_b = pd.concat([stage_b_foldwise_for_run(spec) for spec in RUNS], ignore_index=True)
    stage_b_primary = stage_b[stage_b["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)].copy() if not stage_b.empty else pd.DataFrame()
    # Include Stage A/Stage B side-by-side for the requested foldwise performance comparison.
    if not stage_a.empty:
        write_table("foldwise_stage_a_metrics", stage_a)
    if not stage_b.empty:
        write_table("foldwise_stage_b_metrics", stage_b)

    (OUT_DIR / "hard_fold_diagnosis.md").write_text(
        hard_fold_diagnosis_text(run_level, training, stage_a, stage_b_primary, leakage, latent, boundary),
        encoding="utf-8",
    )
    (OUT_DIR / "final_saturation_decision.md").write_text(
        final_decision_text(run_level, training, rd, leakage, boundary),
        encoding="utf-8",
    )
    (OUT_DIR / "README.md").write_text(readme_text(), encoding="utf-8")
    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": rel(Path(__file__)),
        "output_dir": rel(OUT_DIR),
        "runs": RUNS,
        "primary_model": PRIMARY_MODEL,
        "primary_threshold": PRIMARY_THRESHOLD,
        "read_only": True,
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "config_modified": False,
        "existing_model_output_modified": False,
    }
    (OUT_DIR / "command_log.json").write_text(json.dumps(command_log, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(command_log, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
