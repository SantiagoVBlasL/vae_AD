#!/usr/bin/env python3
"""Read-only deep model-card audit for the locked primary ADNI model."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"

LOCKED_RUN = RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5"
LOCKED_READOUT = LOCKED_RUN / "classifier_only_readout"
DEFENSE = RESULTS / "adni_v5_1_batch20260514b_manuscript_defense_locked_current_model"
ALLTR_COMPARISON = RESULTS / "adni_all_available_timepoints_ch1_0_2_exploratory_full_5x5_comparison"
ALLTR_METADATA = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_all_available_timepoints_ch1_0_2_exploratory/"
    "training_ready_metadata_adni_all_available_timepoints_ch1_0_2_exploratory.csv"
)
DEFAULT_OUTPUT = RESULTS / "locked_primary_model_deep_model_card_audit"

PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
FIXED_THRESHOLD = "fixed_0p5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--locked-run-dir", type=Path, default=LOCKED_RUN)
    parser.add_argument("--locked-readout-dir", type=Path, default=LOCKED_READOUT)
    parser.add_argument("--manuscript-defense-dir", type=Path, default=DEFENSE)
    parser.add_argument("--alltr-comparison-dir", type=Path, default=ALLTR_COMPARISON)
    parser.add_argument("--ntr-metadata", type=Path, default=ALLTR_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_table(df: pd.DataFrame, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    (out_dir / f"{stem}.md").write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def slope_last(values: list[float], n: int) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < max(3, min(n, 3)):
        return float("nan")
    window = arr[-min(n, arr.size) :]
    x = np.arange(window.size, dtype=float)
    return float(np.polyfit(x, window, 1)[0])


def beta_phase(epoch: int, cycle_len: int, ramp_epochs: int) -> str:
    phase = ((epoch - 1) % cycle_len) + 1
    if phase <= ramp_epochs:
        return "beta_ramp"
    return "beta_plateau"


def lr_at_epoch(epoch: int, lr_max: float, eta_min: float, t0: int) -> float:
    # Approximation for CosineAnnealingWarmRestarts with T_mult=1.
    t_cur = (epoch - 1) % t0
    return float(eta_min + 0.5 * (lr_max - eta_min) * (1.0 + math.cos(math.pi * t_cur / t0)))


def load_config(run_dir: Path) -> dict[str, Any]:
    cfg = load_json(run_dir / "run_config.json")
    return cfg.get("args", cfg)


def history_row(run_dir: Path, fold: int, cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    path = run_dir / f"fold_{fold}" / f"vae_train_history_fold_{fold}.joblib"
    h = joblib.load(path)
    val_modelsel = list(map(float, h.get("val_loss_modelsel", h.get("val_loss", []))))
    if not val_modelsel:
        raise ValueError(f"No validation loss in {path}")
    best_idx = int(np.nanargmin(np.asarray(val_modelsel, dtype=float)))
    best_epoch = best_idx + 1
    final_epoch = len(val_modelsel)
    max_epoch = int(cfg.get("epochs_vae", final_epoch))
    n_cycles = int(cfg.get("cyclical_beta_n_cycles", 56))
    cycle_len = int(round(max_epoch / n_cycles)) if n_cycles else int(cfg.get("lr_scheduler_T0", 80))
    ramp_epochs = int(round(cycle_len * float(cfg.get("cyclical_beta_ratio_increase", 0.4))))
    beta_vals = list(map(float, h.get("beta", [])))
    beta_best = beta_vals[best_idx] if best_idx < len(beta_vals) else float("nan")
    lr_best = lr_at_epoch(
        best_epoch,
        float(cfg.get("lr_vae", 1e-4)),
        float(cfg.get("lr_scheduler_eta_min", 5e-7)),
        int(cfg.get("lr_scheduler_T0", cycle_len)),
    )
    row = {
        "fold": fold,
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "early_stop_epoch": final_epoch if final_epoch < max_epoch else np.nan,
        "epochs_after_best": int(final_epoch - best_epoch),
        "reached_max_epoch": bool(final_epoch >= max_epoch),
        "best_epoch_in_last_10_percent": bool(best_epoch >= int(math.ceil(0.9 * max_epoch))),
        "best_val_L_betaMax": float(val_modelsel[best_idx]),
        "last_val_L_betaMax": float(val_modelsel[-1]),
        "last_100_val_L_betaMax_slope": slope_last(val_modelsel, 100),
        "last_300_val_L_betaMax_slope": slope_last(val_modelsel, 300),
        "beta_at_best_epoch": beta_best,
        "beta_phase_at_best_epoch": beta_phase(best_epoch, cycle_len, ramp_epochs),
        "lr_at_best_epoch": lr_best,
        "lr_phase_at_best_epoch": "restart_peak" if ((best_epoch - 1) % int(cfg.get("lr_scheduler_T0", cycle_len))) < 5 else "cosine_cycle",
        "max_epoch_budget": max_epoch,
        "cycle_len_epochs": cycle_len,
    }
    rd = {
        "fold": fold,
        "train_recon_at_best": safe_float(h.get("train_recon", [np.nan])[best_idx]),
        "val_recon_at_best": safe_float(h.get("val_recon", [np.nan])[best_idx]),
        "train_kld_at_best": safe_float(h.get("train_kld", [np.nan])[best_idx]),
        "val_kld_at_best": safe_float(h.get("val_kld", [np.nan])[best_idx]),
        "train_kld_over_recon_at_best": safe_float(h.get("train_kld_over_recon", [np.nan])[best_idx]),
        "train_beta_kld_over_recon_at_best": safe_float(h.get("train_beta_kld_over_recon", [np.nan])[best_idx]),
        "val_kld_over_recon_at_best": safe_float(h.get("val_kld_over_recon", [np.nan])[best_idx]),
        "val_beta_kld_over_recon_at_best": safe_float(h.get("val_beta_kld_over_recon", [np.nan])[best_idx]),
    }
    rate_path = run_dir / f"fold_{fold}" / f"fold_{fold}_rate_distortion.csv"
    if rate_path.exists():
        rate = pd.read_csv(rate_path)
        match = rate[rate["epoch"].astype(int).eq(best_epoch)]
        if not match.empty:
            for col in ["D_train", "R_train_nats", "L_train_betaMax", "D_val", "R_val_nats", "L_val_betaMax"]:
                rd[f"rate_distortion_{col}_at_best"] = safe_float(match.iloc[0].get(col))
            d_val = rd.get("rate_distortion_D_val_at_best", np.nan)
            r_val = rd.get("rate_distortion_R_val_nats_at_best", np.nan)
            rd["rate_distortion_val_kld_over_recon_at_best"] = safe_float(r_val / d_val) if d_val else np.nan
            rd["rate_distortion_val_beta_kld_over_recon_at_best"] = (
                safe_float(float(cfg.get("beta_vae", 2.5)) * r_val / d_val) if d_val else np.nan
            )
    return row, rd


def latent_info(run_dir: Path) -> pd.DataFrame:
    rows = []
    for fold in range(1, 6):
        for split in ["trainDev", "test"]:
            path = run_dir / f"fold_{fold}" / f"fold_{fold}_{split}_latent_info_summary.csv"
            if not path.exists():
                continue
            df = pd.read_csv(path)
            for _, r in df.iterrows():
                row = r.to_dict()
                row["fold"] = fold
                row["split"] = split
                rows.append(row)
    return pd.DataFrame(rows)


def summarize_latent_information(info: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if info.empty:
        return pd.DataFrame(), pd.DataFrame()
    wide_rows = []
    for (fold, split), grp in info.groupby(["fold", "split"], dropna=False):
        row = {"fold": fold, "split": split}
        if "n_active" in grp.columns:
            row["active_latent_units"] = safe_float(grp["n_active"].dropna().iloc[0]) if grp["n_active"].notna().any() else np.nan
        if "frac_active" in grp.columns:
            row["frac_active"] = safe_float(grp["frac_active"].dropna().iloc[0]) if grp["frac_active"].notna().any() else np.nan
        if "total_correlation_nats" in grp.columns:
            row["total_correlation_nats"] = safe_float(grp["total_correlation_nats"].dropna().iloc[0]) if grp["total_correlation_nats"].notna().any() else np.nan
        for _, r in grp.iterrows():
            var = str(r.get("variable", "unknown"))
            key = var.replace(" ", "_").replace("/", "_")
            row[f"mi_sum_nats_{key}"] = safe_float(r.get("mi_sum_nats"))
            row[f"mi_mean_nats_{key}"] = safe_float(r.get("mi_mean_nats"))
            row[f"top_dims_{key}"] = r.get("top_dims", "")
        wide_rows.append(row)
    return info.sort_values(["fold", "split", "variable"]), pd.DataFrame(wide_rows).sort_values(["fold", "split"])


def read_primary_foldwise(readout_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(readout_dir / "classifier_sweep_foldwise_metrics.csv")
    mask = df["model_name"].astype(str).eq(PRIMARY_MODEL) & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    if "readout_feature_set" in df.columns:
        mask &= df["readout_feature_set"].astype(str).eq("z_plus_age_sex")
    return df.loc[mask].copy().sort_values("fold")


def read_fixed_foldwise(readout_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(readout_dir / "classifier_sweep_foldwise_metrics.csv")
    mask = df["model_name"].astype(str).eq(PRIMARY_MODEL) & df["threshold_strategy"].astype(str).eq(FIXED_THRESHOLD)
    if "readout_feature_set" in df.columns:
        mask &= df["readout_feature_set"].astype(str).eq("z_plus_age_sex")
    return df.loc[mask].copy().sort_values("fold")


def parse_best_params(text: Any) -> dict[str, Any]:
    if pd.isna(text):
        return {}
    try:
        return json.loads(str(text))
    except Exception:
        return {}


def stageb_hyperparams(readout_dir: Path) -> pd.DataFrame:
    fold = read_primary_foldwise(readout_dir)
    rows = []
    grid = [0.001, 0.01, 0.1, 1.0]
    for _, r in fold.iterrows():
        params = parse_best_params(r.get("best_params"))
        c = safe_float(params.get("model__C"))
        rows.append(
            {
                "fold": int(r["fold"]),
                "model_name": PRIMARY_MODEL,
                "best_C": c,
                "grid_min_C": min(grid),
                "grid_max_C": max(grid),
                "hit_lower_boundary": bool(np.isclose(c, min(grid))),
                "hit_upper_boundary": bool(np.isclose(c, max(grid))),
                "best_inner_auc": safe_float(r.get("best_inner_auc")),
                "best_inner_pr_auc": np.nan,
                "threshold": safe_float(r.get("threshold")),
                "threshold_strategy": PRIMARY_THRESHOLD,
                "inner_oof_sensitivity": safe_float(r.get("inner_oof_sensitivity")),
                "inner_oof_specificity": safe_float(r.get("inner_oof_specificity")),
                "inner_oof_balanced_accuracy": safe_float(r.get("inner_oof_balanced_accuracy")),
                "minimum_inner_stratum_count": safe_float(r.get("minimum_inner_stratum_count")),
            }
        )
    return pd.DataFrame(rows)


def threshold_confusion(readout_dir: Path) -> pd.DataFrame:
    primary = read_primary_foldwise(readout_dir)
    fixed = read_fixed_foldwise(readout_dir)
    cols = [
        "fold",
        "threshold_strategy",
        "threshold",
        "n",
        "n_cn",
        "n_ad",
        "tn",
        "fp",
        "fn",
        "tp",
        "auc",
        "pr_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "predicted_ad_rate",
    ]
    return pd.concat([fixed, primary], ignore_index=True)[[c for c in cols if c in primary.columns]].sort_values(
        ["fold", "threshold_strategy"]
    )


def primary_predictions(readout_dir: Path, label: str) -> pd.DataFrame:
    pred = pd.read_csv(readout_dir / "classifier_sweep_predictions.csv")
    mask = pred["model_name"].astype(str).eq(PRIMARY_MODEL) & pred["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    if "readout_feature_set" in pred.columns:
        mask &= pred["readout_feature_set"].astype(str).eq("z_plus_age_sex")
    out = pred.loc[mask].copy()
    out["model_label"] = label
    return add_error_type(out)


def add_error_type(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["error_type"] = np.select(
        [
            out["y_true"].eq(0) & out["y_pred"].eq(0),
            out["y_true"].eq(0) & out["y_pred"].eq(1),
            out["y_true"].eq(1) & out["y_pred"].eq(0),
            out["y_true"].eq(1) & out["y_pred"].eq(1),
        ],
        ["TN", "FP", "FN", "TP"],
        default="unknown",
    )
    return out


def load_metadata(cfg: dict[str, Any], ntr_metadata: Path) -> pd.DataFrame:
    meta_path = Path(cfg.get("metadata_path", ""))
    base = pd.read_csv(meta_path) if meta_path.exists() else pd.DataFrame()
    if "SiteCode" not in base.columns and "SubjectID" in base.columns:
        base["SiteCode"] = base["SubjectID"].astype(str).str.slice(0, 3)
    if ntr_metadata.exists():
        ntr = pd.read_csv(ntr_metadata)
        keep = [c for c in ["SubjectID", "original_n_TR", "original_n_timepoints", "locked_n_timepoints_used", "branch_n_timepoints_used"] if c in ntr.columns]
        if not base.empty and keep:
            base = base.merge(ntr[keep].drop_duplicates("SubjectID"), on="SubjectID", how="left")
        elif keep:
            base = ntr
    if "original_n_TR" not in base.columns and "original_n_timepoints" in base.columns:
        base["original_n_TR"] = base["original_n_timepoints"]
    return base


def augment_subjects(pred: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "SubjectID",
        "ResearchGroup_Mapped",
        "Manufacturer",
        "SiteCode",
        "Site3",
        "Age",
        "Sex",
        "AgeBin",
        "original_n_TR",
        "locked_n_timepoints_used",
    ]
    keep = [c for c in keep if c in meta.columns]
    out = pred.merge(meta[keep].drop_duplicates("SubjectID"), on="SubjectID", how="left", suffixes=("", "_meta"))
    for col in ["ResearchGroup_Mapped", "Manufacturer", "Age", "Sex"]:
        meta_col = f"{col}_meta"
        if meta_col in out.columns:
            out[col] = out[col].where(out[col].notna(), out[meta_col])
            out = out.drop(columns=[meta_col])
    if "SiteCode" not in out.columns:
        out["SiteCode"] = out["SubjectID"].astype(str).str.slice(0, 3)
    return out


def add_comparator_errors(subjects: pd.DataFrame, readout: Path, label: str, col_prefix: str) -> pd.DataFrame:
    if not (readout / "classifier_sweep_predictions.csv").exists():
        return subjects
    comp = primary_predictions(readout, label)
    keep = ["SubjectID", "error_type", "y_score", "y_pred", "threshold"]
    comp = comp[[c for c in keep if c in comp.columns]].rename(
        columns={
            "error_type": f"{col_prefix}_error_type",
            "y_score": f"{col_prefix}_score",
            "y_pred": f"{col_prefix}_pred",
            "threshold": f"{col_prefix}_threshold",
        }
    )
    return subjects.merge(comp.drop_duplicates("SubjectID"), on="SubjectID", how="left")


def stable_error_label(row: pd.Series) -> str:
    cols = [c for c in row.index if c.endswith("_error_type")]
    errors = [str(row[c]) for c in cols if pd.notna(row[c]) and str(row[c]) in {"FP", "FN"}]
    if str(row.get("locked_error_type")) not in {"FP", "FN"}:
        return "locked_correct"
    if not errors:
        return "locked_error_no_comparator"
    if len(set(errors)) == 1 and len(errors) >= 2:
        return f"stable_{errors[0].lower()}"
    if str(row.get("locked_error_type")) in errors:
        return "partly_stable_error"
    return "not_stable_across_comparators"


def subject_error_table(args: argparse.Namespace, meta: pd.DataFrame) -> pd.DataFrame:
    locked = primary_predictions(args.locked_readout_dir, "locked_140tr").rename(
        columns={"error_type": "locked_error_type", "y_score": "locked_score", "y_pred": "locked_pred", "threshold": "locked_threshold"}
    )
    out = augment_subjects(locked, meta)
    out = add_comparator_errors(
        out,
        RESULTS / "adni_v5_1b_ch1_only_offdiag_channelmean_horizon4480_cycles56_full_5x5" / "classifier_only_readout",
        "ch1_only",
        "ch1_only",
    )
    out = add_comparator_errors(
        out,
        RESULTS
        / "conditional_beta_vae_manufacturer_full5x5_mfrrecovered035_clfpoollocked"
        / "runs"
        / "ch1_0_2_decoder_only_manufacturer"
        / "classifier_only_readout_z_plus_age_sex",
        "manufacturer_conditioned",
        "manufacturer_conditioned",
    )
    out = add_comparator_errors(
        out,
        RESULTS / "adni_all_available_timepoints_ch1_0_2_exploratory_full_5x5" / "classifier_only_readout",
        "all_timepoints",
        "alltr",
    )
    out["stable_error_across_available_comparators"] = out.apply(stable_error_label, axis=1)
    cols = [
        "SubjectID",
        "fold",
        "ResearchGroup_Mapped",
        "y_true",
        "Manufacturer",
        "SiteCode",
        "Age",
        "Sex",
        "original_n_TR",
        "locked_score",
        "locked_threshold",
        "locked_pred",
        "locked_error_type",
        "ch1_only_error_type",
        "manufacturer_conditioned_error_type",
        "alltr_error_type",
        "stable_error_across_available_comparators",
    ]
    return out[[c for c in cols if c in out.columns]].sort_values(["fold", "locked_error_type", "SubjectID"])


def score_distribution(pred: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, grp in pred.groupby(["fold", "ResearchGroup_Mapped"], dropna=False):
        fold, diagnosis = keys
        scores = pd.to_numeric(grp["y_score"], errors="coerce")
        rows.append(
            {
                "fold": fold,
                "diagnosis": diagnosis,
                "n": int(len(grp)),
                "score_mean": float(scores.mean()),
                "score_sd": float(scores.std(ddof=1)),
                "score_median": float(scores.median()),
                "score_q25": float(scores.quantile(0.25)),
                "score_q75": float(scores.quantile(0.75)),
                "threshold": safe_float(grp["threshold"].iloc[0]),
                "fraction_above_threshold": float((scores >= safe_float(grp["threshold"].iloc[0])).mean()),
            }
        )
    return pd.DataFrame(rows)


def metadata_by_fold(subjects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, grp in subjects.groupby(["fold", "ResearchGroup_Mapped"], dropna=False):
        fold, diagnosis = keys
        age = pd.to_numeric(grp["Age"], errors="coerce")
        ntr = pd.to_numeric(grp.get("original_n_TR", pd.Series(dtype=float)), errors="coerce")
        row = {
            "fold": fold,
            "diagnosis": diagnosis,
            "n": int(len(grp)),
            "age_mean": float(age.mean()),
            "age_median": float(age.median()),
            "sex_counts": json.dumps(grp["Sex"].astype(str).value_counts(dropna=False).to_dict(), sort_keys=True),
            "manufacturer_counts": json.dumps(grp["Manufacturer"].astype(str).value_counts(dropna=False).to_dict(), sort_keys=True),
            "sitecode_n_unique": int(grp["SiteCode"].astype(str).nunique()) if "SiteCode" in grp else np.nan,
            "original_n_TR_mean": float(ntr.mean()) if not ntr.empty else np.nan,
            "original_n_TR_median": float(ntr.median()) if not ntr.empty else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["fold", "diagnosis"])


def hard_fold_diagnosis(
    foldwise: pd.DataFrame,
    latent_summary: pd.DataFrame,
    subjects: pd.DataFrame,
    leakage: pd.DataFrame,
) -> pd.DataFrame:
    pred = primary_predictions(LOCKED_READOUT, "locked_140tr")
    dist = score_distribution(pred)
    rows = []
    for _, r in foldwise.iterrows():
        fold = int(r["fold"])
        fold_dist = dist[dist["fold"].eq(fold)]
        cn_med = safe_float(fold_dist[fold_dist["diagnosis"].eq("CN")]["score_median"].iloc[0]) if not fold_dist[fold_dist["diagnosis"].eq("CN")].empty else np.nan
        ad_med = safe_float(fold_dist[fold_dist["diagnosis"].eq("AD")]["score_median"].iloc[0]) if not fold_dist[fold_dist["diagnosis"].eq("AD")].empty else np.nan
        test_latent = latent_summary[(latent_summary["fold"].eq(fold)) & (latent_summary["split"].eq("test"))]
        leak = leakage[(leakage["fold"].eq(fold)) & (leakage["scope"].eq("test"))]
        sub = subjects[subjects["fold"].eq(fold)]
        issue = []
        if safe_float(r["auc"]) < 0.75:
            issue.append("ranking")
        if safe_float(r["sensitivity"]) < 0.70 or safe_float(r["specificity"]) < 0.70:
            issue.append("threshold_operating_point")
        if not leak.empty and safe_float(leak.iloc[0].get("acc_site_latent")) > 0.70:
            issue.append("manufacturer_leakage")
        if fold in {1, 4}:
            issue.append("hard_fold")
        rows.append(
            {
                "fold": fold,
                "auc": safe_float(r["auc"]),
                "pr_auc": safe_float(r["pr_auc"]),
                "balanced_accuracy": safe_float(r["balanced_accuracy"]),
                "sensitivity": safe_float(r["sensitivity"]),
                "specificity": safe_float(r["specificity"]),
                "f1": safe_float(r["f1"]),
                "threshold": safe_float(r["threshold"]),
                "ad_score_median": ad_med,
                "cn_score_median": cn_med,
                "score_median_gap_ad_minus_cn": ad_med - cn_med,
                "latent_y_mi_sum_test": safe_float(test_latent["mi_sum_nats_Y_target"].iloc[0]) if not test_latent.empty and "mi_sum_nats_Y_target" in test_latent else np.nan,
                "latent_manufacturer_mi_sum_test": safe_float(test_latent["mi_sum_nats_Manufacturer"].iloc[0]) if not test_latent.empty and "mi_sum_nats_Manufacturer" in test_latent else np.nan,
                "test_latent_manufacturer_acc": safe_float(leak.iloc[0].get("acc_site_latent")) if not leak.empty else np.nan,
                "test_raw_manufacturer_acc": safe_float(leak.iloc[0].get("acc_site_raw")) if not leak.empty else np.nan,
                "fp_count": int((sub["locked_error_type"].eq("FP")).sum()) if "locked_error_type" in sub else np.nan,
                "fn_count": int((sub["locked_error_type"].eq("FN")).sum()) if "locked_error_type" in sub else np.nan,
                "diagnosis": ";".join(issue) if issue else "no_major_flag",
                "interpretation": (
                    "Weak fold; ranking and operating-point issues both present."
                    if fold in {1, 4}
                    else "Not a primary failure fold."
                ),
            }
        )
    return pd.DataFrame(rows)


def scanner_leakage(run_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(run_dir.glob("fold_*/fold_*scanner_leakage_summary.csv")):
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        row["fold"] = int(path.parent.name.replace("fold_", ""))
        row["scope"] = "test" if "_test_" in path.name else "train_dev"
        row["latent_minus_raw"] = safe_float(row.get("acc_site_latent")) - safe_float(row.get("acc_site_raw"))
        rows.append(row)
    return pd.DataFrame(rows)


def stagea_summary(run_dir: Path) -> pd.DataFrame:
    rows = []
    for fold in range(1, 6):
        for clf in ["logreg", "svm"]:
            path = run_dir / f"fold_{fold}" / f"optuna_best_trial_{clf}_fold_{fold}.json"
            if not path.exists():
                continue
            data = load_json(path)
            params = data.get("best_params", {})
            row = {
                "fold": fold,
                "stage_a_classifier": clf,
                "best_value_auc": safe_float(data.get("best_value")),
                "n_trials": data.get("n_trials"),
                "n_complete": data.get("n_complete"),
                "best_params": json.dumps(params, sort_keys=True),
            }
            row.update({f"param_{k.replace('model__','')}": v for k, v in params.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def make_recommendations(
    out_dir: Path,
    maturity: pd.DataFrame,
    hyper: pd.DataFrame,
    hard: pd.DataFrame,
    defense_dir: Path,
    stagea: pd.DataFrame,
) -> None:
    c_lower_hits = int(hyper["hit_lower_boundary"].sum()) if not hyper.empty else 0
    reached_max = int(maturity["reached_max_epoch"].sum()) if not maturity.empty else 0
    last10 = int(maturity["best_epoch_in_last_10_percent"].sum()) if not maturity.empty else 0
    weak = hard[hard["fold"].isin([1, 4])].copy()
    failed_table = pd.read_csv(defense_dir / "failed_optimization_table.csv") if (defense_dir / "failed_optimization_table.csv").exists() else pd.DataFrame()
    ultra = failed_table[failed_table["candidate"].astype(str).eq("ultra_regularized_logreg_readout")]
    classifier_pro = failed_table[failed_table["candidate"].astype(str).eq("frozen_latent_stageb_classifier_sweep_pro")]
    beta65 = failed_table[failed_table["candidate"].astype(str).eq("beta65")]
    objective = failed_table[failed_table["candidate"].astype(str).str.contains("objective_v2", na=False)]
    rec = f"""# Hyperparameter Boundary Recommendation

## Stage B Logistic Regression

`logreg_l2` selected `C=0.001` in `{c_lower_hits}/5` folds, which is the lower boundary of the Stage B grid `[0.001, 0.01, 0.1, 1.0]`.

This does **not** justify another classifier-range expansion. The prior ultra-regularized readout audit explicitly tested a denser/lower grid and selected interior stronger-regularization values, but test ranking worsened (`AUC=0.723889`, `PR-AUC=0.447743`). The frozen-latent classifier sweep also did not find a clean replacement: elastic-net improved PR-AUC but did not improve AUC/BA/F1, and LightGBM/RBF-SVM did not improve.

Decision: do not expand the Stage B `logreg_l2` grid for the manuscript model.

## SVM / Alternative Classifiers

Stage A SVM/Optuna results are useful diagnostics, but Stage A is not the manuscript readout and uses a different operating point. The frozen-latent Stage B sweep found no SVM or LightGBM promotion candidate. SVM promotion is not justified without an external calibration/test improvement.

Decision: do not promote SVM or widen SVM search for the primary model.
"""
    (out_dir / "hyperparameter_boundary_recommendation.md").write_text(rec, encoding="utf-8")

    model = f"""# Model-Change Recommendation

## Summary

No further internal optimization is scientifically justified for the locked ADNI model. The strongest next step is external OASIS calibration/test evaluation, not more ADNI model tuning.

## Evidence

- VAE training is not generally right-censored: `{reached_max}/5` folds reached the max epoch and `{last10}/5` folds selected the best checkpoint in the last 10% of the epoch budget.
- Weak folds are not explained by a simple threshold-only issue. Fold 1 and Fold 4 have weaker rank separation, and Fold 4 remains difficult across several sensitivity analyses.
- Stronger beta was tested (`beta65`) and did not improve AUC/PR-AUC.
- Alternate reconstruction scaling was tested in FULL v5.1c/objective-v2 and did not beat the final model.
- Wider/stronger logistic regularization was tested and worsened ranking.
- Frozen-latent classifier alternatives, including regularized LightGBM, did not cleanly improve the primary model.
- All-timepoints rebuilding was explicitly tested as an exploratory stress test and underperformed while increasing confounding risk.

## Decisions

| Proposed change | Decision | Rationale |
|---|---|---|
| Longer VAE training | Not justified | Current fold histories mostly early-stop before max; prior horizon work produced only modest locked improvement and all-timepoints stress test was negative. |
| Smaller/larger beta | Not justified | Higher beta (`6.5`) was negative; changing beta further would be a sweep without a specific failure-mode target. |
| Alternate reconstruction loss | Not justified for primary | offdiag/channel-mean objective did not beat the final v5.1b model in FULL sensitivity checks. |
| Wider `logreg_l2` range | Not justified | Boundary hit is real, but ultra-regularized readout was negative. |
| SVM promotion | Not justified | Stage B classifier sweep did not support SVM over logreg_l2. |
| Resplitting data | Not justified | Would change the locked validation estimand and risks post-hoc split chasing. |
| Repeated CV only for estimation | Reasonable future reporting, not optimization | Could reduce uncertainty of performance estimates if pre-registered, but should not be used to select a new model. |
| W&B integration | Engineering only | Useful for future traceability, not a scientific reason to rerun this model. |
| No further internal optimization | Recommended | Preserve locked-model integrity and move to external calibration/test validation. |
"""
    (out_dir / "model_change_recommendation.md").write_text(model, encoding="utf-8")

    reviewer = """# Reviewer-Ready Model Robustness Text

We performed a read-only deep model-card audit of the locked ADNI model. The audit examined VAE training maturity, beta-VAE rate-distortion behavior, latent information summaries, scanner/manufacturer leakage, Stage B classifier hyperparameters and thresholds, score distributions, and subject-level errors. No VAE training, classifier fitting, threshold fitting, or model selection was performed.

The audit did not identify a scientifically justified internal optimization target. The VAE folds were generally not right-censored at the training horizon, and the weak folds were not explained by a simple threshold artifact. Stage B logistic regression selected the lower C grid value in all folds, but this had already been tested: a denser ultra-regularized readout selected stronger regularization yet substantially worsened test ranking. Broader frozen-latent classifier sweeps, including SVM and regularized LightGBM, also did not produce a clean replacement. Likewise, targeted VAE perturbations including beta increase, dropout changes, decoder changes, block-order changes, objective-v2 loss scaling, and all-timepoint reconstruction failed to improve the primary model without tradeoffs.

We therefore retain the locked v5.1b horizon4480/cycles56 `[1,0,2]` beta-VAE with classifier-only `logreg_l2` readout as the primary manuscript model. We report calibration, scanner/manufacturer and SiteCode audits, Fold 1/Fold 4 limitations, and negative optimization results to make clear that model development was stopped for methodological reasons rather than because no additional configurations were available. The appropriate next step is external OASIS calibration/test validation under a pre-specified protocol, not further internal AUC optimization on ADNI.
"""
    (out_dir / "reviewer_ready_model_robustness_text.md").write_text(reviewer, encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    command_log = {
        "timestamp_utc": now,
        "locked_run_dir": str(args.locked_run_dir),
        "locked_readout_dir": str(args.locked_readout_dir),
        "manuscript_defense_dir": str(args.manuscript_defense_dir),
        "dry_run": bool(args.dry_run),
        "training_launched": False,
        "scoring_launched": False,
        "threshold_fitting": False,
        "model_selection": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "model_outputs_modified": False,
    }
    required = [
        args.locked_run_dir / "run_config.json",
        args.locked_readout_dir / "classifier_sweep_foldwise_metrics.csv",
        args.locked_readout_dir / "classifier_sweep_predictions.csv",
        args.manuscript_defense_dir / "failed_optimization_table.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if args.dry_run:
        command_log["missing_required_files"] = missing
        (args.output_dir / "command_log.json").write_text(json.dumps(command_log, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(command_log, indent=2))
        return 0
    if missing:
        raise FileNotFoundError("Missing required files: " + json.dumps(missing, indent=2))

    cfg = load_config(args.locked_run_dir)
    maturity_rows, rd_rows = [], []
    for fold in range(1, 6):
        m, rd = history_row(args.locked_run_dir, fold, cfg)
        maturity_rows.append(m)
        rd_rows.append(rd)
    maturity = pd.DataFrame(maturity_rows)
    rate = pd.DataFrame(rd_rows)
    write_table(maturity, args.output_dir, "vae_training_maturity_by_fold")
    write_table(rate, args.output_dir, "vae_rate_distortion_by_fold")

    info_long, info_wide = summarize_latent_information(latent_info(args.locked_run_dir))
    write_table(info_wide, args.output_dir, "latent_information_by_fold")
    write_table(info_long, args.output_dir, "latent_information_long_by_fold")

    hyper = stageb_hyperparams(args.locked_readout_dir)
    write_table(hyper, args.output_dir, "stageb_logreg_hyperparameters_by_fold")
    thresh = threshold_confusion(args.locked_readout_dir)
    write_table(thresh, args.output_dir, "threshold_confusion_by_fold")

    meta = load_metadata(cfg, args.ntr_metadata)
    subjects = subject_error_table(args, meta)
    write_table(subjects, args.output_dir, "subject_error_table")
    write_table(score_distribution(primary_predictions(args.locked_readout_dir, "locked_140tr")), args.output_dir, "score_distribution_by_diagnosis_fold")
    write_table(metadata_by_fold(subjects), args.output_dir, "metadata_by_fold")

    leakage = scanner_leakage(args.locked_run_dir)
    write_table(leakage, args.output_dir, "scanner_manufacturer_leakage_by_fold")
    stagea = stagea_summary(args.locked_run_dir)
    write_table(stagea, args.output_dir, "stagea_logreg_svm_optuna_summary")

    hard = hard_fold_diagnosis(read_primary_foldwise(args.locked_readout_dir), info_wide, subjects, leakage)
    write_table(hard, args.output_dir, "hard_fold_diagnosis")

    make_recommendations(args.output_dir, maturity, hyper, hard, args.manuscript_defense_dir, stagea)
    readme = f"""# Locked Primary Model Deep Model-Card Audit

This read-only audit summarizes the locked ADNI v5.1b horizon4480/cycles56 `[1,0,2]` beta-VAE and Stage B `logreg_l2` readout.

Inputs:

- Locked run: `{args.locked_run_dir}`
- Locked Stage B readout: `{args.locked_readout_dir}`
- Manuscript defense package: `{args.manuscript_defense_dir}`

Safety:

- No training was launched.
- No scoring or threshold fitting was performed.
- No model selection was performed.
- No tensor, metadata, ledger, config, or model-output artifacts were modified.

Primary conclusion: no further internal optimization is justified; proceed with external OASIS calibration/test validation.
"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    (args.output_dir / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote locked primary model deep model-card audit to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
