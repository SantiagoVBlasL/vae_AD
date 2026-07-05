#!/usr/bin/env python3
"""Read-only performance-lineage audit for the promoted ADNI model."""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "revision_bspc_2026"
OUT = RESULTS / "promoted_model_performance_lineage_audit_20260601"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
FEATURE = "z_plus_age_sex"


RUN_SPECS: list[dict[str, Any]] = [
    {
        "lineage_order": 1,
        "run_name": "locked_v5_1b_horizon4480_cycles56",
        "run_dir": RESULTS / "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5",
        "config": ROOT / "configs/runs/adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5.json",
        "score_modes": ["raw"],
        "decision": "superseded_reference",
        "change_class": "locked_baseline",
    },
    {
        "lineage_order": 2,
        "run_name": "recover035_latent384_beta2p5_T80_h10000_p560",
        "run_dir": RESULTS / "recover035_latent384_T80_h10000_p560_full5x5",
        "config": ROOT / "configs/runs/adni_v5_1c_recover035_latent384_T80_h10000_p560_full5x5.json",
        "score_modes": ["raw"],
        "decision": "reject",
        "change_class": "vae_capacity_dataset_schedule",
    },
    {
        "lineage_order": 3,
        "run_name": "recover035_latent384_beta3p75_T80_h10000_p560",
        "run_dir": RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5",
        "config": ROOT / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5.json",
        "oof_dir": RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration",
        "score_modes": ["raw", "oof_logitz", "oof_ecdf"],
        "decision": "promote",
        "change_class": "vae_beta_plus_score_harmonization",
    },
    {
        "lineage_order": 4,
        "run_name": "recover035_latent384_beta3p75_drop0p10",
        "run_dir": RESULTS / "recover035_latent384_beta3p75_drop0p10_T80_h10000_p560_full5x5",
        "config": ROOT / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_drop0p10_T80_h10000_p560_full5x5.json",
        "oof_dir": RESULTS / "drop0p10_oof_logitz_readout_audit_20260531",
        "score_modes": ["raw", "oof_logitz", "oof_ecdf"],
        "decision": "reject",
        "change_class": "dropout_global_ablation",
    },
    {
        "lineage_order": 5,
        "run_name": "recover035_latent384_beta3p75_enc015_dec010",
        "run_dir": RESULTS / "recover035_latent384_beta3p75_encdrop0p15_decdrop0p10_T80_h10000_p560_full5x5",
        "config": ROOT / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_encdrop0p15_decdrop0p10_T80_h10000_p560_full5x5.json",
        "score_modes": ["raw"],
        "decision": "reject",
        "change_class": "dropout_scope_ablation",
    },
    {
        "lineage_order": 6,
        "run_name": "recover035_latent128_beta2p5",
        "run_dir": RESULTS / "recover035_latent128_beta2p5_lockedSchedule_full5x5",
        "config": ROOT / "configs/runs/adni_v5_1c_recover035_latent128_beta2p5_lockedSchedule_full5x5.json",
        "oof_dir": RESULTS / "recover035_latent128_beta2p5_lockedSchedule_full5x5_stageB_oof_logitz",
        "score_modes": ["raw", "oof_logitz", "oof_ecdf"],
        "decision": "reject",
        "change_class": "latent128_capacity_control",
    },
    {
        "lineage_order": 7,
        "run_name": "recover035_latent128_beta1p25",
        "run_dir": RESULTS / "recover035_latent128_beta1p25_lockedSchedule_full5x5",
        "config": ROOT / "configs/runs/adni_v5_1c_recover035_latent128_beta1p25_lockedSchedule_full5x5.json",
        "oof_dir": RESULTS / "recover035_latent128_beta1p25_lockedSchedule_full5x5_stageB_oof_logitz",
        "score_modes": ["raw", "oof_logitz", "oof_ecdf"],
        "decision": "reject",
        "change_class": "latent128_beta_scaled",
    },
]


OASIS_MODEL_MAP = {
    ("locked_v5_1b_horizon4480_cycles56", "raw"): "primary_v5_1b_horizon4480_classifier_only",
    ("recover035_latent384_beta3p75_T80_h10000_p560", "raw"): "recover035_raw",
    ("recover035_latent384_beta3p75_T80_h10000_p560", "oof_logitz"): "recover035_oof_logitz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--n-permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame({"read_error": [str(exc)], "source_path": [str(path)]})


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_table(df: pd.DataFrame, csv_path: Path, md_path: Path | None = None) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    if md_path is not None:
        try:
            md = df.to_markdown(index=False)
        except Exception:
            md = df.to_string(index=False)
        md_path.write_text(md + "\n", encoding="utf-8")


def metric_value(row: pd.Series, names: list[str], default: Any = np.nan) -> Any:
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return default


def load_config(spec: dict[str, Any]) -> dict[str, Any]:
    run_cfg = spec["run_dir"] / "run_config.json"
    cfg = read_json(run_cfg)
    if cfg:
        return cfg
    return read_json(spec["config"])


def params_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    if "parameters" in cfg and isinstance(cfg["parameters"], dict):
        return cfg["parameters"]
    if "args" in cfg and isinstance(cfg["args"], dict):
        return cfg["args"]
    return cfg


def paths_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    if "paths" in cfg and isinstance(cfg["paths"], dict):
        return cfg["paths"]
    if "args" in cfg and isinstance(cfg["args"], dict):
        return cfg["args"]
    return {}


def count_metadata(cfg: dict[str, Any]) -> dict[str, Any]:
    meta_path = paths_from_config(cfg).get("metadata_path")
    if not meta_path:
        return {}
    path = Path(meta_path)
    if not path.is_absolute():
        path = ROOT / path
    df = read_csv(path)
    if df.empty:
        return {"metadata_path": str(path), "metadata_rows": np.nan}
    group_col = "ResearchGroup_Mapped" if "ResearchGroup_Mapped" in df.columns else "ResearchGroup"
    counts = df[group_col].astype(str).value_counts(dropna=False).to_dict() if group_col in df.columns else {}
    return {
        "metadata_path": str(path),
        "metadata_rows": int(len(df)),
        "vae_pool_cn": int(counts.get("CN", 0)),
        "vae_pool_mci": int(counts.get("MCI", 0)),
        "vae_pool_ad": int(counts.get("AD", 0)),
        "classifier_pool_cn": int(counts.get("CN", 0)),
        "classifier_pool_ad": int(counts.get("AD", 0)),
    }


def raw_metrics(spec: dict[str, Any]) -> pd.DataFrame:
    path = spec["run_dir"] / "classifier_only_readout" / "classifier_sweep_pooled_metrics.csv"
    df = read_csv(path)
    if df.empty:
        return df
    if "model_name" in df.columns:
        df = df[df["model_name"].astype(str) == "logreg_l2"]
    if "readout_feature_set" in df.columns:
        df = df[df["readout_feature_set"].astype(str) == FEATURE]
    if "threshold_strategy" in df.columns:
        df = df[df["threshold_strategy"].astype(str) == PRIMARY_THRESHOLD]
    df = df.copy()
    df["source_path"] = str(path)
    return df


def oof_metrics(spec: dict[str, Any], method: str) -> pd.DataFrame:
    oof_dir = spec.get("oof_dir")
    if oof_dir is None:
        return pd.DataFrame()
    path = oof_dir / "calib_pooled_metrics.csv"
    df = read_csv(path)
    if df.empty:
        return df
    df = df[
        (df.get("model_name", "").astype(str) == "logreg_l2_original")
        & (df.get("feature_set", "").astype(str) == FEATURE)
        & (df.get("calib_method", "").astype(str) == method)
        & (df.get("threshold_strategy", "").astype(str) == PRIMARY_THRESHOLD)
    ].copy()
    df["source_path"] = str(path)
    return df


def scanner_summary(run_dir: Path) -> dict[str, Any]:
    rows = []
    for fold in range(1, 6):
        for suffix in ["test_scanner_leakage_summary", "scanner_leakage_summary"]:
            p = run_dir / f"fold_{fold}" / f"fold_{fold}_{suffix}.csv"
            df = read_csv(p)
            if not df.empty and {"acc_site_raw", "acc_site_latent"}.issubset(df.columns):
                rows.append(df.assign(fold=fold, source_path=str(p)))
                break
    if not rows:
        return {}
    df = pd.concat(rows, ignore_index=True)
    raw = pd.to_numeric(df["acc_site_raw"], errors="coerce")
    lat = pd.to_numeric(df["acc_site_latent"], errors="coerce")
    return {
        "scanner_raw_ba": float(raw.mean()),
        "scanner_latent_ba": float(lat.mean()),
        "scanner_latent_minus_raw": float((lat - raw).mean()),
    }


def philips_fp_raw(spec: dict[str, Any]) -> dict[str, Any]:
    path = spec["run_dir"] / "classifier_only_readout" / "classifier_sweep_subgroup_metrics_by_manufacturer.csv"
    df = read_csv(path)
    if df.empty:
        return {}
    if "model_name" in df.columns:
        df = df[df["model_name"].astype(str) == "logreg_l2"]
    if "readout_feature_set" in df.columns:
        df = df[df["readout_feature_set"].astype(str) == FEATURE]
    if "threshold_strategy" in df.columns:
        df = df[df["threshold_strategy"].astype(str) == PRIMARY_THRESHOLD]
    manu = "Manufacturer" if "Manufacturer" in df.columns else "manufacturer"
    if manu not in df.columns:
        return {}
    ph = df[df[manu].astype(str).str.lower().eq("philips")]
    if ph.empty:
        return {}
    n_cn = pd.to_numeric(ph.get("n_cn"), errors="coerce").sum()
    fp = pd.to_numeric(ph.get("fp"), errors="coerce").sum()
    return {"philips_cn_n": n_cn, "philips_cn_fp": fp, "philips_cn_fpr": fp / n_cn if n_cn else np.nan}


def philips_fp_oof(spec: dict[str, Any], method: str) -> dict[str, Any]:
    oof_dir = spec.get("oof_dir")
    if oof_dir is None:
        return {}
    path = oof_dir / "calib_philips_fpr_pooled.csv"
    df = read_csv(path)
    if df.empty:
        return {}
    df = df[
        (df.get("model_name", "").astype(str) == "logreg_l2_original")
        & (df.get("feature_set", "").astype(str) == FEATURE)
        & (df.get("calib_method", "").astype(str) == method)
        & (df.get("threshold_strategy", "").astype(str) == PRIMARY_THRESHOLD)
    ].copy()
    manu = "manufacturer" if "manufacturer" in df.columns else "Manufacturer"
    if manu in df.columns:
        df = df[df[manu].astype(str).str.lower().eq("philips")]
    if df.empty:
        return {}
    r = df.iloc[0]
    return {
        "philips_cn_n": r.get("n_cn_pooled", r.get("n_cn", np.nan)),
        "philips_cn_fp": r.get("fp_cn_pooled", r.get("fp", np.nan)),
        "philips_cn_fpr": r.get("fpr_cn_pooled", np.nan),
    }


def oasis_mega_result(run_name: str, score_mode: str) -> dict[str, Any]:
    adni_model = OASIS_MODEL_MAP.get((run_name, score_mode))
    if not adni_model:
        return {}
    path = RESULTS / "oasis_mega_90cn_90ad_pooled_external_validation_20260531" / "pooled_ranking_metrics.csv"
    df = read_csv(path)
    if df.empty or "adni_model" not in df.columns:
        return {}
    sub = df[df["adni_model"].astype(str) == adni_model].copy()
    if sub.empty:
        return {}
    sub["auc"] = pd.to_numeric(sub["auc"], errors="coerce")
    sub = sub.sort_values(["auc", "pr_auc"], ascending=False)
    r = sub.iloc[0]
    return {
        "oasis_mega_model": adni_model,
        "oasis_mega_best_build": r.get("build_candidate"),
        "oasis_mega_auc": r.get("auc"),
        "oasis_mega_pr_auc": r.get("pr_auc"),
        "oasis_mega_auc_ci": f"{r.get('auc_bootstrap95_lo', np.nan)}-{r.get('auc_bootstrap95_hi', np.nan)}",
    }


def metric_record(spec: dict[str, Any], score_mode: str, row: pd.Series, cfg: dict[str, Any]) -> dict[str, Any]:
    p = params_from_config(cfg)
    meta_counts = count_metadata(cfg)
    leakage = scanner_summary(spec["run_dir"])
    philips = philips_fp_raw(spec) if score_mode == "raw" else philips_fp_oof(spec, score_mode)
    oasis = oasis_mega_result(spec["run_name"], score_mode)
    channel_indices = p.get("channels_to_use", np.nan)
    channel_names = cfg.get("selected_channel_names", p.get("selected_channel_names", np.nan))
    master_names = cfg.get("channel_names_master_in_tensor_order", cfg.get("DEFAULT_CHANNEL_NAMES", p.get("all_original_channel_names", [])))
    if (isinstance(channel_names, float) and math.isnan(channel_names)) or channel_names is np.nan:
        if isinstance(channel_indices, list) and isinstance(master_names, list):
            channel_names = [master_names[i] for i in channel_indices if isinstance(i, int) and i < len(master_names)]
    return {
        "lineage_order": spec["lineage_order"],
        "run_name": spec["run_name"],
        "change_class": spec["change_class"],
        "score_mode": score_mode,
        "subject_pool_cn": meta_counts.get("vae_pool_cn", np.nan),
        "subject_pool_mci": meta_counts.get("vae_pool_mci", np.nan),
        "subject_pool_ad": meta_counts.get("vae_pool_ad", np.nan),
        "classifier_pool_cn": metric_value(row, ["n_cn"], meta_counts.get("classifier_pool_cn", np.nan)),
        "classifier_pool_ad": metric_value(row, ["n_ad"], meta_counts.get("classifier_pool_ad", np.nan)),
        "channels": channel_indices,
        "selected_channel_names": channel_names,
        "latent_dim": p.get("latent_dim", np.nan),
        "beta_vae": p.get("beta_vae", np.nan),
        "dropout_rate_vae": p.get("dropout_rate_vae", np.nan),
        "encoder_dropout_rate_vae": p.get("encoder_dropout_rate_vae", np.nan),
        "decoder_dropout_rate_vae": p.get("decoder_dropout_rate_vae", np.nan),
        "vae_dropout_scope": p.get("vae_dropout_scope", np.nan),
        "epochs_vae": p.get("epochs_vae", np.nan),
        "cycles": p.get("cyclical_beta_n_cycles", np.nan),
        "lr_scheduler_T0": p.get("lr_scheduler_T0", np.nan),
        "early_stopping_patience_vae": p.get("early_stopping_patience_vae", np.nan),
        "final_activation": p.get("vae_final_activation", p.get("final_activation", np.nan)),
        "readout_type": "StageB logreg_l2 z_plus_age_sex",
        "score_harmonization_type": score_mode,
        "auc": metric_value(row, ["auc", "AUC"]),
        "pr_auc": metric_value(row, ["pr_auc", "PR-AUC"]),
        "balanced_accuracy": metric_value(row, ["balanced_accuracy", "BA"]),
        "sensitivity": metric_value(row, ["sensitivity", "Sens"]),
        "specificity": metric_value(row, ["specificity", "Spec"]),
        "f1": metric_value(row, ["f1", "F1"]),
        "tn": metric_value(row, ["tn", "TN"]),
        "fp": metric_value(row, ["fp", "FP"]),
        "fn": metric_value(row, ["fn", "FN"]),
        "tp": metric_value(row, ["tp", "TP"]),
        "scanner_raw_ba": leakage.get("scanner_raw_ba", np.nan),
        "scanner_latent_ba": leakage.get("scanner_latent_ba", np.nan),
        "scanner_latent_minus_raw": leakage.get("scanner_latent_minus_raw", np.nan),
        "philips_cn_n": philips.get("philips_cn_n", np.nan),
        "philips_cn_fp": philips.get("philips_cn_fp", np.nan),
        "philips_cn_fpr": philips.get("philips_cn_fpr", np.nan),
        "oasis_mega_model": oasis.get("oasis_mega_model", np.nan),
        "oasis_mega_best_build": oasis.get("oasis_mega_best_build", np.nan),
        "oasis_mega_auc": oasis.get("oasis_mega_auc", np.nan),
        "oasis_mega_pr_auc": oasis.get("oasis_mega_pr_auc", np.nan),
        "oasis_mega_auc_ci": oasis.get("oasis_mega_auc_ci", np.nan),
        "decision": spec["decision"],
        "source_path": row.get("source_path", np.nan),
    }


def build_lineage() -> pd.DataFrame:
    records = []
    for spec in RUN_SPECS:
        cfg = load_config(spec)
        for mode in spec["score_modes"]:
            df = raw_metrics(spec) if mode == "raw" else oof_metrics(spec, mode)
            if df.empty:
                records.append(
                    {
                        "lineage_order": spec["lineage_order"],
                        "run_name": spec["run_name"],
                        "score_mode": mode,
                        "decision": spec["decision"],
                        "source_status": "missing_metrics",
                    }
                )
                continue
            for _, row in df.iterrows():
                rec = metric_record(spec, mode, row, cfg)
                rec["source_status"] = "ok"
                records.append(rec)
    out = pd.DataFrame(records)
    if not out.empty:
        score_order = {"raw": 0, "oof_logitz": 1, "oof_ecdf": 2}
        out["_score_order"] = out["score_mode"].map(score_order).fillna(99)
        out = out.sort_values(["lineage_order", "run_name", "_score_order", "score_mode"]).drop(columns=["_score_order"])
    return out


def load_predictions(run_name: str, score_mode: str) -> pd.DataFrame:
    spec = next(s for s in RUN_SPECS if s["run_name"] == run_name)
    if score_mode == "raw":
        path = spec["run_dir"] / "classifier_only_readout" / "classifier_sweep_predictions.csv"
        df = read_csv(path)
        if df.empty:
            return df
        df = df[(df.get("model_name", "").astype(str) == "logreg_l2") & (df.get("threshold_strategy", "").astype(str) == PRIMARY_THRESHOLD)].copy()
    else:
        path = spec["oof_dir"] / "calib_predictions.csv"
        df = read_csv(path)
        if df.empty:
            return df
        df = df[
            (df.get("model_name", "").astype(str) == "logreg_l2_original")
            & (df.get("feature_set", "").astype(str) == FEATURE)
            & (df.get("calib_method", "").astype(str) == score_mode)
            & (df.get("threshold_strategy", "").astype(str) == PRIMARY_THRESHOLD)
        ].copy()
    df["source_run"] = run_name
    df["score_mode"] = score_mode
    df["source_path"] = str(path)
    return df


def paired_prediction_frame(candidate_mode: str) -> pd.DataFrame:
    locked = load_predictions("locked_v5_1b_horizon4480_cycles56", "raw")
    cand = load_predictions("recover035_latent384_beta3p75_T80_h10000_p560", candidate_mode)
    cols = ["SubjectID", "fold", "y_true", "y_score"]
    locked = locked[[c for c in cols if c in locked.columns]].rename(columns={"fold": "fold_locked", "y_true": "y_true_locked", "y_score": "score_locked"})
    cand = cand[[c for c in cols if c in cand.columns]].rename(columns={"fold": "fold_candidate", "y_true": "y_true_candidate", "y_score": "score_candidate"})
    merged = locked.merge(cand, on="SubjectID", how="inner")
    merged = merged[merged["y_true_locked"].astype(int) == merged["y_true_candidate"].astype(int)].copy()
    merged["y_true"] = merged["y_true_locked"].astype(int)
    return merged


def bootstrap_and_permutation(candidate_mode: str, n_boot: int, n_perm: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    paired = paired_prediction_frame(candidate_mode)
    if paired.empty:
        return pd.DataFrame(), pd.DataFrame()
    rng = np.random.default_rng(seed)
    y = paired["y_true"].to_numpy()
    s0 = pd.to_numeric(paired["score_locked"], errors="coerce").to_numpy()
    s1 = pd.to_numeric(paired["score_candidate"], errors="coerce").to_numpy()
    keep = np.isfinite(y) & np.isfinite(s0) & np.isfinite(s1)
    y, s0, s1 = y[keep], s0[keep], s1[keep]
    n = len(y)

    def metrics(scores: np.ndarray) -> tuple[float, float]:
        return float(roc_auc_score(y, scores)), float(average_precision_score(y, scores))

    auc0, ap0 = metrics(s0)
    auc1, ap1 = metrics(s1)
    boot_auc = []
    boot_ap = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        boot_auc.append(float(roc_auc_score(yy, s1[idx]) - roc_auc_score(yy, s0[idx])))
        boot_ap.append(float(average_precision_score(yy, s1[idx]) - average_precision_score(yy, s0[idx])))

    obs_auc = auc1 - auc0
    obs_ap = ap1 - ap0
    perm_auc = []
    perm_ap = []
    for _ in range(n_perm):
        swap = rng.random(n) < 0.5
        a = np.where(swap, s1, s0)
        b = np.where(swap, s0, s1)
        perm_auc.append(float(roc_auc_score(y, b) - roc_auc_score(y, a)))
        perm_ap.append(float(average_precision_score(y, b) - average_precision_score(y, a)))

    boot_df = pd.DataFrame(
        [
            {
                "candidate_score_mode": candidate_mode,
                "comparison": "candidate_minus_locked_raw_overlap_subjects",
                "n_overlap": n,
                "locked_auc": auc0,
                "candidate_auc": auc1,
                "auc_delta": obs_auc,
                "auc_delta_bootstrap95_lo": float(np.percentile(boot_auc, 2.5)) if boot_auc else np.nan,
                "auc_delta_bootstrap95_hi": float(np.percentile(boot_auc, 97.5)) if boot_auc else np.nan,
                "locked_pr_auc": ap0,
                "candidate_pr_auc": ap1,
                "pr_auc_delta": obs_ap,
                "pr_auc_delta_bootstrap95_lo": float(np.percentile(boot_ap, 2.5)) if boot_ap else np.nan,
                "pr_auc_delta_bootstrap95_hi": float(np.percentile(boot_ap, 97.5)) if boot_ap else np.nan,
                "n_bootstrap": n_boot,
            }
        ]
    )
    perm_df = pd.DataFrame(
        [
            {
                "candidate_score_mode": candidate_mode,
                "metric": "auc",
                "observed_delta": obs_auc,
                "paired_permutation_two_sided_p": float((np.sum(np.abs(perm_auc) >= abs(obs_auc)) + 1) / (len(perm_auc) + 1)),
                "n_permutations": n_perm,
                "test_note": "Subject-level paired score-swap permutation; DeLong not used.",
            },
            {
                "candidate_score_mode": candidate_mode,
                "metric": "pr_auc",
                "observed_delta": obs_ap,
                "paired_permutation_two_sided_p": float((np.sum(np.abs(perm_ap) >= abs(obs_ap)) + 1) / (len(perm_ap) + 1)),
                "n_permutations": n_perm,
                "test_note": "Subject-level paired score-swap permutation; DeLong not used.",
            },
        ]
    )
    return boot_df, perm_df


def foldwise_metrics(run_name: str, score_mode: str) -> pd.DataFrame:
    spec = next(s for s in RUN_SPECS if s["run_name"] == run_name)
    if score_mode == "raw":
        path = spec["run_dir"] / "classifier_only_readout" / "classifier_sweep_foldwise_metrics.csv"
        df = read_csv(path)
        if df.empty:
            return df
        df = df[(df.get("model_name", "").astype(str) == "logreg_l2") & (df.get("threshold_strategy", "").astype(str) == PRIMARY_THRESHOLD)].copy()
    else:
        path = spec["oof_dir"] / "calib_foldwise_metrics.csv"
        df = read_csv(path)
        if df.empty:
            return df
        df = df[
            (df.get("model_name", "").astype(str) == "logreg_l2_original")
            & (df.get("feature_set", "").astype(str) == FEATURE)
            & (df.get("calib_method", "").astype(str) == score_mode)
            & (df.get("threshold_strategy", "").astype(str) == PRIMARY_THRESHOLD)
        ].copy()
    df["run_name"] = run_name
    df["score_mode"] = score_mode
    df["source_path"] = str(path)
    return df


def foldwise_delta_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    locked = foldwise_metrics("locked_v5_1b_horizon4480_cycles56", "raw")
    rows = []
    for mode in ["raw", "oof_logitz", "oof_ecdf"]:
        cand = foldwise_metrics("recover035_latent384_beta3p75_T80_h10000_p560", mode)
        if locked.empty or cand.empty:
            continue
        merged = locked.merge(cand, on="fold", suffixes=("_locked", "_candidate"))
        for _, r in merged.iterrows():
            rec = {"fold": r["fold"], "candidate_score_mode": mode}
            for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
                rec[f"locked_{metric}"] = r.get(f"{metric}_locked", np.nan)
                rec[f"candidate_{metric}"] = r.get(f"{metric}_candidate", np.nan)
                rec[f"delta_{metric}"] = pd.to_numeric(pd.Series([r.get(f"{metric}_candidate", np.nan)]), errors="coerce").iloc[0] - pd.to_numeric(pd.Series([r.get(f"{metric}_locked", np.nan)]), errors="coerce").iloc[0]
            rows.append(rec)
    fold_delta = pd.DataFrame(rows)
    test_rows = []
    for mode, grp in fold_delta.groupby("candidate_score_mode") if not fold_delta.empty else []:
        for metric in ["auc", "pr_auc", "balanced_accuracy", "f1"]:
            d = pd.to_numeric(grp[f"delta_{metric}"], errors="coerce").dropna().to_numpy()
            if len(d) == 0:
                continue
            rec = {
                "candidate_score_mode": mode,
                "metric": metric,
                "n_folds": len(d),
                "mean_delta": float(np.mean(d)),
                "median_delta": float(np.median(d)),
                "n_positive_folds": int(np.sum(d > 0)),
                "screening_note": "n=5 foldwise screening only; folds are not independent enough for confirmatory inference.",
            }
            try:
                from scipy import stats

                rec["paired_t_p"] = float(stats.ttest_1samp(d, 0.0).pvalue) if len(d) >= 2 else np.nan
                rec["wilcoxon_p"] = float(stats.wilcoxon(d, zero_method="wilcox").pvalue) if len(d) >= 2 and np.any(d != 0) else np.nan
            except Exception:
                rec["paired_t_p"] = np.nan
                rec["wilcoxon_p"] = np.nan
            test_rows.append(rec)
    return fold_delta, pd.DataFrame(test_rows)


def build_attribution(lineage: pd.DataFrame) -> pd.DataFrame:
    def pick(run: str, mode: str) -> pd.Series:
        sub = lineage[(lineage["run_name"] == run) & (lineage["score_mode"] == mode)]
        return sub.iloc[0] if not sub.empty else pd.Series(dtype=object)

    comparisons = [
        ("locked -> latent384_beta2p5_raw", pick("locked_v5_1b_horizon4480_cycles56", "raw"), pick("recover035_latent384_beta2p5_T80_h10000_p560", "raw"), "Dataset recovery + latent_dim 384 + longer schedule at beta2.5; VAE/readout raw."),
        ("latent384_beta2p5_raw -> beta3p75_raw", pick("recover035_latent384_beta2p5_T80_h10000_p560", "raw"), pick("recover035_latent384_beta3p75_T80_h10000_p560", "raw"), "VAE beta 2.5 -> 3.75, raw Stage B."),
        ("beta3p75_raw -> beta3p75_oof_logitz", pick("recover035_latent384_beta3p75_T80_h10000_p560", "raw"), pick("recover035_latent384_beta3p75_T80_h10000_p560", "oof_logitz"), "Readout score harmonization only, OOF-logitz."),
        ("beta3p75_raw -> beta3p75_oof_ecdf", pick("recover035_latent384_beta3p75_T80_h10000_p560", "raw"), pick("recover035_latent384_beta3p75_T80_h10000_p560", "oof_ecdf"), "Readout score harmonization only, OOF-ECDF."),
        ("locked -> beta3p75_oof_logitz", pick("locked_v5_1b_horizon4480_cycles56", "raw"), pick("recover035_latent384_beta3p75_T80_h10000_p560", "oof_logitz"), "Total promoted lineage effect using requested OOF-logitz reference."),
        ("locked -> beta3p75_oof_ecdf", pick("locked_v5_1b_horizon4480_cycles56", "raw"), pick("recover035_latent384_beta3p75_T80_h10000_p560", "oof_ecdf"), "Total promoted lineage effect using highest parsed score-harmonized row."),
    ]
    rows = []
    for label, a, b, note in comparisons:
        rec = {"comparison": label, "interpretation": note}
        for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
            rec[f"from_{metric}"] = a.get(metric, np.nan)
            rec[f"to_{metric}"] = b.get(metric, np.nan)
            rec[f"delta_{metric}"] = pd.to_numeric(pd.Series([b.get(metric, np.nan)]), errors="coerce").iloc[0] - pd.to_numeric(pd.Series([a.get(metric, np.nan)]), errors="coerce").iloc[0]
        rows.append(rec)
    return pd.DataFrame(rows)


def channel_variant_inventory() -> pd.DataFrame:
    paths = [
        RESULTS / "channel_pair_ablation_fast3x3_offdiag_channelmean_ch1_anchor" / "primary_pair_table.csv",
        RESULTS / "channel_ablation_fast3x3_offdiag_channelmean" / "primary_ablation_table.csv",
    ]
    rows = []
    for path in paths:
        df = read_csv(path)
        if df.empty:
            continue
        for _, r in df.iterrows():
            channels = str(r.get("channels", ""))
            if channels in {"[1, 0]", "[1,0]", "[1, 0, 4]", "[1,0,4]"}:
                rec = r.to_dict()
                rec["source_path"] = str(path)
                rec["availability_note"] = "FAST 3x3 only; not a promoted/full lineage run."
                rows.append(rec)
    if not any(str(r.get("channels", "")) in {"[1, 0, 4]", "[1,0,4]"} for r in rows):
        rows.append({"run_key": "ch1_0_4", "channels": "[1,0,4]", "availability_note": "No matching completed variant found under revision_bspc_2026."})
    return pd.DataFrame(rows)


def write_readme(out_dir: Path, lineage: pd.DataFrame, boot: pd.DataFrame) -> None:
    prom = lineage[(lineage["run_name"] == "recover035_latent384_beta3p75_T80_h10000_p560") & (lineage["score_mode"] == "oof_logitz")]
    r = prom.iloc[0].to_dict() if not prom.empty else {}
    text = f"""# Promoted Model Performance Lineage Audit

Generated: {now_iso()}

This package is read-only. It compares the promoted ADNI model against the
locked v5.1b horizon4480 classifier-only readout and major completed negative
or sensitivity branches.

## Promoted Reference

Requested reference: `recover035_latent384_beta3p75_T80_h10000_p560_full5x5`
with p=0.15 `legacy_all` dropout and OOF-logitz score harmonization.

- AUC: {r.get('auc', np.nan)}
- PR-AUC: {r.get('pr_auc', np.nan)}
- BA: {r.get('balanced_accuracy', np.nan)}
- Sensitivity: {r.get('sensitivity', np.nan)}
- Specificity: {r.get('specificity', np.nan)}
- F1: {r.get('f1', np.nan)}

OOF-ECDF is also reported because it is marginally higher internally, but the
OASIS mega audit contains the OOF-logitz scoring path.

## Robustness

Bootstrap and paired permutation tests are computed on overlapping subjects only
when comparing against the locked v5.1b run, because the promoted all-eligible
branch includes recovered `035_S_6927`.

## Guardrails

- No training was launched.
- No model selection was performed.
- No tensor, metadata, or model artifact was modified.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir: Path = args.output_dir
    if args.dry_run:
        print(f"Would write lineage audit to {out_dir}")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)

    command_log = {
        "script": str(Path(__file__).resolve()),
        "started_at": now_iso(),
        "safety": {
            "training_launched": False,
            "model_selection_performed": False,
            "tensor_modified": False,
            "metadata_modified": False,
            "model_artifacts_modified": False,
        },
        "n_bootstrap": args.n_bootstrap,
        "n_permutations": args.n_permutations,
        "seed": args.seed,
    }

    lineage = build_lineage()
    write_table(lineage, out_dir / "chronological_lineage_table.csv", out_dir / "chronological_lineage_table.md")

    locked = lineage[(lineage["run_name"] == "locked_v5_1b_horizon4480_cycles56") & (lineage["score_mode"] == "raw")]
    delta_rows = []
    if not locked.empty:
        base = locked.iloc[0]
        for _, r in lineage.iterrows():
            rec = {"run_name": r.get("run_name"), "score_mode": r.get("score_mode"), "decision": r.get("decision")}
            for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]:
                rec[f"{metric}"] = r.get(metric)
                rec[f"delta_vs_locked_{metric}"] = pd.to_numeric(pd.Series([r.get(metric, np.nan)]), errors="coerce").iloc[0] - pd.to_numeric(pd.Series([base.get(metric, np.nan)]), errors="coerce").iloc[0]
            delta_rows.append(rec)
    deltas = pd.DataFrame(delta_rows)
    write_table(deltas, out_dir / "lineage_metric_deltas_vs_locked.csv", out_dir / "lineage_metric_deltas_vs_locked.md")

    attribution = build_attribution(lineage)
    write_table(attribution, out_dir / "improvement_attribution.csv", out_dir / "improvement_attribution.md")

    fold_delta, fold_tests = foldwise_delta_table()
    write_table(fold_delta, out_dir / "paired_foldwise_deltas.csv", out_dir / "paired_foldwise_deltas.md")
    write_table(fold_tests, out_dir / "paired_foldwise_tests.csv", out_dir / "paired_foldwise_tests.md")
    fold45 = fold_delta[fold_delta["fold"].isin([4, 5])] if not fold_delta.empty else pd.DataFrame()
    write_table(fold45, out_dir / "fold4_fold5_deltas.csv", out_dir / "fold4_fold5_deltas.md")

    boot_rows = []
    perm_rows = []
    for mode in ["raw", "oof_logitz", "oof_ecdf"]:
        boot, perm = bootstrap_and_permutation(mode, args.n_bootstrap, args.n_permutations, args.seed)
        if not boot.empty:
            boot_rows.append(boot)
        if not perm.empty:
            perm_rows.append(perm)
    boot_df = pd.concat(boot_rows, ignore_index=True) if boot_rows else pd.DataFrame()
    perm_df = pd.concat(perm_rows, ignore_index=True) if perm_rows else pd.DataFrame()
    write_table(boot_df, out_dir / "bootstrap_auc_pr_delta.csv", out_dir / "bootstrap_auc_pr_delta.md")
    write_table(perm_df, out_dir / "paired_permutation_tests.csv", out_dir / "paired_permutation_tests.md")

    channels = channel_variant_inventory()
    write_table(channels, out_dir / "channel_variant_inventory.csv", out_dir / "channel_variant_inventory.md")

    recommendation = [
        "# Next Optimization Recommendation",
        "",
        "Recommendation: no additional internal ADNI optimization target is scientifically justified from this lineage audit.",
        "",
        "Rationale:",
        "- The largest internal ranking gain coincides with score harmonization rather than a clean raw-VAE improvement over the locked baseline.",
        "- Latent384 beta3.75 OOF-logitz/OOF-ECDF improves AUC and PR-AUC, but BA/F1 are not uniformly superior to the locked thresholded readout.",
        "- Dropout, latent128, and channel-pair variants do not provide a cleaner improvement profile.",
        "- OASIS mega and pilot-vs-new analyses remain external stress tests and should not drive ADNI model selection.",
        "",
        "Scientifically justified next work:",
        "- Freeze the promoted ADNI model for manuscript reporting.",
        "- Continue OASIS calibration/test and interpretability reporting as validation/analysis, not internal optimization.",
        "- If additional modeling is required later, require a pre-registered external-validation target rather than another internal ADNI sweep.",
    ]
    (out_dir / "next_optimization_recommendation.md").write_text("\n".join(recommendation) + "\n", encoding="utf-8")

    write_readme(out_dir, lineage, boot_df)
    command_log["finished_at"] = now_iso()
    command_log["outputs"] = sorted(str(p.relative_to(ROOT)) for p in out_dir.glob("*"))
    (out_dir / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote performance-lineage audit to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
