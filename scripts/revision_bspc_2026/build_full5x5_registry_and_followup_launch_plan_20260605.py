#!/usr/bin/env python3
"""Build a read-only FULL 5x5 registry and prepare follow-up launch commands.

This script reads completed ADNI FULL 5x5 artifacts under results/revision_bspc_2026
and writes a registry plus launch-plan package. It does not launch training,
score OASIS, modify tensors, modify metadata, or modify model artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/revision_bspc_2026"
CONFIGS = ROOT / "configs/runs"
OUT = RESULTS / "full5x5_completed_run_registry_followup_plan_20260605"
PYTHON = "/home/diego/anaconda3/envs/vae_ad/bin/python"
TRAINING_SCRIPT = ROOT / "scripts/run_vae_clf_ad_inference.py"
STAGE_B_SCRIPT = ROOT / "scripts/revision_bspc_2026/run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep.py"
OOF_SCRIPT = ROOT / "scripts/revision_bspc_2026/run_recover035_latent384_beta3p75_stageB_oof_score_calibration.py"
BASE_CONFIG = CONFIGS / "adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5.json"

PRIMARY_FEATURE_SET = "z_plus_age_sex"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
PRIMARY_RAW_MODEL = "logreg_l2"
PRIMARY_OOF_MODEL = "logreg_l2_original"
CALIB_METHODS = ["raw", "oof_zscore", "oof_logitz", "oof_ecdf", "oof_platt", "oof_isotonic"]
DATE_TAG = "20260605"

PROMOTED_REF = {
    "run_label": "promoted_ch1_0_2_latent384_beta3p75",
    "run_name": "recover035_latent384_beta3p75_T80_h10000_p560_full5x5",
    "oof_ecdf_auc": 0.795155,
    "oof_ecdf_pr_auc": 0.573934,
    "oof_ecdf_ba": 0.725979,
    "oof_ecdf_sens": 0.731959,
    "oof_ecdf_spec": 0.720000,
    "oof_ecdf_f1": 0.563492,
    "oof_ecdf_philips_cn_fpr": 0.454545,
}
CH1_REF = {
    "run_label": "ch1only_latent384_beta3p75",
    "run_name": "recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5",
    "oof_ecdf_auc": 0.800378,
    "oof_ecdf_pr_auc": 0.585842,
    "oof_logitz_auc": 0.800344,
    "oof_logitz_pr_auc": 0.589002,
    "oof_ecdf_ba": 0.721134,
    "oof_ecdf_sens": 0.742268,
    "oof_ecdf_spec": 0.700000,
    "oof_ecdf_f1": 0.555985,
    "oof_ecdf_philips_cn_fpr": 0.464646,
}

CHANNEL_NAMES = [
    "Pearson_OMST_GCE_Signed_Weighted",
    "Pearson_Full_FisherZ_Signed",
    "MI_KNN_Symmetric",
    "dFC_AbsDiffMean",
    "dFC_StdDev",
    "DistanceCorr",
    "Granger_F_lag1",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output-dir", type=Path, default=OUT)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def safe_float(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def safe_div(num: float, den: float) -> float:
    if den == 0 or math.isnan(num) or math.isnan(den):
        return float("nan")
    return num / den


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def to_md(df: pd.DataFrame, max_rows: int = 120) -> str:
    if df.empty:
        return "_No rows available._\n"
    view = df.head(max_rows).copy()
    try:
        text = view.to_markdown(index=False)
    except Exception:
        text = view.to_string(index=False)
    if len(df) > max_rows:
        text += f"\n\n_Showing {max_rows} of {len(df)} rows._"
    return text + "\n"


def write_table(df: pd.DataFrame, out_dir: Path, stem: str, max_rows: int = 120) -> None:
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    (out_dir / f"{stem}.md").write_text(to_md(df, max_rows=max_rows), encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def discover_run_dirs() -> list[Path]:
    roots: set[Path] = set()
    for entry in RESULTS.iterdir():
        try:
            if entry.is_dir() and all((entry / f"fold_{i}").is_dir() for i in range(1, 6)):
                roots.add(entry)
        except OSError:
            continue
    for fold5 in RESULTS.glob("**/fold_5"):
        try:
            if fold5.is_dir():
                roots.add(fold5.parent)
        except OSError:
            continue
    out = []
    for root in sorted(roots):
        if all((root / f"fold_{i}").is_dir() for i in range(1, 6)):
            out.append(root)
    return out


def load_config_index() -> dict[str, dict[str, Any]]:
    idx: dict[str, dict[str, Any]] = {}
    for path in sorted(CONFIGS.glob("*.json")):
        cfg = load_json(path)
        if not cfg:
            continue
        run_name = str(cfg.get("run_name") or path.stem)
        idx[run_name] = {"config_path": path, "config": cfg}
        output_dir = cfg.get("paths", {}).get("output_dir")
        if output_dir:
            idx[Path(str(output_dir)).name] = {"config_path": path, "config": cfg}
    return idx


def run_name_from_dir(run_dir: Path) -> str:
    manifest = load_json(run_dir / "run_manifest.json")
    if manifest.get("run_name"):
        return str(manifest["run_name"])
    command_log = load_json(run_dir / "command_log.json")
    if command_log.get("run_name"):
        return str(command_log["run_name"])
    cfg = load_json(run_dir / "run_config.json")
    if cfg.get("args", {}).get("output_dir"):
        return Path(str(cfg["args"]["output_dir"])).name
    return run_dir.name


def args_from_run(run_dir: Path, cfg_index: dict[str, dict[str, Any]]) -> tuple[str, dict[str, Any], Path | None]:
    run_name = run_name_from_dir(run_dir)
    run_cfg = load_json(run_dir / "run_config.json")
    if run_cfg.get("args"):
        return run_name, dict(run_cfg["args"]), run_dir / "run_config.json"
    item = cfg_index.get(run_name) or cfg_index.get(run_dir.name)
    if item:
        return run_name, dict(item["config"].get("parameters", {})) | {
            "global_tensor_path": item["config"].get("paths", {}).get("global_tensor_path"),
            "metadata_path": item["config"].get("paths", {}).get("metadata_path"),
        }, item["config_path"]
    return run_name, {}, None


def oof_map() -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for pooled in RESULTS.glob("*/calib_pooled_metrics.csv"):
        out_dir = pooled.parent
        log = load_json(out_dir / "command_log.json")
        run_name = log.get("run_name")
        if run_name:
            mapping[str(run_name)] = out_dir
        # Fallback for exact prefix naming.
        name = out_dir.name
        for suffix in ["_stageB_oof_score_calibration", "_stageB_oof_logitz"]:
            if name.endswith(suffix):
                mapping.setdefault(name[: -len(suffix)], out_dir)
    return mapping


def selected_channel_names(channels: Any) -> list[str]:
    if not isinstance(channels, list):
        return []
    names = []
    for idx in channels:
        try:
            names.append(CHANNEL_NAMES[int(idx)])
        except Exception:
            names.append(str(idx))
    return names


def stagea_summary(run_dir: Path) -> dict[str, Any]:
    matches = sorted(run_dir.glob("all_folds_metrics_MULTI*.csv"))
    out: dict[str, Any] = {}
    if not matches:
        return out
    df = pd.read_csv(matches[0])
    for model in ["logreg", "svm"]:
        rows = df[df.get("actual_classifier_type", pd.Series(dtype=str)).astype(str).eq(model)]
        if rows.empty:
            continue
        for metric in ["auc_final", "pr_auc_final", "auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1_score"]:
            if metric in rows.columns:
                out[f"stageA_{model}_{metric}_mean"] = safe_float(rows[metric].mean())
                out[f"stageA_{model}_{metric}_sd"] = safe_float(rows[metric].std(ddof=1))
    return out


def raw_metrics(run_dir: Path) -> dict[str, Any]:
    df = read_csv(run_dir / "classifier_only_readout/classifier_sweep_pooled_metrics.csv")
    out: dict[str, Any] = {}
    if df.empty:
        return out
    df = df.rename(columns={"readout_feature_set": "feature_set"})
    if "model_name" not in df.columns or "threshold_strategy" not in df.columns:
        return out
    mask = df["model_name"].astype(str).eq(PRIMARY_RAW_MODEL) & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    if "feature_set" in df.columns:
        mask &= df["feature_set"].astype(str).eq(PRIMARY_FEATURE_SET)
    rows = df[mask]
    if rows.empty:
        return out
    r = rows.iloc[0]
    for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "tn", "fp", "fn", "tp"]:
        out[f"stageB_raw_{metric}"] = safe_float(r.get(metric))
    return out


def oof_metrics(oof_dir: Path | None) -> tuple[dict[str, Any], pd.DataFrame]:
    out: dict[str, Any] = {}
    long_rows: list[dict[str, Any]] = []
    if oof_dir is None:
        return out, pd.DataFrame()
    pooled = read_csv(oof_dir / "calib_pooled_metrics.csv")
    if pooled.empty:
        return out, pd.DataFrame()
    rows = pooled[
        (pooled["model_name"].astype(str).eq(PRIMARY_OOF_MODEL))
        & (pooled["feature_set"].astype(str).eq(PRIMARY_FEATURE_SET))
        & (pooled["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD))
        & (pooled["calib_method"].astype(str).isin(CALIB_METHODS))
    ].copy()
    for _, r in rows.iterrows():
        method = str(r["calib_method"])
        prefix = method if method.startswith("oof_") else f"calib_{method}"
        rec = {
            "oof_dir": rel(oof_dir),
            "calib_method": method,
            "model_name": r.get("model_name"),
            "feature_set": r.get("feature_set"),
            "threshold_strategy": r.get("threshold_strategy"),
        }
        for metric in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "tn", "fp", "fn", "tp"]:
            rec[metric] = safe_float(r.get(metric))
            out[f"{prefix}_{metric}"] = safe_float(r.get(metric))
        long_rows.append(rec)
    philips = read_csv(oof_dir / "calib_philips_fpr_pooled.csv")
    if not philips.empty:
        philips = philips[
            (philips["model_name"].astype(str).eq(PRIMARY_OOF_MODEL))
            & (philips["feature_set"].astype(str).eq(PRIMARY_FEATURE_SET))
            & (philips["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD))
            & (philips["manufacturer"].astype(str).eq("Philips"))
        ]
        for method in CALIB_METHODS:
            pr = philips[philips["calib_method"].astype(str).eq(method)]
            if not pr.empty:
                prefix = method if method.startswith("oof_") else f"calib_{method}"
                r = pr.iloc[0]
                out[f"{prefix}_philips_cn_fp"] = safe_float(r.get("fp_cn_pooled"))
                out[f"{prefix}_philips_cn_n"] = safe_float(r.get("n_cn_pooled"))
                out[f"{prefix}_philips_cn_fpr"] = safe_float(r.get("fpr_cn_pooled"))
    foldwise = read_csv(oof_dir / "calib_foldwise_metrics.csv")
    if not foldwise.empty:
        for method in ["oof_ecdf", "oof_logitz"]:
            fr = foldwise[
                (foldwise["model_name"].astype(str).eq(PRIMARY_OOF_MODEL))
                & (foldwise["feature_set"].astype(str).eq(PRIMARY_FEATURE_SET))
                & (foldwise["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD))
                & (foldwise["calib_method"].astype(str).eq(method))
            ].copy()
            if not fr.empty:
                out[f"{method}_fold_auc_sd"] = safe_float(fr["auc"].std(ddof=1))
                out[f"{method}_fold_pr_auc_sd"] = safe_float(fr["pr_auc"].std(ddof=1))
                out[f"{method}_fold_auc_mean"] = safe_float(fr["auc"].mean())
                out[f"{method}_fold_pr_auc_mean"] = safe_float(fr["pr_auc"].mean())
                for _, rr in fr.iterrows():
                    fold = int(rr.get("fold"))
                    out[f"{method}_fold{fold}_auc"] = safe_float(rr.get("auc"))
                    out[f"{method}_fold{fold}_pr_auc"] = safe_float(rr.get("pr_auc"))
    return out, pd.DataFrame(long_rows)


def qc_metrics(run_dir: Path, beta: float, latent_dim: float) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    mi_rows: list[dict[str, Any]] = []
    leak_rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        fold_dir = run_dir / f"fold_{fold}"
        rd_path = fold_dir / f"fold_{fold}_rate_distortion.csv"
        if rd_path.exists():
            rd = pd.read_csv(rd_path)
            if not rd.empty:
                idx = rd["L_val_betaMax"].idxmin() if "L_val_betaMax" in rd.columns else rd.index[-1]
                r = rd.loc[idx]
                D = safe_float(r.get("D_val"))
                Rn = safe_float(r.get("R_val_nats"))
                Rb = safe_float(r.get("R_val_bits"))
                rows.append(
                    {
                        "D_val": D,
                        "R_val_bits": Rb,
                        "R_bits_per_latent_dim": safe_div(Rb, latent_dim),
                        "KLD_over_D": safe_div(Rn, D),
                        "beta_KLD_over_D": beta * safe_div(Rn, D),
                    }
                )
        info_path = fold_dir / f"fold_{fold}_test_latent_info_summary.csv"
        if info_path.exists():
            info = pd.read_csv(info_path)
            y = info[info["variable"].astype(str).eq("Y_target")]
            m = info[info["variable"].astype(str).eq("Manufacturer")]
            rec: dict[str, Any] = {}
            if not y.empty:
                yr = y.iloc[0]
                rec["MI_Z_Y_nats"] = safe_float(yr.get("mi_sum_nats"))
                rec["active_units"] = safe_float(yr.get("n_active"))
                rec["total_correlation_nats"] = safe_float(yr.get("total_correlation_nats"))
            if not m.empty:
                mr = m.iloc[0]
                rec["MI_Z_Manufacturer_nats"] = safe_float(mr.get("mi_sum_nats"))
            rec["MI_Manufacturer_over_MI_Y"] = safe_div(rec.get("MI_Z_Manufacturer_nats", np.nan), rec.get("MI_Z_Y_nats", np.nan))
            mi_rows.append(rec)
        leak_path = fold_dir / f"fold_{fold}_test_scanner_leakage_summary.csv"
        if leak_path.exists():
            leak = pd.read_csv(leak_path)
            if not leak.empty:
                lr = leak.iloc[0]
                leak_rows.append(
                    {
                        "test_scanner_raw_acc": safe_float(lr.get("acc_site_raw")),
                        "test_scanner_latent_acc": safe_float(lr.get("acc_site_latent")),
                    }
                )
    out: dict[str, Any] = {}
    if rows:
        df = pd.DataFrame(rows)
        for col in df.columns:
            out[f"{col}_mean"] = safe_float(df[col].mean())
            out[f"{col}_sd"] = safe_float(df[col].std(ddof=1))
    if mi_rows:
        df = pd.DataFrame(mi_rows)
        for col in df.columns:
            out[f"{col}_mean"] = safe_float(df[col].mean())
            out[f"{col}_sd"] = safe_float(df[col].std(ddof=1))
    if leak_rows:
        df = pd.DataFrame(leak_rows)
        for col in df.columns:
            out[f"{col}_mean"] = safe_float(df[col].mean())
            out[f"{col}_sd"] = safe_float(df[col].std(ddof=1))
    return out


def registry() -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg_index = load_config_index()
    omap = oof_map()
    rows: list[dict[str, Any]] = []
    oof_long_frames: list[pd.DataFrame] = []
    for run_dir in discover_run_dirs():
        run_name, args, config_path = args_from_run(run_dir, cfg_index)
        outer = int(args.get("outer_folds") or 5)
        repeats = int(args.get("repeated_outer_folds_n_repeats") or 1)
        fold_complete = [bool((run_dir / f"fold_{i}" / f"vae_model_fold_{i}.pt").exists()) for i in range(1, 6)]
        is_full5 = outer == 5 and repeats == 1 and all(fold_complete)
        # Keep completed 5-fold roots even if older names do not contain full5x5.
        beta = safe_float(args.get("beta_vae"))
        latent = safe_float(args.get("latent_dim"))
        channels = args.get("channels_to_use")
        oof_dir = omap.get(run_name) or omap.get(run_dir.name)
        rec: dict[str, Any] = {
            "run_name": run_name,
            "run_dir": rel(run_dir),
            "config_or_run_config": rel(config_path) if config_path else "",
            "completion_status": "complete_full5x5" if is_full5 else "incomplete_or_nonstandard",
            "n_completed_folds": int(sum(fold_complete)),
            "outer_folds": outer,
            "repeated_outer_folds_n_repeats": repeats,
            "channel_set_order": channels,
            "selected_channel_names": selected_channel_names(channels),
            "beta_vae": beta,
            "latent_dim": latent,
            "recon_loss_mode": args.get("recon_loss_mode"),
            "epochs_vae": args.get("epochs_vae"),
            "cyclical_beta_n_cycles": args.get("cyclical_beta_n_cycles"),
            "lr_scheduler_T0": args.get("lr_scheduler_T0"),
            "early_stopping_patience_vae": args.get("early_stopping_patience_vae"),
            "oof_calibration_dir": rel(oof_dir) if oof_dir else "",
            "stageB_raw_present": (run_dir / "classifier_only_readout/classifier_sweep_pooled_metrics.csv").exists(),
            "oof_calibration_present": bool(oof_dir and (oof_dir / "calib_pooled_metrics.csv").exists()),
        }
        rec.update(stagea_summary(run_dir))
        rec.update(raw_metrics(run_dir))
        ometrics, olong = oof_metrics(oof_dir)
        rec.update(ometrics)
        if not olong.empty:
            olong.insert(0, "run_name", run_name)
            olong.insert(1, "run_dir", rel(run_dir))
            oof_long_frames.append(olong)
        rec.update(qc_metrics(run_dir, beta=beta, latent_dim=latent))
        rows.append(rec)
    df = pd.DataFrame(rows)
    for col in ["oof_ecdf_auc", "stageB_raw_auc", "run_name"]:
        if col not in df.columns:
            df[col] = np.nan if col != "run_name" else ""
    df = df.sort_values(["oof_ecdf_auc", "stageB_raw_auc", "run_name"], ascending=[False, False, True], na_position="last")
    oof_long = pd.concat(oof_long_frames, ignore_index=True) if oof_long_frames else pd.DataFrame()
    return df, oof_long


def command_for_stage_a(params: dict[str, Any], run_name: str, channels: list[int], latent_dim: int, beta: float) -> list[str]:
    local = ROOT / f"results/revision_bspc_2026/{run_name}"
    cmd = [
        PYTHON,
        str(TRAINING_SCRIPT),
        "--global_tensor_path",
        str(params["global_tensor_path"]),
        "--metadata_path",
        str(params["metadata_path"]),
        "--output_dir",
        str(local),
    ]
    for key, value in params.items():
        if key in {"global_tensor_path", "metadata_path", "output_dir", "dry_run", "git_hash", "classifier_n_iter_overrides", "classifier_n_iter_json"}:
            continue
        if key == "channels_to_use":
            value = channels
        elif key == "latent_dim":
            value = latent_dim
        elif key == "beta_vae":
            value = beta
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        elif isinstance(value, list):
            cmd.append(flag)
            cmd.extend(str(v) for v in value)
        elif isinstance(value, dict):
            # Use explicit n_iter flags already present in params; skip dict internals.
            continue
        else:
            cmd.extend([flag, str(value)])
    cmd.extend(["--vae_required_metadata_cols", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex"])
    cmd.append("--vae_abort_if_val_split_fails")
    return cmd


def launch_plan() -> tuple[pd.DataFrame, str]:
    base = load_json(BASE_CONFIG)
    params = dict(base["parameters"])
    params["global_tensor_path"] = base["paths"]["global_tensor_path"]
    params["metadata_path"] = base["paths"]["metadata_path"]
    params["classifier_types"] = ["logreg", "svm"]
    candidates = [
        ("promoted_ch1_0_2_latent384_beta7p0", [1, 0, 2], 384, 7.0),
        ("promoted_ch1_0_2_latent384_beta9p5", [1, 0, 2], 384, 9.5),
        ("promoted_ch1_0_2_latent384_beta11p0", [1, 0, 2], 384, 11.0),
        ("ch1_2_latent384_beta7p0", [1, 2], 384, 7.0),
        ("ch1_2_latent384_beta2p75", [1, 2], 384, 2.75),
        ("promoted_ch1_0_2_latent256_beta9p5", [1, 0, 2], 256, 9.5),
        ("ch1only_latent256_beta3p75", [1], 256, 3.75),
    ]
    rows: list[dict[str, Any]] = []
    blocks: list[str] = []
    for stem, channels, latent_dim, beta in candidates:
        run_name = f"recover035_{stem}_T80_h10000_p560_full5x5_{DATE_TAG}"
        local = ROOT / f"results/revision_bspc_2026/{run_name}"
        big = Path(f"/media/diego/Datos/vae_AD_results/revision_bspc_2026/{run_name}")
        oof = ROOT / f"results/revision_bspc_2026/{run_name}_stageB_oof_score_calibration"
        stage_a = command_for_stage_a(params, run_name, channels, latent_dim, beta)
        stage_b = [
            PYTHON,
            str(STAGE_B_SCRIPT),
            "--run-dir",
            str(local),
            "--output-dir",
            str(local / "classifier_only_readout"),
            "--outer-folds",
            "5",
            "--inner-folds",
            "5",
            "--models",
            "logreg_l2",
            "svm_rbf",
            "--reuse-latent-cache",
        ]
        oof_cmd = [
            PYTHON,
            str(OOF_SCRIPT),
            "--run-dir",
            str(local),
            "--output-dir",
            str(oof),
        ]
        preflight = [
            f"test ! -e {shlex.quote(str(local))}",
            f"test ! -e {shlex.quote(str(big))}",
            f"mkdir -p {shlex.quote(str(big))}",
            f"ln -s {shlex.quote(str(big))} {shlex.quote(str(local))}",
        ]
        rows.append(
            {
                "candidate": stem,
                "run_name": run_name,
                "local_output_dir": str(local),
                "big_disk_output_dir": str(big),
                "channels_to_use": channels,
                "selected_channel_names": selected_channel_names(channels),
                "latent_dim": latent_dim,
                "beta_vae": beta,
                "primary_scientific_diff_vs_promoted": "channels/latent_dim/beta as listed; all other promoted config params preserved",
                "stageA_command": shlex.join(stage_a),
                "stageB_classifier_only_command": shlex.join(stage_b),
                "stageB_oof_score_calibration_command": shlex.join(oof_cmd),
                "prelaunch_symlink_commands": " && ".join(preflight),
                "will_train_if_executed": True,
                "prepared_only_not_run": True,
            }
        )
        blocks.append(
            "\n".join(
                [
                    f"## {run_name}",
                    "",
                    "Prelaunch symlink/clean-path setup:",
                    "",
                    "```bash",
                    "\n".join(preflight),
                    "```",
                    "",
                    "Stage A FULL 5x5 training command:",
                    "",
                    "```bash",
                    shlex.join(stage_a),
                    "```",
                    "",
                    "Stage B classifier-only readout command:",
                    "",
                    "```bash",
                    shlex.join(stage_b),
                    "```",
                    "",
                    "Post-hoc Stage B OOF score-calibration audit command:",
                    "",
                    "```bash",
                    shlex.join(oof_cmd),
                    "```",
                ]
            )
        )
    return pd.DataFrame(rows), "\n\n".join(blocks) + "\n"


def schema_text() -> str:
    cols = [
        "run_name",
        "run_dir",
        "channel_set_order",
        "latent_dim",
        "beta_vae",
        "recon_loss_mode",
        "beta_KLD_over_D_mean",
        "R_val_bits_mean",
        "D_val_mean",
        "active_units_mean",
        "total_correlation_nats_mean",
        "MI_Z_Y_nats_mean",
        "MI_Z_Manufacturer_nats_mean",
        "MI_Manufacturer_over_MI_Y_mean",
        "test_scanner_latent_acc_mean",
        "oof_ecdf_auc",
        "oof_ecdf_pr_auc",
        "oof_ecdf_balanced_accuracy",
        "oof_ecdf_sensitivity",
        "oof_ecdf_specificity",
        "oof_ecdf_f1",
        "oof_logitz_auc",
        "oof_logitz_pr_auc",
        "oof_ecdf_philips_cn_fp",
        "oof_ecdf_philips_cn_n",
        "oof_ecdf_philips_cn_fpr",
        "oof_ecdf_fold_auc_sd",
        "oof_ecdf_fold_pr_auc_sd",
        "completion_status",
        "decision",
    ]
    return "# Estimated Comparison Table Schema\n\n" + "\n".join(f"- `{c}`" for c in cols) + "\n"


def promotion_gate_text() -> str:
    return f"""# Promotion Gate Definition

Primary ranking rows:
- `logreg_l2_original / z_plus_age_sex / oof_ecdf / {PRIMARY_THRESHOLD}`
- `logreg_l2_original / z_plus_age_sex / oof_logitz / {PRIMARY_THRESHOLD}` as paired score-scale sensitivity.

References:
- Promoted [1,0,2] latent384 beta3.75: OOF-ECDF AUC={PROMOTED_REF['oof_ecdf_auc']:.6f}, PR-AUC={PROMOTED_REF['oof_ecdf_pr_auc']:.6f}, BA={PROMOTED_REF['oof_ecdf_ba']:.6f}, Sens={PROMOTED_REF['oof_ecdf_sens']:.6f}, F1={PROMOTED_REF['oof_ecdf_f1']:.6f}, Philips CN FPR={PROMOTED_REF['oof_ecdf_philips_cn_fpr']:.6f}.
- ch1-only latent384 beta3.75: OOF-ECDF AUC={CH1_REF['oof_ecdf_auc']:.6f}, PR-AUC={CH1_REF['oof_ecdf_pr_auc']:.6f}; OOF-logitz AUC={CH1_REF['oof_logitz_auc']:.6f}, PR-AUC={CH1_REF['oof_logitz_pr_auc']:.6f}.

Strict AUC-improvement gate:
- Candidate OOF-ECDF AUC must exceed both promoted [1,0,2] and ch1-only references.
- Candidate OOF-logitz AUC should also exceed both corresponding references or show no contradiction with the OOF-ECDF rank.

Clinical/readout safety gate:
- PR-AUC must be at least the promoted [1,0,2] PR-AUC and preferably not below ch1-only.
- BA, sensitivity, and F1 must not be materially worse than promoted [1,0,2].
- Philips CN FPR must not exceed promoted [1,0,2].
- Test latent scanner/manufacturer leakage must not worsen materially.

Interpretation:
- Passing only AUC but failing PR-AUC/BA/sensitivity/FPR is sensitivity-only, not primary promotion.
- A high-beta run that improves scanner leakage but loses AD/CN ranking is a negative regularization sensitivity.
- No OASIS result should be used for promotion in this launch plan unless a separate locked external protocol is declared in advance.
"""


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if args.dry_run:
        print(f"Dry-run OK. Would write registry and launch plan to {out_dir}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    reg, oof_long = registry()
    plan, command_md = launch_plan()
    write_table(reg, out_dir, "completed_full5x5_registry", max_rows=240)
    write_table(oof_long, out_dir, "completed_full5x5_oof_metrics_long", max_rows=300)
    write_table(plan, out_dir, "candidate_launch_plan", max_rows=20)
    (out_dir / "candidate_launch_commands.md").write_text(command_md, encoding="utf-8")
    (out_dir / "estimated_comparison_table_schema.md").write_text(schema_text(), encoding="utf-8")
    (out_dir / "promotion_gate_definition.md").write_text(promotion_gate_text(), encoding="utf-8")
    readme = [
        "# FULL 5x5 Registry and Proactive Follow-up Launch Plan",
        "",
        "This package inventories completed FULL 5x5 ADNI runs and prepares launch commands for seven follow-up sensitivity/candidate runs.",
        "",
        "No training, no OASIS scoring, no tensor mutation, no metadata mutation, and no model artifact mutation were performed.",
        "",
        "The launch commands are prepared only. They should not be executed without explicit confirmation.",
    ]
    (out_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    write_json(
        out_dir / "command_log.json",
        {
            "created_utc": now_utc(),
            "script": rel(Path(__file__)),
            "output_dir": rel(out_dir),
            "n_completed_full5x5_roots_scanned": int(len(reg)),
            "n_oof_metric_rows": int(len(oof_long)),
            "n_launch_candidates": int(len(plan)),
            "guardrails": [
                "no training launched",
                "no OASIS scoring",
                "no tensor modification",
                "no metadata modification",
                "no model artifact modification",
            ],
        },
    )
    print(f"Wrote package: {out_dir}")
    print(f"completed FULL-like roots: {len(reg)}")
    print(f"launch candidates prepared: {len(plan)}")


if __name__ == "__main__":
    main()
