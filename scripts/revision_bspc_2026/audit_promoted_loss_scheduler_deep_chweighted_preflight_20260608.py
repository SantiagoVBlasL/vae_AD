#!/usr/bin/env python3
"""Read-only VAE objective/scheduler audit plus weighted-loss preflight."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from betavae_xai.normalization import normalize_inter_channel_fold  # noqa: E402
from run_vae_clf_ad_inference import (  # noqa: E402
    RECON_LOSS_MODE_CURRENT,
    RECON_LOSS_MODE_MSE_OFFDIAG_CHANNEL_MEAN_SUM,
    RECON_LOSS_MODE_MSE_OFFDIAG_CHANNEL_WEIGHTED_SUM,
    vae_reconstruction_loss,
)


RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"
OUT_DEFAULT = RESULTS / "promoted_loss_scheduler_deep_audit_chweighted_preflight_20260608"
REFERENCE_CONFIG = PROJECT_ROOT / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5.json"
CANDIDATE_CONFIG = PROJECT_ROOT / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_chweightedPearson50_T80_h10000_p560_full5x5.json"
CANDIDATE_LAUNCHER = PROJECT_ROOT / "scripts/revision_bspc_2026/run_adni_v5_1c_recover035_latent384_beta3p75_chweightedPearson50_T80_h10000_p560_full5x5.py"
SMOKE_TEST = PROJECT_ROOT / "scripts/revision_bspc_2026/smoke_test_vae_recon_loss_modes.py"
EVIDENCE_MAP = RESULTS / "final_full_model_evidence_map_with_ch12_beta2p75_20260607/full_model_evidence_map.csv"

SELECTED_CHANNELS = [1, 0, 2]
SELECTED_NAMES = [
    "Pearson_Full_FisherZ_Signed",
    "Pearson_OMST_GCE_Signed_Weighted",
    "MI_KNN_Symmetric",
]
WEIGHTS = [0.50, 0.25, 0.25]
PRIMARY_MODEL = "logreg_l2_original"
PRIMARY_FEATURE_SET = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"

RUNS = [
    {
        "model_id": "promoted_ch102_latent384_beta3p75",
        "evidence_model_id": "promoted_latent384_beta3p75_ch1_0_2",
        "display_name": "promoted [1,0,2] beta3.75 current loss",
        "run_dir": RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5",
        "config": REFERENCE_CONFIG,
        "stageb_dir": RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration",
    },
    {
        "model_id": "ch1only_latent384_beta3p75",
        "evidence_model_id": "ch1only_latent384_beta3p75",
        "display_name": "ch1-only beta3.75",
        "run_dir": RESULTS / "recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5",
        "config": PROJECT_ROOT / "configs/runs/adni_v5_1c_recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5.json",
        "stageb_dir": RESULTS / "recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5_stageB_oof_score_calibration",
    },
    {
        "model_id": "latent384_beta6p5",
        "evidence_model_id": "latent384_beta6p5",
        "display_name": "[1,0,2] beta6.5 current loss",
        "run_dir": RESULTS / "recover035_latent384_beta6p5_T80_h10000_p560_full5x5",
        "config": PROJECT_ROOT / "configs/runs/adni_v5_1c_recover035_latent384_beta6p5_T80_h10000_p560_full5x5.json",
        "stageb_dir": RESULTS / "recover035_latent384_beta6p5_stageB_oof_score_calibration",
    },
    {
        "model_id": "chmeanloss_latent384_beta3p75",
        "evidence_model_id": "chmeanloss_latent384_beta3p75_stageB_oof_ecdf",
        "display_name": "[1,0,2] beta3.75 channel-mean loss",
        "run_dir": RESULTS / "recover035_latent384_beta3p75_chmeanloss_T80_h10000_p560_full5x5",
        "config": PROJECT_ROOT / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_chmeanloss_T80_h10000_p560_full5x5.json",
        "stageb_dir": RESULTS / "recover035_latent384_beta3p75_chmeanloss_stageB_oof_score_calibration",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--python-executable", default="/home/diego/anaconda3/envs/vae_ad/bin/python")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def md_table(df: pd.DataFrame, max_rows: int = 300) -> str:
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
    return view.to_markdown(index=False) + "\n"


def write_table(stem: Path, df: pd.DataFrame, max_rows: int = 300) -> None:
    df.to_csv(stem.with_suffix(".csv"), index=False)
    stem.with_suffix(".md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def run_cmd(cmd: list[str]) -> dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "cmd": cmd,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def cosine_lr_at_epoch(epoch_1based: int, cfg: dict[str, Any]) -> float:
    params = cfg["parameters"]
    lr = float(params["lr_vae"])
    eta_min = float(params.get("lr_scheduler_eta_min", 0.0))
    t0 = int(params["lr_scheduler_T0"])
    # Scheduler is stepped with zero-based fractional epoch. Approximate at epoch start.
    t_cur = (int(epoch_1based) - 1) % t0
    return float(eta_min + 0.5 * (lr - eta_min) * (1.0 + np.cos(np.pi * t_cur / t0)))


def scheduler_phase(epoch_1based: int, cfg: dict[str, Any]) -> dict[str, Any]:
    params = cfg["parameters"]
    t0 = int(params["lr_scheduler_T0"])
    idx0 = int(epoch_1based) - 1
    pos0 = idx0 % t0
    cycle_len = float(params["epochs_vae"]) / float(params["cyclical_beta_n_cycles"])
    return {
        "cycle_len_epochs": cycle_len,
        "lr_scheduler_T0": t0,
        "best_epoch_cycle_position_1based": int(pos0 + 1),
        "epochs_since_lr_restart": int(pos0),
        "epochs_to_next_lr_restart": int(t0 - pos0) if pos0 else 0,
        "distance_to_nearest_lr_restart": int(min(pos0, t0 - pos0 if pos0 else 0)),
        "estimated_lr_at_best_epoch_start": cosine_lr_at_epoch(epoch_1based, cfg),
    }


def primary_stageb_metrics(stageb_dir: Path) -> dict[str, Any]:
    pooled = pd.read_csv(stageb_dir / "calib_pooled_metrics.csv")
    row = pooled[
        pooled["model_name"].eq(PRIMARY_MODEL)
        & pooled["feature_set"].eq(PRIMARY_FEATURE_SET)
        & pooled["calib_method"].eq(PRIMARY_CALIB)
        & pooled["threshold_strategy"].eq(PRIMARY_THRESHOLD)
    ].iloc[0]
    philips = pd.read_csv(stageb_dir / "calib_philips_fpr_pooled.csv")
    prow = philips[
        philips["model_name"].eq(PRIMARY_MODEL)
        & philips["feature_set"].eq(PRIMARY_FEATURE_SET)
        & philips["calib_method"].eq(PRIMARY_CALIB)
        & philips["threshold_strategy"].eq(PRIMARY_THRESHOLD)
        & philips["manufacturer"].eq("Philips")
    ].iloc[0]
    return {
        "stageB_auc": float(row["auc"]),
        "stageB_pr_auc": float(row["pr_auc"]),
        "stageB_ba": float(row["balanced_accuracy"]),
        "stageB_f1": float(row["f1"]),
        "stageB_sensitivity": float(row["sensitivity"]),
        "stageB_specificity": float(row["specificity"]),
        "philips_cn_fp": int(prow["fp_cn_pooled"]),
        "philips_cn_n": int(prow["n_cn_pooled"]),
        "philips_cn_fpr": float(prow["fpr_cn_pooled"]),
    }


def oasis_status_lookup() -> dict[str, dict[str, Any]]:
    if not EVIDENCE_MAP.exists():
        return {}
    df = pd.read_csv(EVIDENCE_MAP)
    out: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        out[str(row["model_id"])] = {
            "oasis_status": row.get("oasis_status", ""),
            "oasis_concatenated_auc": row.get("oasis_concatenated_auc", np.nan),
            "oasis_concatenated_pr_auc": row.get("oasis_concatenated_pr_auc", np.nan),
            "oasis_runwise164_auc": row.get("oasis_runwise164_auc", np.nan),
            "oasis_runwise164_pr_auc": row.get("oasis_runwise164_pr_auc", np.nan),
            "oasis_runwise140_auc": row.get("oasis_runwise140_auc", np.nan),
            "oasis_runwise140_pr_auc": row.get("oasis_runwise140_pr_auc", np.nan),
        }
    return out


def primary_fold_stageb(stageb_dir: Path) -> pd.DataFrame:
    fold = pd.read_csv(stageb_dir / "calib_foldwise_metrics.csv")
    return fold[
        fold["model_name"].eq(PRIMARY_MODEL)
        & fold["feature_set"].eq(PRIMARY_FEATURE_SET)
        & fold["calib_method"].eq(PRIMARY_CALIB)
        & fold["threshold_strategy"].eq(PRIMARY_THRESHOLD)
    ][["fold", "auc", "pr_auc", "balanced_accuracy", "f1", "sensitivity", "specificity"]].copy()


def latent_info(run_dir: Path, fold: int) -> dict[str, Any]:
    path = run_dir / f"fold_{fold}" / f"fold_{fold}_trainDev_latent_info_summary.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    y = df[df["variable"].eq("Y_target")]
    mfr = df[df["variable"].eq("Manufacturer")]
    out: dict[str, Any] = {}
    if not y.empty:
        yr = y.iloc[0]
        out.update(
            {
                "active_units": float(yr.get("n_active", np.nan)),
                "total_correlation_nats": float(yr.get("total_correlation_nats", np.nan)),
                "MI_Z_Y_nats": float(yr.get("mi_sum_nats", np.nan)),
            }
        )
    if not mfr.empty:
        mr = mfr.iloc[0]
        out["MI_Z_Manufacturer_nats"] = float(mr.get("mi_sum_nats", np.nan))
    if out.get("MI_Z_Y_nats", np.nan):
        out["MI_Manufacturer_over_MI_Y"] = out.get("MI_Z_Manufacturer_nats", np.nan) / out["MI_Z_Y_nats"]
    return out


def run_fold_rows(run: dict[str, Any]) -> list[dict[str, Any]]:
    cfg = load_json(run["config"])
    params = cfg["parameters"]
    stageb_fold = primary_fold_stageb(run["stageb_dir"]).set_index("fold")
    rows: list[dict[str, Any]] = []
    for fold in range(1, int(params["outer_folds"]) + 1):
        rd_path = run["run_dir"] / f"fold_{fold}" / f"fold_{fold}_rate_distortion.csv"
        rd = pd.read_csv(rd_path)
        best = rd.loc[rd["L_val_betaMax"].astype(float).idxmin()]
        best_epoch = int(best["epoch"])
        r_nats = float(best["R_val_nats"])
        d_val = float(best["D_val"])
        beta_max = float(params["beta_vae"])
        row = {
            "model_id": run["model_id"],
            "display_name": run["display_name"],
            "fold": fold,
            "recon_loss_mode": params.get("recon_loss_mode", "mse_sum_batchmean_current"),
            "recon_loss_channel_weights": str(params.get("recon_loss_channel_weights", "")),
            "channels_to_use": str(params.get("channels_to_use")),
            "latent_dim": int(params["latent_dim"]),
            "beta_vae": beta_max,
            "best_epoch": best_epoch,
            "final_epoch_logged": int(rd["epoch"].max()),
            "beta_at_best_epoch": float(best["beta"]),
            "beta_phase_at_best_epoch": float(best["beta"]) / beta_max if beta_max else np.nan,
            "D_val_best": d_val,
            "R_val_nats_best": r_nats,
            "R_val_bits_best": float(best["R_val_bits"]),
            "bits_per_dim_best": float(best["R_val_bits"]) / float(params["latent_dim"]),
            "KLD_over_D_best": r_nats / d_val if d_val else np.nan,
            "beta_KLD_over_D_best": beta_max * r_nats / d_val if d_val else np.nan,
            "L_val_betaMax_best": float(best["L_val_betaMax"]),
        }
        row.update(scheduler_phase(best_epoch, cfg))
        row.update(latent_info(run["run_dir"], fold))
        if fold in stageb_fold.index:
            sf = stageb_fold.loc[fold]
            row.update(
                {
                    "stageB_fold_auc": float(sf["auc"]),
                    "stageB_fold_pr_auc": float(sf["pr_auc"]),
                    "stageB_fold_ba": float(sf["balanced_accuracy"]),
                    "stageB_fold_f1": float(sf["f1"]),
                }
            )
        rows.append(row)
    return rows


def load_tensor_from_config(config_path: Path) -> tuple[np.ndarray, list[str]]:
    cfg = load_json(config_path)
    tensor_path = Path(cfg["paths"]["global_tensor_path"])
    if not tensor_path.is_absolute():
        tensor_path = PROJECT_ROOT / tensor_path
    with np.load(tensor_path, allow_pickle=False) as zf:
        tensor = np.asarray(zf["global_tensor_data"], dtype=np.float32)[:, SELECTED_CHANNELS, :, :]
        channel_names = [str(x) for x in np.asarray(zf["channel_names"]).astype(str).tolist()]
    selected = [channel_names[i] for i in SELECTED_CHANNELS]
    if selected != SELECTED_NAMES:
        raise RuntimeError(f"Selected channel mismatch: {selected}")
    return tensor, selected


def loss_value(x_np: np.ndarray, mode: str, weights: list[float] | None = None) -> float:
    x = torch.as_tensor(x_np, dtype=torch.float32)
    recon = torch.zeros_like(x)
    return float(
        vae_reconstruction_loss(recon, x, mode=mode, channel_weights=weights)
        .detach()
        .cpu()
        .item()
    )


def scale_probe() -> pd.DataFrame:
    tensor, _ = load_tensor_from_config(REFERENCE_CONFIG)
    rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        fold_dir = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5" / f"fold_{fold}"
        pool_idx = np.load(fold_dir / "vae_training_pool_tensor_idx.npy").astype(int)
        train_local = np.load(fold_dir / "vae_actual_train_idx_local_to_pool.npy").astype(int)
        val_local = np.load(fold_dir / "vae_internal_val_idx_local_to_pool.npy").astype(int)
        pool = tensor[pool_idx]
        pool_norm, _ = normalize_inter_channel_fold(
            pool,
            train_local,
            mode="zscore_offdiag",
            selected_channel_original_names=SELECTED_NAMES,
        )
        for split_name, local_idx in [
            ("vae_actual_train", train_local),
            ("vae_internal_val", val_local),
            ("vae_pool_all", np.arange(pool_norm.shape[0], dtype=int)),
        ]:
            x = pool_norm[local_idx]
            current = loss_value(x, RECON_LOSS_MODE_CURRENT)
            chmean = loss_value(x, RECON_LOSS_MODE_MSE_OFFDIAG_CHANNEL_MEAN_SUM)
            weighted = loss_value(x, RECON_LOSS_MODE_MSE_OFFDIAG_CHANNEL_WEIGHTED_SUM, WEIGHTS)
            rows.append(
                {
                    "fold": fold,
                    "split": split_name,
                    "n_subjects": int(len(local_idx)),
                    "D_current_zero_recon": current,
                    "D_chmean_zero_recon": chmean,
                    "D_chweighted_zero_recon": weighted,
                    "chmean_over_current": chmean / current if current else np.nan,
                    "chweighted_over_current": weighted / current if current else np.nan,
                    "chweighted_over_chmean": weighted / chmean if chmean else np.nan,
                    "weights_selected_order": str(WEIGHTS),
                    "selected_channel_names": " | ".join(SELECTED_NAMES),
                }
            )
    return pd.DataFrame(rows)


def validate_candidate_config() -> pd.DataFrame:
    ref = load_json(REFERENCE_CONFIG)
    cand = load_json(CANDIDATE_CONFIG)
    r = dict(ref["parameters"])
    c = dict(cand["parameters"])
    diff = {k: (r.get(k), c.get(k)) for k in sorted(set(r) | set(c)) if r.get(k) != c.get(k)}
    expected = {
        "recon_loss_channel_weights": (None, WEIGHTS),
        "recon_loss_mode": ("mse_sum_batchmean_current", "mse_offdiag_channel_weighted_sum"),
    }
    rows = []
    rows.append({"check": "strict_parameter_diff", "actual": json.dumps(diff, sort_keys=True), "expected": json.dumps(expected, sort_keys=True), "pass": diff == expected})
    rows.append({"check": "weights_sum_to_one", "actual": sum(c["recon_loss_channel_weights"]), "expected": 1.0, "pass": np.isclose(sum(c["recon_loss_channel_weights"]), 1.0)})
    rows.append({"check": "selected_channels_unchanged", "actual": str(c["channels_to_use"]), "expected": str(SELECTED_CHANNELS), "pass": c["channels_to_use"] == SELECTED_CHANNELS})
    rows.append({"check": "selected_channel_names_unchanged", "actual": str(cand["selected_channel_names"]), "expected": str(SELECTED_NAMES), "pass": cand["selected_channel_names"] == SELECTED_NAMES})
    for key in ["global_tensor_path", "metadata_path", "training_script"]:
        rows.append({"check": f"path_{key}_unchanged", "actual": cand["paths"][key], "expected": ref["paths"][key], "pass": cand["paths"][key] == ref["paths"][key]})
    outdir = PROJECT_ROOT / cand["paths"]["output_dir"]
    stale = []
    if outdir.exists():
        stale = [p.name for p in outdir.iterdir() if p.name.startswith("fold_") or p.name in {"classifier_only_readout", "latent_cache", "run_manifest.json"}]
    rows.append({"check": "stale_output_markers", "actual": ";".join(stale), "expected": "", "pass": len(stale) == 0})
    return pd.DataFrame(rows)


def write_texts(out_dir: Path, fold_df: pd.DataFrame, summary_df: pd.DataFrame, scale_df: pd.DataFrame) -> None:
    prom = summary_df[summary_df["model_id"].eq("promoted_ch102_latent384_beta3p75")].iloc[0]
    ch1 = summary_df[summary_df["model_id"].eq("ch1only_latent384_beta3p75")].iloc[0]
    b65 = summary_df[summary_df["model_id"].eq("latent384_beta6p5")].iloc[0]
    chmean = summary_df[summary_df["model_id"].eq("chmeanloss_latent384_beta3p75")].iloc[0]
    beta_linear = float(ch1["beta_KLD_over_D_best_mean"] / prom["beta_KLD_over_D_best_mean"] * prom["beta_vae"]) if prom["beta_KLD_over_D_best_mean"] else np.nan
    b65_gain = (b65["beta_KLD_over_D_best_mean"] - prom["beta_KLD_over_D_best_mean"]) / (6.5 - 3.75)
    beta_to_ch1_from_b65_slope = 3.75 + (ch1["beta_KLD_over_D_best_mean"] - prom["beta_KLD_over_D_best_mean"]) / b65_gain if b65_gain else np.nan
    mean_weighted_over_current = float(scale_df.loc[scale_df["split"].eq("vae_pool_all"), "chweighted_over_current"].mean())
    mean_weighted_over_chmean = float(scale_df.loc[scale_df["split"].eq("vae_pool_all"), "chweighted_over_chmean"].mean())

    scheduler = """# Scheduler/Beta-Cycle Audit

All four audited runs keep `epochs_vae=10000`, `cyclical_beta_n_cycles=125`, and
`lr_scheduler_T0=80`, so beta cycle length is exactly 80 epochs and matches the
cosine-warm restart period. The best epochs do not show a consistent systematic
need for `T0=160`; changing T0 would intentionally de-align the historical
locked schedule from the beta cycles unless cycles are also changed.

Recommendation: `T0=160_not_justified_by_histories`; keep T80 for the prepared
candidate.
"""
    (out_dir / "scheduler_alignment_assessment.md").write_text(scheduler, encoding="utf-8")

    beta_text = f"""# Effective Regularization Interpretation

Mean beta*KLD/D:

- promoted [1,0,2] beta3.75 current loss: `{prom['beta_KLD_over_D_best_mean']:.6f}`
- ch1-only beta3.75: `{ch1['beta_KLD_over_D_best_mean']:.6f}`
- [1,0,2] beta6.5 current loss: `{b65['beta_KLD_over_D_best_mean']:.6f}`
- [1,0,2] chmeanloss beta3.75: `{chmean['beta_KLD_over_D_best_mean']:.6f}`

The ch1-only beta3.75 run is stronger because its reconstruction D is much
smaller under a one-channel objective while R remains substantial; with the same
nominal beta this increases beta*KLD/D. A purely linear analogue from promoted
beta3.75 to the ch1-only effective regime is beta approximately
`{beta_linear:.3f}`.

However, observed beta scaling is not linear: beta6.5 only reached
`{b65['beta_KLD_over_D_best_mean']:.6f}`, far below ch1-only. Using the observed
beta3.75 to beta6.5 slope would imply beta approximately
`{beta_to_ch1_from_b65_slope:.3f}` to match ch1-only, which is not a sensible
single-step continuation after beta6.5 failed the ADNI gate.

The prepared weighted-loss candidate is therefore framed as a loss-scaling
sensitivity, not as a beta9.5/11 retraining recommendation. Scale probe means:
weighted/current=`{mean_weighted_over_current:.6f}`, weighted/chmean=`{mean_weighted_over_chmean:.6f}`.
"""
    (out_dir / "effective_regularization_beta_scaling.md").write_text(beta_text, encoding="utf-8")

    loss_text = """# Loss Implementation Audit

Existing modes:

- `mse_sum_batchmean_current`: historical MSE over all channels and pixels,
  summed over tensor entries and divided by batch size. Its D scale grows with
  channel count and includes diagonal entries.
- `offdiag_channelmean_sum` / `mse_offdiag_channel_mean_sum`: off-diagonal
  squared-error sum within each channel, mean across selected channels, mean
  over batch. This preserves approximate single-channel D scale for identical
  channel errors.

New default-off mode:

- `mse_offdiag_channel_weighted_sum`: off-diagonal squared-error sum within each
  selected channel, explicit non-negative channel weights that must sum to one,
  weighted channel sum, mean over batch.

Candidate weights in selected order `[1,0,2]`:

- Pearson Full: 0.50
- OMST: 0.25
- MI: 0.25

Unit tests validate that weights sum to one, equal weights match
`mse_offdiag_channel_mean_sum`, and a one-hot channel weight matches the
corresponding single-channel off-diagonal sum.
"""
    (out_dir / "loss_implementation_audit.md").write_text(loss_text, encoding="utf-8")

    final = """# Final Recommendation

Decision: `prepare_chweighted_candidate_only_no_training`.

The audit supports the mechanistic explanation that ch1-only beta3.75 has a
larger effective beta*KLD/D because the one-channel reconstruction scale is much
smaller, not because nominal beta is intrinsically stronger. A linear beta
analogue for [1,0,2] is near 9.5 by simple ratio, but observed beta6.5 behavior
was not linear and did not pass the ADNI gate. T0=160 is not justified by the
available histories because the current T80 schedule is exactly aligned with
the 80-epoch beta cycles.

The prepared candidate `recover035_latent384_beta3p75_chweightedPearson50_T80_h10000_p560_full5x5`
keeps beta, architecture, channels, dropout, scheduler, metadata, and split
policy fixed, and changes only the off-diagonal reconstruction loss to a
weighted channel average. It is ready for dry-run/preflight only; no training was
launched.
"""
    (out_dir / "final_recommendation.md").write_text(final, encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    fold_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    oasis_by_evidence_id = oasis_status_lookup()
    for run in RUNS:
        rows = run_fold_rows(run)
        fold_rows.extend(rows)
        fold_df_run = pd.DataFrame(rows)
        stageb = primary_stageb_metrics(run["stageb_dir"])
        summary = {
            "model_id": run["model_id"],
            "display_name": run["display_name"],
            "beta_vae": float(fold_df_run["beta_vae"].iloc[0]),
            "latent_dim": int(fold_df_run["latent_dim"].iloc[0]),
            "recon_loss_mode": str(fold_df_run["recon_loss_mode"].iloc[0]),
            "D_val_best_mean": float(fold_df_run["D_val_best"].mean()),
            "R_val_nats_best_mean": float(fold_df_run["R_val_nats_best"].mean()),
            "R_val_bits_best_mean": float(fold_df_run["R_val_bits_best"].mean()),
            "bits_per_dim_best_mean": float(fold_df_run["bits_per_dim_best"].mean()),
            "KLD_over_D_best_mean": float(fold_df_run["KLD_over_D_best"].mean()),
            "beta_KLD_over_D_best_mean": float(fold_df_run["beta_KLD_over_D_best"].mean()),
            "active_units_mean": float(fold_df_run["active_units"].mean()),
            "total_correlation_nats_mean": float(fold_df_run["total_correlation_nats"].mean()),
            "MI_Z_Y_nats_mean": float(fold_df_run["MI_Z_Y_nats"].mean()),
            "MI_Z_Manufacturer_nats_mean": float(fold_df_run["MI_Z_Manufacturer_nats"].mean()),
            "MI_Manufacturer_over_MI_Y_mean": float(fold_df_run["MI_Manufacturer_over_MI_Y"].mean()),
        }
        summary.update(stageb)
        summary.update(oasis_by_evidence_id.get(str(run["evidence_model_id"]), {}))
        summary_rows.append(summary)

    fold_df = pd.DataFrame(fold_rows)
    summary_df = pd.DataFrame(summary_rows)
    scale_df = scale_probe()
    config_guard = validate_candidate_config()

    write_table(out_dir / "vae_objective_scheduler_by_fold", fold_df)
    write_table(out_dir / "vae_objective_scheduler_summary", summary_df)
    write_table(out_dir / "loss_scale_probe", scale_df)
    write_table(out_dir / "candidate_strict_diff_guard", config_guard)

    pycompile = run_cmd([args.python_executable, "-m", "py_compile", str(CANDIDATE_LAUNCHER), str(SMOKE_TEST), str(Path(__file__))])
    smoke = run_cmd([args.python_executable, str(SMOKE_TEST)])
    dryrun = run_cmd([args.python_executable, str(CANDIDATE_LAUNCHER), "--dry-run"])
    validations = pd.DataFrame(
        [
            {"check": "py_compile", "returncode": pycompile["returncode"], "pass": pycompile["returncode"] == 0},
            {"check": "weighted_loss_unit_tests", "returncode": smoke["returncode"], "pass": smoke["returncode"] == 0},
            {"check": "candidate_launcher_dry_run", "returncode": dryrun["returncode"], "pass": dryrun["returncode"] == 0},
        ]
    )
    write_table(out_dir / "validation_checks", validations)
    (out_dir / "candidate_launcher_dryrun_stdout.txt").write_text(dryrun["stdout"], encoding="utf-8")
    (out_dir / "candidate_launcher_dryrun_stderr.txt").write_text(dryrun["stderr"], encoding="utf-8")
    (out_dir / "weighted_loss_unit_test_stdout.txt").write_text(smoke["stdout"], encoding="utf-8")
    (out_dir / "weighted_loss_unit_test_stderr.txt").write_text(smoke["stderr"], encoding="utf-8")

    run_manifest = pd.DataFrame(
        [
            {
                "candidate_run_name": "recover035_latent384_beta3p75_chweightedPearson50_T80_h10000_p560_full5x5",
                "config": str(CANDIDATE_CONFIG),
                "launcher": str(CANDIDATE_LAUNCHER),
                "selected_channels": str(SELECTED_CHANNELS),
                "selected_channel_names": " | ".join(SELECTED_NAMES),
                "recon_loss_mode": RECON_LOSS_MODE_MSE_OFFDIAG_CHANNEL_WEIGHTED_SUM,
                "recon_loss_channel_weights": str(WEIGHTS),
                "training_launched": False,
                "oasis_scoring_run": False,
            }
        ]
    )
    write_table(out_dir / "run_manifest", run_manifest)
    write_texts(out_dir, fold_df, summary_df, scale_df)
    readme = """# Promoted Loss/Scheduler Deep Audit and Chweighted Preflight

This package audits the VAE objective and scheduler regime for the promoted
[1,0,2] beta3.75 model, ch1-only beta3.75, beta6.5, and chmeanloss, then
preflights a weighted off-diagonal reconstruction-loss candidate.

No training, OASIS scoring, tensor modification, metadata modification, or model
artifact modification was performed.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    command_log = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__)),
        "output_dir": str(out_dir),
        "runs": [{k: str(v) for k, v in run.items()} for run in RUNS],
        "candidate_config": str(CANDIDATE_CONFIG),
        "candidate_launcher": str(CANDIDATE_LAUNCHER),
        "validation_returncodes": {
            "py_compile": pycompile["returncode"],
            "weighted_loss_unit_tests": smoke["returncode"],
            "candidate_launcher_dry_run": dryrun["returncode"],
        },
        "guardrails": [
            "no training",
            "no OASIS scoring",
            "no tensor modification",
            "no metadata modification",
            "no model artifact modification",
        ],
    }
    (out_dir / "command_log.json").write_text(json.dumps(command_log, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "output_dir": str(out_dir),
        "fold_rows": int(fold_df.shape[0]),
        "summary_rows": int(summary_df.shape[0]),
        "scale_probe_rows": int(scale_df.shape[0]),
        "validation_pass": bool(validations["pass"].all() and config_guard["pass"].all()),
        "dry_run_command": shlex.join([args.python_executable, str(CANDIDATE_LAUNCHER), "--dry-run"]),
    }, indent=2))
    if not bool(validations["pass"].all() and config_guard["pass"].all()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
