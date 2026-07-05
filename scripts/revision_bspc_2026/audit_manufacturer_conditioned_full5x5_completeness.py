#!/usr/bin/env python
"""Read-only completeness audit for the Manufacturer-conditioned FULL 5x5 VAE.

The audit checks training maturity, data/split integrity, classifier
hyperparameter boundaries, and OASIS external sensitivity results. It writes
only new audit artifacts and never modifies tensors, metadata, ledgers, configs,
or existing model outputs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BRANCH_ROOT = PROJECT_ROOT / "results" / "revision_bspc_2026" / "conditional_beta_vae_manufacturer_full5x5_mfrrecovered035_clfpoollocked"
LOCKED_RUN = PROJECT_ROOT / "results" / "revision_bspc_2026" / "adni_v5_1_batch20260514b_ch1_0_2_horizon4480_cycles56_full_5x5"
OASIS_SECONDARY = PROJECT_ROOT / "results" / "revision_bspc_2026" / "oasis_tanda_2026_05_25_external_scoring_secondary_models"
OASIS_140TR = PROJECT_ROOT / "results" / "revision_bspc_2026" / "oasis_tanda_2026_05_25_140TR_sensitivity"
OUTPUT_DIR = PROJECT_ROOT / "results" / "revision_bspc_2026" / "conditional_beta_vae_manufacturer_full5x5_completeness_audit"
TARGET_STRATEGY = "inner_oof_target_sens_ge_0p70_max_spec"
MODEL_GRID_C_VALUES = [0.001, 0.01, 0.1, 1.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--branch-root", type=Path, default=BRANCH_ROOT)
    parser.add_argument("--locked-run", type=Path, default=LOCKED_RUN)
    parser.add_argument("--oasis-secondary-dir", type=Path, default=OASIS_SECONDARY)
    parser.add_argument("--oasis-140tr-dir", type=Path, default=OASIS_140TR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def write_csv_md(df: pd.DataFrame, csv_path: Path, md_path: Path, title: str) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        if df.empty:
            f.write("_No rows._\n")
        else:
            f.write(df.to_markdown(index=False))
            f.write("\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_runs(branch_root: Path) -> dict[str, Path]:
    return {
        "matched_unconditioned_baseline": branch_root / "runs" / "ch1_0_2_baseline_unconditioned",
        "manufacturer_conditioned": branch_root / "runs" / "ch1_0_2_decoder_only_manufacturer",
    }


def numeric_list(payload: dict[str, Any], key: str) -> np.ndarray:
    arr = np.asarray(payload.get(key, []), dtype=float)
    return arr[np.isfinite(arr)]


def phase_labels(epoch: int, cycle_len: int = 80, beta_ratio_increase: float = 0.4) -> tuple[int, str, str]:
    phase = int(epoch % cycle_len)
    if phase <= 5:
        lr_phase = "near_restart"
    elif phase >= cycle_len - 5:
        lr_phase = "near_trough_pre_restart"
    else:
        lr_phase = "cosine_decay_mid_cycle"
    ramp_len = int(round(cycle_len * beta_ratio_increase))
    beta_phase = "beta_ramp" if phase < ramp_len else "beta_plateau"
    return phase, lr_phase, beta_phase


def history_row(run_dir: Path, candidate_label: str, fold: int, max_epochs: int) -> dict[str, Any]:
    hist_path = run_dir / f"fold_{fold}" / f"vae_train_history_fold_{fold}.joblib"
    require(hist_path, f"{candidate_label} fold {fold} VAE history")
    hist = joblib.load(hist_path)
    val_modelsel = numeric_list(hist, "val_loss_modelsel")
    if val_modelsel.size == 0:
        val_modelsel = numeric_list(hist, "val_loss")
    best_epoch = int(np.nanargmin(val_modelsel))
    final_epoch = int(len(val_modelsel) - 1)
    epochs_after_best = final_epoch - best_epoch
    reached_max = bool(final_epoch >= max_epochs - 1)
    early_stop_epoch = np.nan if reached_max else final_epoch
    last_100 = val_modelsel[-100:] if len(val_modelsel) >= 100 else val_modelsel
    x = np.arange(len(last_100), dtype=float)
    slope = float(np.polyfit(x, last_100, 1)[0]) if len(last_100) >= 2 else np.nan

    def at(key: str) -> float:
        arr = np.asarray(hist.get(key, []), dtype=float)
        return float(arr[best_epoch]) if best_epoch < len(arr) else np.nan

    phase, lr_phase, beta_phase = phase_labels(best_epoch)
    return {
        "candidate_label": candidate_label,
        "fold": fold,
        "history_path": str(hist_path),
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "early_stop_epoch": early_stop_epoch,
        "epochs_after_best": epochs_after_best,
        "reached_max_epoch_4480": reached_max,
        "best_epoch_within_last_10pct_of_budget": bool(best_epoch >= int(0.9 * max_epochs)),
        "last_100_val_loss_modelsel_slope": slope,
        "best_beta": at("beta"),
        "cycle_phase_at_best_epoch": phase,
        "lr_phase_at_best_epoch": lr_phase,
        "beta_phase_at_best_epoch": beta_phase,
        "train_recon_at_best": at("train_recon"),
        "val_recon_at_best": at("val_recon"),
        "train_kld_at_best": at("train_kld"),
        "val_kld_at_best": at("val_kld"),
        "train_beta_kld_over_recon_at_best": at("train_beta_kld_over_recon"),
        "val_beta_kld_over_recon_at_best": at("val_beta_kld_over_recon"),
        "val_kld_over_recon_at_best": at("val_kld_over_recon"),
        "best_val_loss_modelsel": float(val_modelsel[best_epoch]),
        "last_val_loss_modelsel": float(val_modelsel[-1]),
    }


def stageb_metrics_for_run(run_dir: Path, readout_dir_name: str, candidate_label: str) -> pd.DataFrame:
    readout = run_dir / readout_dir_name
    path = readout / "classifier_sweep_foldwise_metrics.csv"
    require(path, f"{candidate_label} Stage B foldwise metrics")
    df = pd.read_csv(path)
    sub = df[(df["model_name"].eq("logreg_l2")) & (df["threshold_strategy"].eq(TARGET_STRATEGY))].copy()
    sub["candidate_label"] = candidate_label
    return sub


def latent_summary(branch_root: Path, candidate_label: str) -> pd.DataFrame:
    p = branch_root / "latent_information_summary.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    cand_id = "ch1_0_2_decoder_only_manufacturer" if candidate_label == "manufacturer_conditioned" else "ch1_0_2_baseline_unconditioned"
    sub = df[(df["candidate_id"].eq(cand_id)) & (df["latent_split"].eq("trainDev"))].copy()
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for fold, g in sub.groupby("fold"):
        row = {
            "candidate_label": candidate_label,
            "fold": int(fold),
            "active_units": float(g["n_active"].dropna().iloc[0]) if g["n_active"].notna().any() else np.nan,
            "total_correlation_nats": float(g["total_correlation_nats"].dropna().iloc[0]) if g["total_correlation_nats"].notna().any() else np.nan,
        }
        for var in ["Y_target", "Manufacturer", "Sex", "Age"]:
            gv = g[g["variable"].eq(var)]
            if not gv.empty:
                row[f"mi_sum_{var}"] = float(gv["mi_sum_nats"].iloc[0])
                row[f"mi_mean_{var}"] = float(gv["mi_mean_nats"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)


def training_completeness(args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for label, run_dir in candidate_runs(args.branch_root).items():
        cfg = load_json(run_dir / "run_config.json")
        max_epochs = int(cfg.get("args", {}).get("epochs_vae", cfg.get("epochs_vae", 4480)))
        for fold in range(1, 6):
            rows.append(history_row(run_dir, label, fold, max_epochs=max_epochs))
    out = pd.DataFrame(rows)
    metric_frames = []
    for label, run_dir in candidate_runs(args.branch_root).items():
        metric_frames.append(stageb_metrics_for_run(run_dir, "classifier_only_readout_z_plus_age_sex", label))
    metrics = pd.concat(metric_frames, ignore_index=True)
    metric_cols = [
        "candidate_label",
        "fold",
        "auc",
        "pr_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
    ]
    out = out.merge(metrics[metric_cols], on=["candidate_label", "fold"], how="left")
    lat = pd.concat(
        [latent_summary(args.branch_root, "matched_unconditioned_baseline"), latent_summary(args.branch_root, "manufacturer_conditioned")],
        ignore_index=True,
    )
    if not lat.empty:
        out = out.merge(lat, on=["candidate_label", "fold"], how="left")
    return out


def counts_by_group(df: pd.DataFrame, columns: Sequence[str]) -> dict[str, Any]:
    if df.empty:
        return {}
    g = df.groupby(list(columns), dropna=False).size().reset_index(name="n")
    return {"; ".join([str(row[c]) for c in columns]): int(row["n"]) for _, row in g.iterrows()}


def data_pool_audit(args: argparse.Namespace) -> pd.DataFrame:
    meta_path = args.branch_root / "metadata" / "training_ready_metadata_v5_1b_mfrrecovered035_clfpoollocked.csv"
    require(meta_path, "branch metadata")
    meta = pd.read_csv(meta_path)
    rows = []
    for label, run_dir in candidate_runs(args.branch_root).items():
        for fold in range(1, 6):
            fold_dir = run_dir / f"fold_{fold}"
            train = pd.read_csv(fold_dir / "train_dev_subjects_fold.csv")
            test = pd.read_csv(fold_dir / "test_subjects_fold.csv")
            pool_idx = np.load(fold_dir / "vae_training_pool_tensor_idx.npy")
            actual_local = np.load(fold_dir / "vae_actual_train_idx_local_to_pool.npy")
            val_local = np.load(fold_dir / "vae_internal_val_idx_local_to_pool.npy")
            pool = meta[meta["tensor_index"].isin(pool_idx)].copy()
            actual = meta[meta["tensor_index"].isin(pool_idx[actual_local])].copy()
            val = meta[meta["tensor_index"].isin(pool_idx[val_local])].copy()
            train_meta = train.merge(meta[["SubjectID", "Manufacturer", "Sex", "Age"]], on="SubjectID", how="left")
            test_meta = test.merge(meta[["SubjectID", "Manufacturer", "Sex", "Age"]], on="SubjectID", how="left")
            classifier_all = pd.concat([train.assign(split="train_dev"), test.assign(split="test")], ignore_index=True)
            rows.append(
                {
                    "candidate_label": label,
                    "fold": fold,
                    "vae_pool_n": int(len(pool)),
                    "vae_actual_train_n": int(len(actual)),
                    "vae_internal_val_n": int(len(val)),
                    "vae_pool_diagnosis_counts": json.dumps(pool["ResearchGroup_Mapped"].value_counts(dropna=False).to_dict(), sort_keys=True),
                    "vae_pool_manufacturer_counts": json.dumps(pool["Manufacturer"].value_counts(dropna=False).to_dict(), sort_keys=True),
                    "vae_actual_train_diagnosis_counts": json.dumps(actual["ResearchGroup_Mapped"].value_counts(dropna=False).to_dict(), sort_keys=True),
                    "vae_internal_val_diagnosis_counts": json.dumps(val["ResearchGroup_Mapped"].value_counts(dropna=False).to_dict(), sort_keys=True),
                    "classifier_train_dev_n": int(len(train)),
                    "classifier_test_n": int(len(test)),
                    "classifier_train_dev_diagnosis_counts": json.dumps(train["ResearchGroup_Mapped"].value_counts(dropna=False).to_dict(), sort_keys=True),
                    "classifier_test_diagnosis_counts": json.dumps(test["ResearchGroup_Mapped"].value_counts(dropna=False).to_dict(), sort_keys=True),
                    "classifier_train_dev_manufacturer_counts": json.dumps(train_meta["Manufacturer"].value_counts(dropna=False).to_dict(), sort_keys=True),
                    "classifier_test_manufacturer_counts": json.dumps(test_meta["Manufacturer"].value_counts(dropna=False).to_dict(), sort_keys=True),
                    "035_in_vae_pool": bool(pool["SubjectID"].eq("035_S_6927").any()),
                    "035_in_supervised_train_dev": bool(train["SubjectID"].eq("035_S_6927").any()),
                    "035_in_supervised_test": bool(test["SubjectID"].eq("035_S_6927").any()),
                    "128_in_vae_pool": bool(pool["SubjectID"].eq("128_S_2002").any()),
                    "128_in_supervised_train_dev": bool(train["SubjectID"].eq("128_S_2002").any()),
                    "128_in_supervised_test": bool(test["SubjectID"].eq("128_S_2002").any()),
                    "classifier_pool_n_this_fold": int(len(classifier_all)),
                    "classifier_pool_cn_this_fold": int((classifier_all["ResearchGroup_Mapped"] == "CN").sum()),
                    "classifier_pool_ad_this_fold": int((classifier_all["ResearchGroup_Mapped"] == "AD").sum()),
                    "oasis_used_in_training_or_thresholding": False,
                }
            )
    return pd.DataFrame(rows)


def parse_c(params: Any) -> float:
    try:
        data = json.loads(params) if isinstance(params, str) else params
        return float(data.get("model__C", np.nan))
    except Exception:
        return np.nan


def classifier_audit_for_run(run_dir: Path, readout_dir_name: str, candidate_label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    readout = run_dir / readout_dir_name
    status = pd.read_csv(readout / "classifier_sweep_model_status.csv")
    status = status[status["model_name"].eq("logreg_l2")].copy()
    status["candidate_label"] = candidate_label
    status["best_C"] = status["best_params"].map(parse_c)
    status["c_grid_min"] = min(MODEL_GRID_C_VALUES)
    status["c_grid_max"] = max(MODEL_GRID_C_VALUES)
    status["best_C_at_lower_boundary"] = np.isclose(status["best_C"], min(MODEL_GRID_C_VALUES))
    status["best_C_at_upper_boundary"] = np.isclose(status["best_C"], max(MODEL_GRID_C_VALUES))
    status["boundary_status"] = np.select(
        [status["best_C_at_lower_boundary"], status["best_C_at_upper_boundary"]],
        ["lower_boundary", "upper_boundary"],
        default="interior",
    )
    thresholds = pd.read_csv(readout / "classifier_sweep_thresholds_by_fold.csv")
    thresholds = thresholds[
        thresholds["model_name"].eq("logreg_l2") & thresholds["threshold_strategy"].eq(TARGET_STRATEGY)
    ].copy()
    thresholds["candidate_label"] = candidate_label
    return status, thresholds


def classifier_audit(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = []
    thr_frames = []
    for label, run_dir in candidate_runs(args.branch_root).items():
        s, t = classifier_audit_for_run(run_dir, "classifier_only_readout_z_plus_age_sex", label)
        frames.append(s)
        thr_frames.append(t)
    locked_s, locked_t = classifier_audit_for_run(args.locked_run, "classifier_only_readout", "locked_final_v5_1b")
    frames.append(locked_s)
    thr_frames.append(locked_t)
    return pd.concat(frames, ignore_index=True), pd.concat(thr_frames, ignore_index=True)


def oasis_metrics(args: argparse.Namespace) -> pd.DataFrame:
    frames = []
    secondary = args.oasis_secondary_dir / "primary_metrics.csv"
    if secondary.exists():
        df = pd.read_csv(secondary)
        df = df[df["prediction_level"].eq("ensemble_mean_score_majority_vote")].copy()
        df["oasis_source"] = "oasis_concatenated_and_runwise164_secondary_scoring"
        frames.append(df)
    tr140 = args.oasis_140tr_dir / "external_scoring_metrics.csv"
    if tr140.exists():
        df = pd.read_csv(tr140)
        df = df[df["prediction_level"].eq("ensemble_mean_score_majority_vote")].copy()
        df["oasis_source"] = "oasis_runwise140TR_sensitivity"
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    keep_models = [
        "locked_v5_1b_ch1_0_2_horizon4480",
        "secondary_manufacturer_conditioned_deconfounding",
    ]
    out = out[out["adni_model"].isin(keep_models)].copy()
    out["external_model_class"] = out["adni_model"].map(
        {
            "locked_v5_1b_ch1_0_2_horizon4480": "locked_final",
            "secondary_manufacturer_conditioned_deconfounding": "manufacturer_conditioned",
        }
    )
    return out


def oasis_stability(oasis: pd.DataFrame) -> pd.DataFrame:
    if oasis.empty:
        return oasis
    rows = []
    metrics = ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1"]
    for build, g in oasis.groupby("build_candidate", dropna=False):
        locked = g[g["external_model_class"].eq("locked_final")]
        mfr = g[g["external_model_class"].eq("manufacturer_conditioned")]
        if locked.empty or mfr.empty:
            continue
        lr = locked.iloc[0]
        mr = mfr.iloc[0]
        row = {"build_candidate": build, "mfr_advantage_stable_input": build}
        for metric in metrics:
            row[f"locked_{metric}"] = float(lr[metric])
            row[f"manufacturer_conditioned_{metric}"] = float(mr[metric])
            row[f"delta_mfr_minus_locked_{metric}"] = float(mr[metric]) - float(lr[metric])
        rows.append(row)
    return pd.DataFrame(rows)


def make_decision(training: pd.DataFrame, classifier: pd.DataFrame, oasis_delta: pd.DataFrame) -> tuple[str, list[str]]:
    mfr = training[training["candidate_label"].eq("manufacturer_conditioned")].copy()
    reached_max = int(mfr["reached_max_epoch_4480"].sum()) if not mfr.empty else 0
    late_best = int(mfr["best_epoch_within_last_10pct_of_budget"].sum()) if not mfr.empty else 0
    improving_late = int((mfr["last_100_val_loss_modelsel_slope"] < 0).sum()) if not mfr.empty else 0
    mfr_cls = classifier[classifier["candidate_label"].eq("manufacturer_conditioned")]
    lower = int(mfr_cls["best_C_at_lower_boundary"].sum()) if not mfr_cls.empty else 0
    upper = int(mfr_cls["best_C_at_upper_boundary"].sum()) if not mfr_cls.empty else 0
    stable_oasis = False
    if not oasis_delta.empty:
        # Stable means positive AUC and PR-AUC deltas across every available run handling.
        stable_oasis = bool(
            (oasis_delta["delta_mfr_minus_locked_auc"] > 0).all()
            and (oasis_delta["delta_mfr_minus_locked_pr_auc"] > 0).all()
        )

    reasons = [
        f"Manufacturer-conditioned folds reaching max epoch: {reached_max}/5.",
        f"Manufacturer-conditioned folds with best epoch in last 10% of 4480 budget: {late_best}/5.",
        f"Manufacturer-conditioned folds with negative last-100 ValL slope: {improving_late}/5.",
        f"logreg_l2 best C lower-bound hits: {lower}/5; upper-bound hits: {upper}/5.",
        f"OASIS manufacturer-conditioned advantage stable across available run handling: {stable_oasis}.",
    ]

    if reached_max > 0 or late_best >= 2:
        if stable_oasis:
            decision = "fold_specific_long_horizon_diagnostic"
            reasons.append("There is some horizon signal plus stable external sensitivity; start with a fold-specific diagnostic, not a full rerun.")
        else:
            decision = "fold_specific_long_horizon_diagnostic"
            reasons.append("There is possible maturity signal, but external advantage is not stable; only a fold-specific diagnostic would be defensible.")
    else:
        decision = "no_longer_rerun_justified"
        reasons.append("All folds early-stopped well before 4480 and best epochs were not right-censored; longer FULL rerun is not scientifically justified.")
    if lower == len(mfr_cls) and not stable_oasis:
        reasons.append("C lower-bound hits are noted, but prior ultra-regularization hurt test ranking; grid extension alone is not justified.")
    if decision != "no_longer_rerun_justified" and not stable_oasis and reached_max == 0 and late_best == 0:
        decision = "no_longer_rerun_justified"
    return decision, reasons


def write_reports(
    outdir: Path,
    training: pd.DataFrame,
    data: pd.DataFrame,
    classifier: pd.DataFrame,
    thresholds: pd.DataFrame,
    oasis: pd.DataFrame,
    oasis_delta: pd.DataFrame,
    decision: str,
    reasons: list[str],
) -> None:
    md = [
        "# Manufacturer-Conditioned FULL 5x5 Completeness Audit",
        "",
        f"Decision: `{decision}`",
        "",
        "## Decision Basis",
        "",
    ]
    md.extend([f"- {r}" for r in reasons])
    md += [
        "",
        "## Safety",
        "",
        "- No training was run.",
        "- No tensors, metadata, ledgers, configs, or model-output folders were modified.",
        "- OASIS data were used only for external metric summaries, not for training, thresholds, or model selection.",
        "",
        "## Key Tables",
        "",
        "- `fold_training_completeness.csv`",
        "- `data_pool_completeness.csv`",
        "- `classifier_hyperparameter_audit.csv`",
        "- `threshold_behavior_audit.csv`",
        "- `oasis_external_stability.csv`",
    ]
    (outdir / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    detail = [
        "# Long-Horizon Rerun Decision",
        "",
        f"Decision: `{decision}`",
        "",
        "The Manufacturer-conditioned FULL 5x5 run is evaluated against three requirements: training maturity, data completeness/leakage safety, and external-sensitivity stability.",
        "",
        "## Evidence",
        "",
    ]
    detail.extend([f"- {r}" for r in reasons])
    detail += [
        "",
        "## Interpretation",
        "",
        "A full long-horizon rerun is justified only if the model is plausibly right-censored and the signal is not just a threshold or external-pilot artifact. "
        "If folds early-stopped hundreds of epochs after their best validation loss, extending all folds would mainly add compute without targeting an observed failure mode.",
    ]
    (outdir / "long_horizon_rerun_decision.md").write_text("\n".join(detail) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    require(args.branch_root, "Manufacturer-conditioned branch root")
    require(args.locked_run, "locked final run")

    training = training_completeness(args)
    data = data_pool_audit(args)
    classifier, thresholds = classifier_audit(args)
    oasis = oasis_metrics(args)
    oasis_delta = oasis_stability(oasis)

    decision, reasons = make_decision(training, classifier, oasis_delta)

    write_csv_md(training, outdir / "fold_training_completeness.csv", outdir / "fold_training_completeness.md", "Fold Training Completeness")
    write_csv_md(data, outdir / "data_pool_completeness.csv", outdir / "data_pool_completeness.md", "Data Pool Completeness")
    write_csv_md(classifier, outdir / "classifier_hyperparameter_audit.csv", outdir / "classifier_hyperparameter_audit.md", "Classifier Hyperparameter Audit")
    write_csv_md(thresholds, outdir / "threshold_behavior_audit.csv", outdir / "threshold_behavior_audit.md", "Threshold Behavior Audit")
    write_csv_md(oasis, outdir / "oasis_external_metrics.csv", outdir / "oasis_external_metrics.md", "OASIS External Metrics")
    write_csv_md(oasis_delta, outdir / "oasis_external_stability.csv", outdir / "oasis_external_stability.md", "OASIS External Stability")
    write_reports(outdir, training, data, classifier, thresholds, oasis, oasis_delta, decision, reasons)

    write_json(
        outdir / "command_log.json",
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).resolve()),
            "mode": "read_only_audit",
            "decision": decision,
            "trained": False,
            "modified_tensors": False,
            "modified_metadata": False,
            "modified_ledger": False,
            "modified_model_outputs": False,
            "branch_root": str(args.branch_root),
            "locked_run": str(args.locked_run),
            "oasis_secondary_dir": str(args.oasis_secondary_dir),
            "oasis_140tr_dir": str(args.oasis_140tr_dir),
            "reasons": reasons,
        },
    )
    print(json.dumps({"output_dir": str(outdir), "decision": decision}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
