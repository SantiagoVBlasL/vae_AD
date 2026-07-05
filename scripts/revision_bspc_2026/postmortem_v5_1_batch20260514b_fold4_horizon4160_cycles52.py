#!/usr/bin/env python3
"""Read-only postmortem for the Fold-4 horizon4160/cycles52 diagnostic."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOCKED_RUN = PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_ch1_0_2_mfrsplit_3840_final_candidate"
LOCKED_STAGE_B = PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep"
CANDIDATE_RUN = PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_ch1_0_2_fold4_horizon4160_cycles52"
CANDIDATE_STAGE_B = CANDIDATE_RUN / "classifier_only_readout"
OUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/adni_v5_1_batch20260514b_fold4_horizon4160_cycles52_postmortem"

FOLD = 4
PRIMARY_MODEL = "logreg_l2"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"


def md_table(df: pd.DataFrame, path: Path) -> None:
    text = "_No rows._" if df.empty else df.to_markdown(index=False, floatfmt=".6g")
    path.write_text(text + "\n", encoding="utf-8")


def load_history(run_dir: Path, run_id: str) -> dict[str, Any]:
    path = run_dir / f"fold_{FOLD}" / f"vae_train_history_fold_{FOLD}.joblib"
    if not path.exists():
        raise FileNotFoundError(path)
    obj = joblib.load(path)
    hist = obj.copy() if isinstance(obj, pd.DataFrame) else pd.DataFrame(obj)
    hist = hist.reset_index(drop=True)
    hist.insert(0, "epoch", np.arange(1, len(hist) + 1))
    best_idx = int(hist["val_loss_modelsel"].astype(float).idxmin())
    best = hist.iloc[best_idx]
    train_recon = float(best.get("train_recon", np.nan))
    val_recon = float(best.get("val_recon", np.nan))
    train_kld = float(best.get("train_kld", np.nan))
    val_kld = float(best.get("val_kld", np.nan))
    return {
        "run_id": run_id,
        "fold": FOLD,
        "best_epoch": int(best["epoch"]),
        "final_epoch_or_early_stop_epoch": int(hist["epoch"].iloc[-1]),
        "stopped_by_epoch_budget": int(hist["epoch"].iloc[-1]) >= (4160 if "horizon" in run_id else 3840),
        "epochs_after_best": int(hist["epoch"].iloc[-1] - best["epoch"]),
        "best_val_loss_beta_max": float(best["val_loss_modelsel"]),
        "train_recon_at_best": train_recon,
        "val_recon_at_best": val_recon,
        "val_minus_train_recon_at_best": val_recon - train_recon,
        "train_kld_at_best": train_kld,
        "val_kld_at_best": val_kld,
        "train_kld_over_recon_at_best": train_kld / train_recon if train_recon else np.nan,
        "val_kld_over_recon_at_best": val_kld / val_recon if val_recon else np.nan,
        "beta_at_best_epoch": float(best.get("beta", np.nan)),
    }


def load_stage_a(run_dir: Path, run_id: str) -> pd.DataFrame:
    paths = sorted(run_dir.glob("all_folds_metrics_MULTI_*.csv"))
    if not paths:
        return pd.DataFrame()
    df = pd.read_csv(paths[0])
    df = df[df["fold"].astype(int).eq(FOLD)].copy()
    if df.empty:
        return df
    df.insert(0, "run_id", run_id)
    keep = [
        "run_id",
        "fold",
        "actual_classifier_type",
        "auc",
        "pr_auc",
        "auc_raw",
        "pr_auc_raw",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1_score",
    ]
    return df[[c for c in keep if c in df.columns]]


def load_stage_b(stage_b_dir: Path, run_id: str) -> pd.DataFrame:
    path = stage_b_dir / "classifier_sweep_foldwise_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    mask = (
        df["fold"].astype(int).eq(FOLD)
        & df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    )
    out = df[mask].copy()
    out.insert(0, "run_id", run_id)
    keep = [
        "run_id",
        "fold",
        "model_name",
        "threshold_strategy",
        "threshold",
        "auc",
        "pr_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "tn",
        "fp",
        "fn",
        "tp",
        "accuracy",
        "predicted_ad_rate",
        "inner_oof_sensitivity",
        "inner_oof_specificity",
        "inner_oof_balanced_accuracy",
    ]
    return out[[c for c in keep if c in out.columns]]


def load_scanner(run_dir: Path, run_id: str) -> pd.DataFrame:
    rows = []
    for split, name in [("trainDev", f"fold_{FOLD}_scanner_leakage_summary.csv"), ("test", f"fold_{FOLD}_test_scanner_leakage_summary.csv")]:
        path = run_dir / f"fold_{FOLD}" / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        row["run_id"] = run_id
        row["fold"] = FOLD
        row["split"] = split
        if pd.notna(row.get("acc_site_raw")) and pd.notna(row.get("acc_site_latent")):
            row["latent_minus_raw_site_acc"] = row["acc_site_latent"] - row["acc_site_raw"]
        rows.append(row)
    return pd.DataFrame(rows)


def load_latent_info(run_dir: Path, run_id: str) -> pd.DataFrame:
    rows = []
    for split in ["trainDev", "test"]:
        path = run_dir / f"fold_{FOLD}" / f"fold_{FOLD}_{split}_latent_info_summary.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        df = df.copy()
        df.insert(0, "run_id", run_id)
        df.insert(1, "fold", FOLD)
        df.insert(2, "split", split)
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def confirm_only_fold4(candidate_run: Path) -> dict[str, Any]:
    fold_dirs = sorted(p.name for p in candidate_run.glob("fold_*") if p.is_dir())
    top_metrics = sorted(candidate_run.glob("all_folds_metrics_MULTI_*.csv"))
    metric_folds: list[int] = []
    if top_metrics:
        df = pd.read_csv(top_metrics[0])
        metric_folds = sorted(df["fold"].astype(int).unique().tolist())
    stage_b_path = CANDIDATE_STAGE_B / "classifier_sweep_foldwise_metrics.csv"
    stage_b_folds: list[int] = []
    if stage_b_path.exists():
        df = pd.read_csv(stage_b_path)
        stage_b_folds = sorted(df["fold"].astype(int).unique().tolist())
    latent_cache_folds = sorted(
        {
            int(p.name.split("_")[1])
            for p in (CANDIDATE_STAGE_B / "latent_cache").glob("fold_*_*_latent_mu.csv")
            if len(p.name.split("_")) > 2 and p.name.split("_")[1].isdigit()
        }
    )
    ok = fold_dirs == ["fold_4"] and metric_folds == [4] and (not stage_b_folds or stage_b_folds == [4]) and (not latent_cache_folds or latent_cache_folds == [4])
    return {
        "only_fold4_confirmed": ok,
        "top_level_fold_dirs": ",".join(fold_dirs),
        "stage_a_metric_folds": ",".join(map(str, metric_folds)),
        "stage_b_metric_folds": ",".join(map(str, stage_b_folds)),
        "stage_b_latent_cache_folds": ",".join(map(str, latent_cache_folds)),
    }


def add_delta(candidate: pd.Series, locked: pd.Series, columns: list[str]) -> dict[str, Any]:
    out = {}
    for col in columns:
        if col in candidate.index and col in locked.index and pd.notna(candidate[col]) and pd.notna(locked[col]):
            out[f"delta_{col}_vs_locked"] = candidate[col] - locked[col]
    return out


def build_main_table(
    dynamics: pd.DataFrame,
    stage_a: pd.DataFrame,
    stage_b: pd.DataFrame,
    scanner: pd.DataFrame,
    latent: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for run_id in ["locked_current_fold4", "horizon4160_cycles52_fold4"]:
        dyn = dynamics[dynamics["run_id"].eq(run_id)].iloc[0].to_dict()
        sb = stage_b[stage_b["run_id"].eq(run_id)].iloc[0].to_dict() if not stage_b[stage_b["run_id"].eq(run_id)].empty else {}
        logreg = stage_a[(stage_a["run_id"].eq(run_id)) & (stage_a["actual_classifier_type"].eq("logreg"))]
        svm = stage_a[(stage_a["run_id"].eq(run_id)) & (stage_a["actual_classifier_type"].eq("svm"))]
        test_scanner = scanner[(scanner["run_id"].eq(run_id)) & (scanner["split"].eq("test"))]
        latent_y = latent[(latent["run_id"].eq(run_id)) & (latent["split"].eq("trainDev")) & (latent["variable"].astype(str).eq("Y_target"))]
        row = {
            "run_id": run_id,
            "fold": FOLD,
            "best_epoch": dyn.get("best_epoch"),
            "final_epoch_or_early_stop_epoch": dyn.get("final_epoch_or_early_stop_epoch"),
            "epochs_after_best": dyn.get("epochs_after_best"),
            "best_val_loss_beta_max": dyn.get("best_val_loss_beta_max"),
            "stage_a_logreg_auc": logreg["auc"].iloc[0] if not logreg.empty else np.nan,
            "stage_a_logreg_pr_auc": logreg["pr_auc"].iloc[0] if not logreg.empty else np.nan,
            "stage_a_svm_auc": svm["auc"].iloc[0] if not svm.empty else np.nan,
            "stage_a_svm_pr_auc": svm["pr_auc"].iloc[0] if not svm.empty else np.nan,
            "stage_b_auc": sb.get("auc", np.nan),
            "stage_b_pr_auc": sb.get("pr_auc", np.nan),
            "stage_b_ba": sb.get("balanced_accuracy", np.nan),
            "stage_b_sens": sb.get("sensitivity", np.nan),
            "stage_b_spec": sb.get("specificity", np.nan),
            "stage_b_f1": sb.get("f1", np.nan),
            "stage_b_threshold": sb.get("threshold", np.nan),
            "tn": sb.get("tn", np.nan),
            "fp": sb.get("fp", np.nan),
            "fn": sb.get("fn", np.nan),
            "tp": sb.get("tp", np.nan),
            "test_acc_site_raw": test_scanner["acc_site_raw"].iloc[0] if not test_scanner.empty else np.nan,
            "test_acc_site_latent": test_scanner["acc_site_latent"].iloc[0] if not test_scanner.empty else np.nan,
            "test_latent_minus_raw_site_acc": test_scanner["latent_minus_raw_site_acc"].iloc[0] if not test_scanner.empty else np.nan,
            "trainDev_active_units": latent_y["n_active"].iloc[0] if not latent_y.empty else np.nan,
            "trainDev_total_correlation_nats": latent_y["total_correlation_nats"].iloc[0] if not latent_y.empty else np.nan,
            "trainDev_y_mi_sum_nats": latent_y["mi_sum_nats"].iloc[0] if not latent_y.empty else np.nan,
            "val_kld_at_best": dyn.get("val_kld_at_best"),
            "val_kld_over_recon_at_best": dyn.get("val_kld_over_recon_at_best"),
        }
        rows.append(row)
    table = pd.DataFrame(rows)
    if len(table) == 2:
        locked = table[table["run_id"].eq("locked_current_fold4")].iloc[0]
        cand_idx = table[table["run_id"].eq("horizon4160_cycles52_fold4")].index[0]
        deltas = add_delta(
            table.loc[cand_idx],
            locked,
            [
                "best_epoch",
                "best_val_loss_beta_max",
                "stage_a_logreg_auc",
                "stage_a_logreg_pr_auc",
                "stage_a_svm_auc",
                "stage_a_svm_pr_auc",
                "stage_b_auc",
                "stage_b_pr_auc",
                "stage_b_ba",
                "stage_b_sens",
                "stage_b_spec",
                "stage_b_f1",
                "test_acc_site_latent",
                "test_latent_minus_raw_site_acc",
                "trainDev_total_correlation_nats",
                "val_kld_over_recon_at_best",
            ],
        )
        for key, value in deltas.items():
            table.loc[cand_idx, key] = value
    return table


def write_recommendation(main: pd.DataFrame, only_fold4: dict[str, Any]) -> None:
    locked = main[main["run_id"].eq("locked_current_fold4")].iloc[0]
    cand = main[main["run_id"].eq("horizon4160_cycles52_fold4")].iloc[0]
    auc_delta = cand.get("delta_stage_b_auc_vs_locked", np.nan)
    pr_delta = cand.get("delta_stage_b_pr_auc_vs_locked", np.nan)
    val_delta = cand.get("delta_best_val_loss_beta_max_vs_locked", np.nan)
    leakage_delta = cand.get("delta_test_acc_site_latent_vs_locked", np.nan)
    horizon_used = bool(cand["final_epoch_or_early_stop_epoch"] > 3840 or cand["best_epoch"] > 3840)
    improved_ranking = pd.notna(auc_delta) and pd.notna(pr_delta) and auc_delta > 0 and pr_delta >= 0
    improved_vae = pd.notna(val_delta) and val_delta < 0
    leakage_ok = pd.isna(leakage_delta) or leakage_delta <= 0.02
    justify_full = bool(only_fold4["only_fold4_confirmed"] and improved_ranking and improved_vae and leakage_ok and horizon_used)
    decision = (
        "A FULL 5x5 horizon4160_cycles52 run is justified as a targeted confirmation."
        if justify_full
        else "Do not launch a FULL 5x5 horizon4160_cycles52 run from this diagnostic."
    )
    lines = [
        "# Fold-4 Horizon4160/Cycles52 Postmortem Recommendation",
        "",
        f"- Only Fold 4 confirmed: `{only_fold4['only_fold4_confirmed']}`.",
        f"- Locked Fold 4 Stage B AUC/PR-AUC: `{locked['stage_b_auc']:.6f}` / `{locked['stage_b_pr_auc']:.6f}`.",
        f"- Horizon Fold 4 Stage B AUC/PR-AUC: `{cand['stage_b_auc']:.6f}` / `{cand['stage_b_pr_auc']:.6f}`.",
        f"- Delta Stage B AUC/PR-AUC: `{auc_delta:.6f}` / `{pr_delta:.6f}`.",
        f"- Horizon candidate best/final epoch: `{int(cand['best_epoch'])}` / `{int(cand['final_epoch_or_early_stop_epoch'])}`.",
        f"- Extra horizon actually used beyond locked 3840 epochs: `{horizon_used}`.",
        f"- Delta best ValL(beta_max): `{val_delta:.6f}`.",
        f"- Delta test latent manufacturer leakage acc: `{leakage_delta:.6f}`.",
        "",
        f"**Decision: {decision}**",
        "",
        "Interpretation:",
        "- This is a one-fold diagnostic, not a replacement candidate for the locked manuscript model.",
        "- The candidate improved Fold-4 ranking, but it did not use the added 3840->4160 horizon: best epoch was 2707 and early stop/final epoch was 3027. The improvement is therefore better interpreted as stochastic rerun/checkpoint variability than as evidence that a longer horizon is needed.",
        "- Test latent manufacturer leakage worsened by the diagnostic rule, so the one-fold improvement is not clean enough to justify a new full run.",
        "- A follow-up FULL run would require consistent improvement in fold-4 ranking, VAE model-selection loss, and no meaningful scanner/manufacturer leakage worsening.",
        "- If the diagnostic improves threshold metrics but not AUC/PR-AUC, it should be treated as operating-point movement rather than a representation improvement.",
        "",
        "Safety: this postmortem was read-only; it did not train or modify tensor, metadata, ledger, config, or existing model-output files.",
    ]
    (OUT_DIR / "recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    required = [
        LOCKED_RUN / f"fold_{FOLD}" / f"vae_train_history_fold_{FOLD}.joblib",
        CANDIDATE_RUN / f"fold_{FOLD}" / f"vae_train_history_fold_{FOLD}.joblib",
        LOCKED_STAGE_B / "classifier_sweep_foldwise_metrics.csv",
        CANDIDATE_STAGE_B / "classifier_sweep_foldwise_metrics.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required artifacts:\n" + "\n".join(missing))

    only_fold4 = confirm_only_fold4(CANDIDATE_RUN)
    dynamics = pd.DataFrame(
        [
            load_history(LOCKED_RUN, "locked_current_fold4"),
            load_history(CANDIDATE_RUN, "horizon4160_cycles52_fold4"),
        ]
    )
    stage_a = pd.concat(
        [
            load_stage_a(LOCKED_RUN, "locked_current_fold4"),
            load_stage_a(CANDIDATE_RUN, "horizon4160_cycles52_fold4"),
        ],
        ignore_index=True,
    )
    stage_b = pd.concat(
        [
            load_stage_b(LOCKED_STAGE_B, "locked_current_fold4"),
            load_stage_b(CANDIDATE_STAGE_B, "horizon4160_cycles52_fold4"),
        ],
        ignore_index=True,
    )
    scanner = pd.concat(
        [
            load_scanner(LOCKED_RUN, "locked_current_fold4"),
            load_scanner(CANDIDATE_RUN, "horizon4160_cycles52_fold4"),
        ],
        ignore_index=True,
    )
    latent = pd.concat(
        [
            load_latent_info(LOCKED_RUN, "locked_current_fold4"),
            load_latent_info(CANDIDATE_RUN, "horizon4160_cycles52_fold4"),
        ],
        ignore_index=True,
    )
    main = build_main_table(dynamics, stage_a, stage_b, scanner, latent)
    only_fold4_df = pd.DataFrame([only_fold4])
    main = pd.concat([main.assign(only_fold4_confirmed=only_fold4["only_fold4_confirmed"])], ignore_index=True)

    main.to_csv(OUT_DIR / "fold4_comparison_table.csv", index=False)
    stage_b.to_csv(OUT_DIR / "stage_b_fold4_comparison.csv", index=False)
    dynamics.to_csv(OUT_DIR / "training_dynamics_comparison.csv", index=False)
    only_fold4_df.to_csv(OUT_DIR / "fold4_run_scope_confirmation.csv", index=False)
    stage_a.to_csv(OUT_DIR / "stage_a_fold4_comparison.csv", index=False)
    scanner.to_csv(OUT_DIR / "manufacturer_leakage_fold4_comparison.csv", index=False)
    latent.to_csv(OUT_DIR / "latent_info_fold4_comparison.csv", index=False)
    md_table(main, OUT_DIR / "fold4_comparison_table.md")
    md_table(stage_b, OUT_DIR / "stage_b_fold4_comparison.md")
    md_table(dynamics, OUT_DIR / "training_dynamics_comparison.md")
    md_table(only_fold4_df, OUT_DIR / "fold4_run_scope_confirmation.md")
    md_table(stage_a, OUT_DIR / "stage_a_fold4_comparison.md")
    md_table(scanner, OUT_DIR / "manufacturer_leakage_fold4_comparison.md")
    md_table(latent, OUT_DIR / "latent_info_fold4_comparison.md")
    write_recommendation(main, only_fold4)
    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "candidate_run": str(CANDIDATE_RUN),
        "locked_run": str(LOCKED_RUN),
        "locked_stage_b": str(LOCKED_STAGE_B),
        "candidate_stage_b": str(CANDIDATE_STAGE_B),
        "target_fold": FOLD,
        "only_fold4_confirmed": only_fold4["only_fold4_confirmed"],
        "output_dir": str(OUT_DIR),
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "ledger_modified": False,
        "config_modified": False,
        "existing_model_output_modified": False,
    }
    (OUT_DIR / "command_log.json").write_text(json.dumps(command_log, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote Fold-4 horizon postmortem to {OUT_DIR}")
    print(main[["run_id", "best_epoch", "final_epoch_or_early_stop_epoch", "best_val_loss_beta_max", "stage_b_auc", "stage_b_pr_auc", "stage_b_ba", "stage_b_f1"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
