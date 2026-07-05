#!/usr/bin/env python3
"""Final read-only decision audit for FULL pair_ch1_3 beta=1.25.

No training, no classifier refit, no OASIS. The script reads an already
completed FULL run and its existing Stage B OOF-ECDF outputs.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


PROJECT = Path("/home/diego/proyectos/vae_AD")
RESULTS = PROJECT / "results" / "revision_bspc_2026"
OUT = RESULTS / "post_revision_exploratory_20260630" / "full_pair_ch13_beta1p25_final_decision_audit_20260705"

RUN = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/post_revision_exploratory_20260630/"
    "full_pair_ch13_beta1p25_channelmean_20260704"
)
STAGEB = RESULTS / "post_revision_exploratory_20260630" / "full_pair_ch13_beta1p25_channelmean_20260704_stageB_oof_ecdf"
LOCKED_STAGEB = RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration"
LOCKED_RUN = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/"
    "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
)

PRIMARY_MODEL = "logreg_l2_original"
PRIMARY_FEATURE = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"
PRIMARY_THRESH = "inner_oof_target_sens_ge_0p70_max_spec"

LOCKED_AUC = 0.795155
LOCKED_PR = 0.573934
LOCKED_N = 397

COMMAND_LOG: List[Dict[str, Any]] = []


def log(action: str, **kwargs: Any) -> None:
    COMMAND_LOG.append({"utc": datetime.now(timezone.utc).isoformat(), "action": action, **kwargs})


def read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    log("read_csv", path=str(path))
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path)


def md_table(df: pd.DataFrame, max_rows: int = 120) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6f}")
    out = view.to_markdown(index=False)
    if len(df) > max_rows:
        out += f"\n\n_Showing {max_rows} of {len(df)} rows._"
    return out + "\n"


def write_pair(stem: str, df: pd.DataFrame, max_rows: int = 120) -> None:
    df.to_csv(OUT / f"{stem}.csv", index=False)
    (OUT / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def primary_rows(pred: pd.DataFrame) -> pd.DataFrame:
    q = (
        pred["model_name"].eq(PRIMARY_MODEL)
        & pred["feature_set"].eq(PRIMARY_FEATURE)
        & pred["calib_method"].eq(PRIMARY_CALIB)
        & pred["threshold_strategy"].eq(PRIMARY_THRESH)
    )
    return pred.loc[q].copy()


def primary_metric_row(metrics: pd.DataFrame) -> pd.DataFrame:
    q = (
        metrics["model_name"].eq(PRIMARY_MODEL)
        & metrics["feature_set"].eq(PRIMARY_FEATURE)
        & metrics["calib_method"].eq(PRIMARY_CALIB)
        & metrics["threshold_strategy"].eq(PRIMARY_THRESH)
    )
    return metrics.loc[q].copy()


def subject_hash(ids: List[str]) -> str:
    payload = "\n".join(ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def fold_hash_audit(new_primary: pd.DataFrame, locked_primary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold in range(1, 6):
        n_ids = new_primary[new_primary["fold"].eq(fold)]["SubjectID"].astype(str).tolist()
        l_ids = locked_primary[locked_primary["fold"].eq(fold)]["SubjectID"].astype(str).tolist()
        rows.append(
            {
                "fold": fold,
                "new_n": len(n_ids),
                "locked_n": len(l_ids),
                "new_hash_ordered": subject_hash(n_ids),
                "locked_hash_ordered": subject_hash(l_ids),
                "ordered_equal": n_ids == l_ids,
                "set_equal": set(n_ids) == set(l_ids),
                "new_only": ",".join(sorted(set(n_ids) - set(l_ids))),
                "locked_only": ",".join(sorted(set(l_ids) - set(n_ids))),
            }
        )
    return pd.DataFrame(rows)


def integrity_audit(new_primary: pd.DataFrame, locked_primary: pd.DataFrame) -> pd.DataFrame:
    checks = []
    checkpoint_paths = [RUN / f"fold_{i}" / f"vae_model_fold_{i}.pt" for i in range(1, 6)]
    latent_paths = []
    for fold in range(1, 6):
        latent_paths.append(RUN / "classifier_only_readout" / "latent_cache" / f"fold_{fold}_test_latent_mu.csv")
        latent_paths.append(RUN / "classifier_only_readout" / "latent_cache" / f"fold_{fold}_trainDev_latent_mu.csv")
    checks.append({"check": "vae_checkpoints_present", "observed": sum(p.exists() for p in checkpoint_paths), "expected": 5, "pass": all(p.exists() for p in checkpoint_paths)})
    checks.append({"check": "latent_caches_present", "observed": sum(p.exists() for p in latent_paths), "expected": 10, "pass": all(p.exists() for p in latent_paths)})
    checks.append({"check": "primary_oof_rows", "observed": len(new_primary), "expected": LOCKED_N, "pass": len(new_primary) == LOCKED_N})
    checks.append({"check": "primary_oof_unique_subjects", "observed": new_primary["SubjectID"].nunique(), "expected": LOCKED_N, "pass": new_primary["SubjectID"].nunique() == LOCKED_N})
    checks.append({"check": "primary_oof_duplicates", "observed": int(new_primary["SubjectID"].duplicated().sum()), "expected": 0, "pass": not new_primary["SubjectID"].duplicated().any()})
    checks.append({"check": "locked_oof_rows", "observed": len(locked_primary), "expected": LOCKED_N, "pass": len(locked_primary) == LOCKED_N})
    fh = fold_hash_audit(new_primary, locked_primary)
    checks.append({"check": "fold_subject_hash_equality_all_folds", "observed": bool(fh["ordered_equal"].all()), "expected": True, "pass": bool(fh["ordered_equal"].all())})
    return pd.DataFrame(checks)


def bootstrap_vs_locked(new_primary: pd.DataFrame, locked_primary: pd.DataFrame, n_boot: int = 10000) -> pd.DataFrame:
    merged = new_primary[["SubjectID", "y_true", "y_score"]].rename(columns={"y_score": "new_score", "y_true": "new_y"}).merge(
        locked_primary[["SubjectID", "y_true", "y_score"]].rename(columns={"y_score": "locked_score", "y_true": "locked_y"}),
        on="SubjectID",
        how="inner",
    )
    if len(merged) != LOCKED_N:
        raise RuntimeError(f"paired bootstrap merge has {len(merged)} rows, expected {LOCKED_N}")
    if not np.array_equal(merged["new_y"].to_numpy(), merged["locked_y"].to_numpy()):
        raise RuntimeError("label mismatch in paired bootstrap merge")
    y = merged["new_y"].astype(int).to_numpy()
    s_new = merged["new_score"].astype(float).to_numpy()
    s_lock = merged["locked_score"].astype(float).to_numpy()
    rng = np.random.default_rng(20260705)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    d_auc, d_pr = [], []
    for _ in range(n_boot):
        idx = np.r_[rng.choice(pos, size=len(pos), replace=True), rng.choice(neg, size=len(neg), replace=True)]
        yy = y[idx]
        d_auc.append(roc_auc_score(yy, s_new[idx]) - roc_auc_score(yy, s_lock[idx]))
        d_pr.append(average_precision_score(yy, s_new[idx]) - average_precision_score(yy, s_lock[idx]))
    d_auc = np.asarray(d_auc)
    d_pr = np.asarray(d_pr)
    row = {
        "comparison": "pair_ch1_3_beta1p25_primary_minus_locked_primary",
        "n_subjects": int(len(merged)),
        "n_boot": n_boot,
        "new_auc": float(roc_auc_score(y, s_new)),
        "locked_auc": float(roc_auc_score(y, s_lock)),
        "delta_auc": float(roc_auc_score(y, s_new) - roc_auc_score(y, s_lock)),
        "delta_auc_ci_low": float(np.percentile(d_auc, 2.5)),
        "delta_auc_ci_high": float(np.percentile(d_auc, 97.5)),
        "p_delta_auc_gt0": float((d_auc > 0).mean()),
        "new_pr_auc": float(average_precision_score(y, s_new)),
        "locked_pr_auc": float(average_precision_score(y, s_lock)),
        "delta_pr_auc": float(average_precision_score(y, s_new) - average_precision_score(y, s_lock)),
        "delta_pr_auc_ci_low": float(np.percentile(d_pr, 2.5)),
        "delta_pr_auc_ci_high": float(np.percentile(d_pr, 97.5)),
        "p_delta_pr_auc_gt0": float((d_pr > 0).mean()),
    }
    return pd.DataFrame([row])


def best_exploratory(metrics: pd.DataFrame) -> pd.DataFrame:
    q = metrics["feature_set"].eq(PRIMARY_FEATURE) & metrics["threshold_strategy"].eq(PRIMARY_THRESH)
    return metrics.loc[q].sort_values(["auc", "pr_auc"], ascending=False).head(12).copy()


def foldwise_failure(new_fold: pd.DataFrame, locked_fold: pd.DataFrame, new_primary: pd.DataFrame, locked_primary: pd.DataFrame) -> pd.DataFrame:
    q_new = primary_metric_row(new_fold)
    q_lock = primary_metric_row(locked_fold)
    rows = []
    for fold in range(1, 6):
        nr = q_new[q_new["fold"].eq(fold)].iloc[0]
        lr = q_lock[q_lock["fold"].eq(fold)].iloc[0]
        # Pooled metrics with this fold removed.
        n_minus = new_primary[~new_primary["fold"].eq(fold)]
        l_minus = locked_primary[~locked_primary["fold"].eq(fold)]
        y_n = n_minus["y_true"].astype(int)
        y_l = l_minus["y_true"].astype(int)
        rows.append(
            {
                "fold": fold,
                "new_n": int(nr["n"]),
                "new_auc": float(nr["auc"]),
                "new_pr_auc": float(nr["pr_auc"]),
                "locked_auc": float(lr["auc"]),
                "locked_pr_auc": float(lr["pr_auc"]),
                "delta_auc_new_minus_locked": float(nr["auc"] - lr["auc"]),
                "delta_pr_auc_new_minus_locked": float(nr["pr_auc"] - lr["pr_auc"]),
                "new_pooled_auc_without_fold": float(roc_auc_score(y_n, n_minus["y_score"])),
                "new_pooled_pr_auc_without_fold": float(average_precision_score(y_n, n_minus["y_score"])),
                "locked_pooled_auc_without_fold": float(roc_auc_score(y_l, l_minus["y_score"])),
                "locked_pooled_pr_auc_without_fold": float(average_precision_score(y_l, l_minus["y_score"])),
                "fold4_flag": fold == 4,
            }
        )
    return pd.DataFrame(rows)


def checkpoint_rd_active_units() -> pd.DataFrame:
    rows = []
    for fold in range(1, 6):
        hist_path = RUN / f"fold_{fold}" / f"vae_train_history_fold_{fold}.joblib"
        rd_path = RUN / f"fold_{fold}" / f"fold_{fold}_rate_distortion.csv"
        train_info_path = RUN / f"fold_{fold}" / f"fold_{fold}_trainDev_latent_info_summary.csv"
        test_info_path = RUN / f"fold_{fold}" / f"fold_{fold}_test_latent_info_summary.csv"
        hist = joblib.load(hist_path)
        val_loss = np.asarray(hist["val_loss_modelsel"], dtype=float)
        beta = np.asarray(hist["beta"], dtype=float)
        selected_idx = int(np.nanargmin(val_loss))
        selected_epoch = selected_idx + 1
        selected_beta = float(beta[selected_idx])
        rd = read_csv(rd_path)
        rr = rd[rd["epoch"].astype(int).eq(selected_epoch)]
        if rr.empty:
            raise RuntimeError(f"fold {fold}: selected epoch {selected_epoch} missing in RD table")
        r = rr.iloc[0]
        train_info = read_csv(train_info_path)
        test_info = read_csv(test_info_path)
        def active(info: pd.DataFrame, var: str) -> float:
            sub = info[info["variable"].eq(var)]
            return float(sub.iloc[0]["n_active"]) if not sub.empty else np.nan
        rows.append(
            {
                "fold": fold,
                "history_file": str(hist_path),
                "selected_epoch_inferred": selected_epoch,
                "selected_beta_inferred": selected_beta,
                "best_val_loss_modelsel": float(val_loss[selected_idx]),
                "train_reconstruction_D": float(r["D_train"]),
                "train_kld_R_nats": float(r["R_train_nats"]),
                "train_rho_beta_kld_over_recon": selected_beta * float(r["R_train_nats"]) / float(r["D_train"]),
                "val_reconstruction_D": float(r["D_val"]),
                "val_kld_R_nats": float(r["R_val_nats"]),
                "val_rho_beta_kld_over_recon": selected_beta * float(r["R_val_nats"]) / float(r["D_val"]),
                "train_active_units_y": active(train_info, "Y_target"),
                "train_active_units_manufacturer": active(train_info, "Manufacturer"),
                "test_active_units_y": active(test_info, "Y_target"),
                "test_active_units_manufacturer": active(test_info, "Manufacturer"),
                "notes": "Selected epoch inferred as argmin val_loss_modelsel; no new checkpoint selection performed.",
            }
        )
    return pd.DataFrame(rows)


def manufacturer_leakage() -> pd.DataFrame:
    rows = []
    for fold in range(1, 6):
        for split, suffix in [("train_dev", "scanner_leakage"), ("test", "test_scanner_leakage")]:
            p = RUN / f"fold_{fold}" / f"fold_{fold}_{suffix}.csv"
            df = read_csv(p)
            for _, r in df.iterrows():
                rows.append(
                    {
                        "fold": fold,
                        "split": split,
                        "representation": r["representation"],
                        "balanced_accuracy_mean": r["balanced_accuracy_mean"],
                        "balanced_accuracy_std": r["balanced_accuracy_std"],
                        "n_classes": r["n_classes"],
                        "chance_level": r["chance_level"],
                        "n_samples": r["n_samples"],
                        "ordinary_accuracy": np.nan,
                        "macro_f1": np.nan,
                        "majority_baseline": np.nan,
                        "permutation_p_value": np.nan,
                        "source_file": str(p),
                        "notes": "Existing artifact reports balanced accuracy only; no leakage model refit performed.",
                    }
                )
    return pd.DataFrame(rows)


def philips_fpr_comparison(new_fpr: pd.DataFrame, locked_fpr: pd.DataFrame) -> pd.DataFrame:
    def row_for(df: pd.DataFrame, label: str) -> pd.DataFrame:
        q = (
            df["model_name"].eq(PRIMARY_MODEL)
            & df["feature_set"].eq(PRIMARY_FEATURE)
            & df["calib_method"].eq(PRIMARY_CALIB)
            & df["threshold_strategy"].eq(PRIMARY_THRESH)
            & df["manufacturer"].eq("Philips")
        )
        r = df.loc[q].iloc[0].copy()
        return pd.DataFrame(
            [
                {
                    "model": label,
                    "manufacturer": "Philips",
                    "n_cn_pooled": int(r["n_cn_pooled"]),
                    "fp_cn_pooled": int(r["fp_cn_pooled"]),
                    "fpr_cn_pooled": float(r["fpr_cn_pooled"]),
                    "specificity_cn_pooled": float(r["specificity_cn_pooled"]),
                    "policy": f"{PRIMARY_MODEL}/{PRIMARY_FEATURE}/{PRIMARY_CALIB}/{PRIMARY_THRESH}",
                }
            ]
        )

    out = pd.concat([row_for(new_fpr, "pair_ch1_3_beta1p25"), row_for(locked_fpr, "locked_ch102_beta3p75")], ignore_index=True)
    new = out[out["model"].eq("pair_ch1_3_beta1p25")].iloc[0]
    locked = out[out["model"].eq("locked_ch102_beta3p75")].iloc[0]
    out["delta_fpr_vs_locked"] = out["fpr_cn_pooled"] - float(locked["fpr_cn_pooled"])
    out["delta_fp_count_vs_locked"] = out["fp_cn_pooled"] - int(locked["fp_cn_pooled"])
    return out


def report_label_correction(final_report: Path) -> str:
    text = final_report.read_text(encoding="utf-8")
    flags = {
        "has_inherited_beta3p75_title": "recover035_latent384_beta3p75_T80_h10000_p560_full5x5" in text,
        "has_obsolete_auc_ref_0p782951": "0.782951" in text,
        "has_obsolete_pr_ref_0p559873": "0.559873" in text,
    }
    return f"""# Stage B Final Report Label Correction

Audited file:

`{final_report}`

## Flags

| check | value |
|---|---:|
| inherited beta3p75 title present | {flags['has_inherited_beta3p75_title']} |
| obsolete AUC reference 0.782951 present | {flags['has_obsolete_auc_ref_0p782951']} |
| obsolete PR-AUC reference 0.559873 present | {flags['has_obsolete_pr_ref_0p559873']} |

## Correction

The file is a calibration-output report generated from inherited script text.
For the current decision, use the actual run identity
`full_pair_ch13_beta1p25_channelmean_20260704` and the locked manuscript
reference ROC-AUC={LOCKED_AUC:.6f}, PR-AUC={LOCKED_PR:.6f}, N={LOCKED_N}.

Do not use the report's inherited beta3p75 title or its old promotion reference
0.782951 / 0.559873 for the current promotion decision.
"""


def final_decision(primary: pd.DataFrame, boot: pd.DataFrame, best: pd.DataFrame, fpr: pd.DataFrame, rd: pd.DataFrame) -> str:
    p = primary.iloc[0]
    b = boot.iloc[0]
    best_row = best.iloc[0]
    new_fpr = fpr[fpr["model"].eq("pair_ch1_3_beta1p25")].iloc[0]
    locked_fpr = fpr[fpr["model"].eq("locked_ch102_beta3p75")].iloc[0]
    promote = (
        float(p["auc"]) >= LOCKED_AUC
        and float(p["pr_auc"]) >= LOCKED_PR
        and float(new_fpr["fpr_cn_pooled"]) <= float(locked_fpr["fpr_cn_pooled"])
    )
    decision = "reject_do_not_promote" if not promote else "candidate_requires_external_validation_before_any_replacement"
    return f"""# Final Promotion Decision

Decision: **{decision}**

## Primary Endpoint

Primary convention:
`{PRIMARY_MODEL} / {PRIMARY_FEATURE} / {PRIMARY_CALIB} / {PRIMARY_THRESH}`

| model | N | ROC-AUC | PR-AUC | BA | Sens | Spec | F1 | Philips CN FPR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pair_ch1_3 beta=1.25 | {int(p['n'])} | {float(p['auc']):.6f} | {float(p['pr_auc']):.6f} | {float(p['balanced_accuracy']):.6f} | {float(p['sensitivity']):.6f} | {float(p['specificity']):.6f} | {float(p['f1']):.6f} | {float(new_fpr['fpr_cn_pooled']):.4f} |
| locked [1,0,2] beta=3.75 | {LOCKED_N} | {LOCKED_AUC:.6f} | {LOCKED_PR:.6f} | reference | reference | reference | reference | {float(locked_fpr['fpr_cn_pooled']):.4f} |

## Paired Bootstrap vs Locked

- ΔROC-AUC = {float(b['delta_auc']):.6f}, 95% CI [{float(b['delta_auc_ci_low']):.6f}, {float(b['delta_auc_ci_high']):.6f}], P(Δ>0)={float(b['p_delta_auc_gt0']):.4f}
- ΔPR-AUC = {float(b['delta_pr_auc']):.6f}, 95% CI [{float(b['delta_pr_auc_ci_low']):.6f}, {float(b['delta_pr_auc_ci_high']):.6f}], P(Δ>0)={float(b['p_delta_pr_auc_gt0']):.4f}

## Best Exploratory Calibration

Best exploratory row in this Stage B package:
`{best_row['model_name']} / {best_row['feature_set']} / {best_row['calib_method']} / {best_row['threshold_strategy']}`
with ROC-AUC={float(best_row['auc']):.6f}, PR-AUC={float(best_row['pr_auc']):.6f}.
This is reported separately and does not replace the primary endpoint.

## Interpretation

The beta=1.25 `[1,3]` FULL run underperforms the locked model on both primary
ROC-AUC and PR-AUC. Philips CN FPR is also slightly worse than locked
({float(new_fpr['fpr_cn_pooled']):.4f} vs {float(locked_fpr['fpr_cn_pooled']):.4f}).
Fold-wise analysis should be interpreted as diagnostic only; FAST results are
not reinterpreted as FULL evidence. No OASIS or manuscript update is warranted
from this run.
"""


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    log("start", output_dir=str(OUT), guardrails="read_only_no_training_no_refit_no_oasis")

    new_pooled = read_csv(STAGEB / "calib_pooled_metrics.csv")
    new_fold = read_csv(STAGEB / "calib_foldwise_metrics.csv")
    new_pred = read_csv(STAGEB / "calib_predictions.csv")
    new_fpr = read_csv(STAGEB / "calib_philips_fpr_pooled.csv")
    locked_pooled = read_csv(LOCKED_STAGEB / "calib_pooled_metrics.csv")
    locked_fold = read_csv(LOCKED_STAGEB / "calib_foldwise_metrics.csv")
    locked_pred = read_csv(LOCKED_STAGEB / "calib_predictions.csv")
    locked_fpr = read_csv(LOCKED_STAGEB / "calib_philips_fpr_pooled.csv")

    new_primary = primary_rows(new_pred)
    locked_primary = primary_rows(locked_pred)
    primary = primary_metric_row(new_pooled)
    if len(primary) != 1:
        raise RuntimeError(f"primary metric row count {len(primary)}")
    primary = primary.copy()
    primary["locked_auc_reference"] = LOCKED_AUC
    primary["locked_pr_auc_reference"] = LOCKED_PR
    primary["delta_auc_vs_locked"] = primary["auc"].astype(float) - LOCKED_AUC
    primary["delta_pr_auc_vs_locked"] = primary["pr_auc"].astype(float) - LOCKED_PR
    write_pair("primary_pooled_metrics", primary)

    integrity = integrity_audit(new_primary, locked_primary)
    fold_hash = fold_hash_audit(new_primary, locked_primary)
    integrity_text = "# Post-run Integrity Audit\n\n" + md_table(integrity) + "\n## Fold Subject Hashes\n\n" + md_table(fold_hash)
    (OUT / "postrun_integrity.md").write_text(integrity_text, encoding="utf-8")

    boot = bootstrap_vs_locked(new_primary, locked_primary)
    write_pair("paired_bootstrap_vs_locked", boot)

    fold_fail = foldwise_failure(new_fold, locked_fold, new_primary, locked_primary)
    write_pair("foldwise_failure_analysis", fold_fail)

    rd = checkpoint_rd_active_units()
    write_pair("rate_distortion_active_units", rd)

    leak = manufacturer_leakage()
    write_pair("manufacturer_leakage", leak, max_rows=240)

    fpr_cmp = philips_fpr_comparison(new_fpr, locked_fpr)
    write_pair("philips_cn_fpr_comparison", fpr_cmp)

    best = best_exploratory(new_pooled)
    best.to_csv(OUT / "best_exploratory_calibration.csv", index=False)

    (OUT / "report_label_correction.md").write_text(
        report_label_correction(STAGEB / "final_report.md"), encoding="utf-8"
    )
    (OUT / "final_promotion_decision.md").write_text(
        final_decision(primary, boot, best, fpr_cmp, rd), encoding="utf-8"
    )

    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).resolve()),
        "inputs": {
            "run": str(RUN),
            "stageb": str(STAGEB),
            "locked_stageb": str(LOCKED_STAGEB),
            "locked_run": str(LOCKED_RUN),
        },
        "guardrails": {
            "did_train_vae": False,
            "did_refit_classifier": False,
            "did_refit_calibration": False,
            "did_run_oasis": False,
            "did_modify_existing_outputs": False,
            "did_modify_manuscript": False,
        },
        "steps": COMMAND_LOG,
    }
    (OUT / "command_log.json").write_text(json.dumps(command_log, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    required = [
        "postrun_integrity.md",
        "primary_pooled_metrics.csv",
        "primary_pooled_metrics.md",
        "paired_bootstrap_vs_locked.csv",
        "paired_bootstrap_vs_locked.md",
        "foldwise_failure_analysis.csv",
        "foldwise_failure_analysis.md",
        "rate_distortion_active_units.csv",
        "rate_distortion_active_units.md",
        "manufacturer_leakage.csv",
        "manufacturer_leakage.md",
        "philips_cn_fpr_comparison.csv",
        "philips_cn_fpr_comparison.md",
        "report_label_correction.md",
        "final_promotion_decision.md",
        "command_log.json",
    ]
    missing = [x for x in required if not (OUT / x).exists() or (OUT / x).stat().st_size == 0]
    if missing:
        raise RuntimeError(f"missing/empty outputs: {missing}")
    print(f"Wrote final decision audit to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
