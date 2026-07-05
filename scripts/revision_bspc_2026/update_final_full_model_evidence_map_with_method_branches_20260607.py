#!/usr/bin/env python3
"""Update the final FULL-model evidence map with completed method branches.

Adds:
  1. chmeanloss [1,0,2] latent384 beta3.75
  2. foldcombat_mfr_age_sex [1,0,2] latent384 beta3.75

Reads completed Stage B + OOF artifacts only. No training, scoring, threshold
fitting, calibration fitting, tensor edits, metadata edits, or model artifact
edits are performed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path("results/revision_bspc_2026")
BASE_DIR = ROOT / "final_full_model_evidence_map_with_beta6p5_20260606"
BASE_MAP = BASE_DIR / "full_model_evidence_map.csv"
OUT_DIR = ROOT / "final_full_model_evidence_map_with_method_branches_20260607"

CHMEAN_AUDIT = ROOT / "chmeanloss_stageB_oof_completion_audit_20260607"
FOLDCOMBAT_AUDIT = ROOT / "foldcombat_stageB_oof_completion_audit_20260607"

PRIMARY_MODEL = "logreg_l2_original"
PRIMARY_FEATURE = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"

CHANNEL_NAMES = (
    '["Pearson_Full_FisherZ_Signed", '
    '"Pearson_OMST_GCE_Signed_Weighted", "MI_KNN_Symmetric"]'
)

OOF_PRED_DIRS = {
    "promoted_latent384_beta3p75_ch1_0_2": ROOT / "recover035_latent384_beta3p75_stageB_oof_score_calibration",
    "ch1only_latent384_beta3p75": ROOT / "recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5_stageB_oof_score_calibration",
    "latent384_beta6p5": ROOT / "recover035_latent384_beta6p5_stageB_oof_score_calibration",
    "latent448_beta4p0": ROOT / "recover035_latent448_beta4p0_stageB_oof_score_calibration",
    "latent512_beta3p75": ROOT / "recover035_latent512_beta3p75_stageB_oof_score_calibration",
    "chmeanloss_latent384_beta3p75_stageB_oof_ecdf": ROOT / "recover035_latent384_beta3p75_chmeanloss_stageB_oof_score_calibration",
    "foldcombat_mfr_age_sex_latent384_beta3p75_stageB_oof_ecdf": ROOT / "recover035_latent384_beta3p75_foldcombat_stageB_oof_score_calibration",
}

REFERENCE_IDS = [
    "promoted_latent384_beta3p75_ch1_0_2",
    "ch1only_latent384_beta3p75",
    "latent384_beta6p5",
    "latent448_beta4p0",
    "latent512_beta3p75",
    "chmeanloss_latent384_beta3p75_stageB_oof_ecdf",
    "foldcombat_mfr_age_sex_latent384_beta3p75_stageB_oof_ecdf",
]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def maybe_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def md_table(df: pd.DataFrame, max_rows: int = 120) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
    text = view.to_markdown(index=False)
    if len(df) > max_rows:
        text += f"\n\n_Showing {max_rows} of {len(df)} rows._"
    return text + "\n"


def write_table(df: pd.DataFrame, stem: str, max_rows: int = 120) -> None:
    df.to_csv(OUT_DIR / f"{stem}.csv", index=False)
    (OUT_DIR / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def get_single(df: pd.DataFrame, **filters: Any) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for col, val in filters.items():
        mask &= df[col].eq(val)
    sub = df.loc[mask]
    if len(sub) != 1:
        raise RuntimeError(f"Expected one row for {filters}, found {len(sub)}")
    return sub.iloc[0]


def safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def first_mean_row(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=object)
    if "fold" in df.columns:
        sub = df[df["fold"].astype(str).eq("mean")]
        if not sub.empty:
            return sub.iloc[0]
    return df.iloc[0]


def scanner_value(scanner: pd.DataFrame, split: str, representation: str, col: str = "mean_balanced_accuracy") -> float:
    if scanner.empty:
        return float("nan")
    if "representation" in scanner.columns:
        sub = scanner[(scanner["split"].eq(split)) & (scanner["representation"].eq(representation))]
    else:
        sub = scanner[scanner["split"].eq(split)] if "split" in scanner.columns else scanner
    if sub.empty:
        return float("nan")
    if col in sub.columns:
        return safe_float(sub[col].iloc[0])
    if "acc_site_latent" in sub.columns:
        return safe_float(sub["acc_site_latent"].iloc[0])
    return float("nan")


def primary_oof_row(audit_dir: Path) -> pd.Series:
    pooled = read_csv(audit_dir / "stageb_oof_calibration_metrics.csv")
    return get_single(
        pooled,
        model_name=PRIMARY_MODEL,
        feature_set=PRIMARY_FEATURE,
        calib_method=PRIMARY_CALIB,
        threshold_strategy=PRIMARY_THRESHOLD,
    )


def raw_stageb_row(audit_dir: Path) -> pd.Series:
    raw = read_csv(audit_dir / "stageb_classifier_only_metrics.csv")
    return get_single(
        raw,
        model_name="logreg_l2",
        readout_feature_set=PRIMARY_FEATURE,
        threshold_strategy=PRIMARY_THRESHOLD,
    )


def logitz_row(audit_dir: Path) -> pd.Series:
    pooled = read_csv(audit_dir / "stageb_oof_calibration_metrics.csv")
    return get_single(
        pooled,
        model_name=PRIMARY_MODEL,
        feature_set=PRIMARY_FEATURE,
        calib_method="oof_logitz",
        threshold_strategy=PRIMARY_THRESHOLD,
    )


def stagea_delta_row(audit_dir: Path) -> pd.Series:
    df = maybe_csv(audit_dir / "stageA_to_stageB_delta.csv")
    return df.iloc[0] if not df.empty else pd.Series(dtype=object)


def philips_cn_fpr(audit_dir: Path) -> tuple[float, float, float]:
    if (audit_dir / "manufacturer_cn_fpr.csv").exists():
        df = read_csv(audit_dir / "manufacturer_cn_fpr.csv")
        sub = df[df["manufacturer"].astype(str).eq("Philips")]
        if not sub.empty:
            return safe_float(sub["cn_fpr"].iloc[0]), safe_float(sub["fp"].iloc[0]), safe_float(sub["cn_n"].iloc[0])
    df = read_csv(audit_dir / "philips_cn_fpr.csv")
    sub = df[
        (df["model_name"].eq(PRIMARY_MODEL))
        & (df["feature_set"].eq(PRIMARY_FEATURE))
        & (df["calib_method"].eq(PRIMARY_CALIB))
        & (df["threshold_strategy"].eq(PRIMARY_THRESHOLD))
        & (df["manufacturer"].astype(str).eq("Philips"))
    ]
    if sub.empty:
        return float("nan"), float("nan"), float("nan")
    return safe_float(sub["fpr_cn_pooled"].iloc[0]), safe_float(sub["fp_cn_pooled"].iloc[0]), safe_float(sub["n_cn_pooled"].iloc[0])


def build_method_row(
    base_columns: list[str],
    audit_dir: Path,
    model_id: str,
    display_name: str,
    decision_class: str,
    final_decision: str,
    run_name: str,
    run_dir: str,
    oasis_status: str,
) -> dict[str, Any]:
    primary = primary_oof_row(audit_dir)
    raw = raw_stageb_row(audit_dir)
    logitz = logitz_row(audit_dir)
    delta = stagea_delta_row(audit_dir)
    scanner = maybe_csv(audit_dir / "scanner_leakage_summary.csv")
    rd = first_mean_row(maybe_csv(audit_dir / "rate_distortion_summary.csv"))
    latent = maybe_csv(audit_dir / "latent_mi_signal_nuisance_summary.csv")
    if not latent.empty and {"fold", "split"}.issubset(latent.columns):
        latent_mean = latent[(latent["fold"].astype(str).eq("mean")) & (latent["split"].eq("test"))]
    else:
        latent_mean = pd.DataFrame()
    latent_row = latent_mean.iloc[0] if not latent_mean.empty else first_mean_row(latent)
    fpr, fp, cn_n = philips_cn_fpr(audit_dir)
    row = {c: np.nan for c in base_columns}
    row.update(
        {
            "model_id": model_id,
            "display_name": display_name,
            "decision_class": decision_class,
            "final_decision": final_decision,
            "run_name": run_name,
            "run_dir": run_dir,
            "channel_set_order": "[1,0,2]",
            "selected_channel_names": CHANNEL_NAMES,
            "latent_dim": 384,
            "beta_vae": 3.75,
            "adni_oof_ecdf_auc": primary["auc"],
            "adni_oof_ecdf_pr_auc": primary["pr_auc"],
            "adni_oof_ecdf_ba": primary["balanced_accuracy"],
            "adni_oof_ecdf_sens": primary["sensitivity"],
            "adni_oof_ecdf_spec": primary["specificity"],
            "adni_oof_ecdf_f1": primary["f1"],
            "adni_oof_logitz_auc": logitz["auc"],
            "adni_oof_logitz_pr_auc": logitz["pr_auc"],
            "adni_oof_logitz_ba": logitz["balanced_accuracy"],
            "adni_oof_logitz_sens": logitz["sensitivity"],
            "adni_oof_logitz_spec": logitz["specificity"],
            "adni_oof_logitz_f1": logitz["f1"],
            "adni_raw_auc": raw["auc"],
            "adni_raw_pr_auc": raw["pr_auc"],
            "adni_raw_ba": raw["balanced_accuracy"],
            "adni_raw_sens": raw["sensitivity"],
            "adni_raw_spec": raw["specificity"],
            "adni_raw_f1": raw["f1"],
            "philips_cn_fpr": fpr,
            "philips_cn_fp": fp,
            "philips_cn_n": cn_n,
            "stageA_logreg_auc": delta.get("stageA_logreg_final_auc", np.nan),
            "stageA_logreg_pr_auc": delta.get("stageA_logreg_final_pr_auc", np.nan),
            "stageB_minus_stageA_logreg_auc": delta.get("stageA_to_oof_ecdf_delta_auc", delta.get("delta_auc", np.nan)),
            "stageB_minus_stageA_logreg_pr_auc": delta.get(
                "stageA_to_oof_ecdf_delta_pr_auc", delta.get("delta_pr_auc", np.nan)
            ),
            "scanner_leakage_train_latent_acc": scanner_value(scanner, "train_dev", "latent_mu"),
            "scanner_leakage_latent_acc": scanner_value(scanner, "test", "latent_mu"),
            "D_val_best_mean": rd.get("D_val_best", np.nan),
            "R_val_bits_best_mean": rd.get("R_val_bits_best", np.nan),
            "beta_KLD_over_D_best_mean": rd.get("beta_KLD_over_D_best", np.nan),
            "bits_per_latent_dim_best_mean": rd.get("bits_per_latent_dim_best", np.nan),
            "active_units_mean": latent_row.get("active_units", np.nan),
            "total_correlation_nats_mean": latent_row.get("total_correlation_nats", np.nan),
            "MI_Z_Y_nats_mean": latent_row.get("MI_Z_Y_nats", np.nan),
            "MI_Z_Manufacturer_nats_mean": latent_row.get("MI_Z_Manufacturer_nats", np.nan),
            "MI_Manufacturer_over_MI_Y_mean": latent_row.get("MI_Manufacturer_over_MI_Y", np.nan),
            "oasis_status": oasis_status,
        }
    )
    return row


def manufacturer_error_from_predictions(model_id: str, oof_dir: Path) -> pd.DataFrame:
    pred_path = oof_dir / "calib_predictions.csv"
    if not pred_path.exists():
        return pd.DataFrame()
    df = read_csv(pred_path)
    sub = df[
        (df["model_name"].eq(PRIMARY_MODEL))
        & (df["feature_set"].eq(PRIMARY_FEATURE))
        & (df["calib_method"].eq(PRIMARY_CALIB))
        & (df["threshold_strategy"].eq(PRIMARY_THRESHOLD))
    ].copy()
    rows = []
    for mfr, g in sub.groupby("Manufacturer", dropna=False):
        cn = g[g["y_true"] == 0]
        ad = g[g["y_true"] == 1]
        fp = int((cn["y_pred"] == 1).sum())
        tn = int((cn["y_pred"] == 0).sum())
        fn = int((ad["y_pred"] == 0).sum())
        tp = int((ad["y_pred"] == 1).sum())
        rows.append(
            {
                "model_id": model_id,
                "manufacturer": mfr,
                "cn_n": len(cn),
                "cn_fp": fp,
                "cn_tn": tn,
                "cn_fpr": fp / (fp + tn) if fp + tn else np.nan,
                "ad_n": len(ad),
                "ad_fn": fn,
                "ad_tp": tp,
                "ad_fnr": fn / (fn + tp) if fn + tp else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_manufacturer_error_table() -> pd.DataFrame:
    frames = []
    for model_id, oof_dir in OOF_PRED_DIRS.items():
        df = manufacturer_error_from_predictions(model_id, oof_dir)
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compact_table(evidence: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "model_id",
        "display_name",
        "decision_class",
        "channel_set_order",
        "latent_dim",
        "beta_vae",
        "adni_oof_ecdf_auc",
        "adni_oof_ecdf_pr_auc",
        "adni_oof_ecdf_ba",
        "adni_oof_ecdf_sens",
        "adni_oof_ecdf_spec",
        "adni_oof_ecdf_f1",
        "philips_cn_fpr",
        "scanner_leakage_latent_acc",
        "beta_KLD_over_D_best_mean",
        "MI_Z_Y_nats_mean",
        "MI_Z_Manufacturer_nats_mean",
        "MI_Manufacturer_over_MI_Y_mean",
        "oasis_status",
        "final_decision",
    ]
    return evidence[[c for c in cols if c in evidence.columns]].copy()


def method_branch_comparison(evidence: pd.DataFrame) -> pd.DataFrame:
    subset = evidence[evidence["model_id"].isin(REFERENCE_IDS)].copy()
    order = {m: i for i, m in enumerate(REFERENCE_IDS)}
    subset["_order"] = subset["model_id"].map(order)
    subset = subset.sort_values("_order").drop(columns=["_order"])
    return compact_table(subset)


def write_trend_and_recommendation(evidence: pd.DataFrame, mfr_errors: pd.DataFrame) -> None:
    by_id = evidence.set_index("model_id")
    promoted = by_id.loc["promoted_latent384_beta3p75_ch1_0_2"]
    ch1 = by_id.loc["ch1only_latent384_beta3p75"]
    chmean = by_id.loc["chmeanloss_latent384_beta3p75_stageB_oof_ecdf"]
    combat = by_id.loc["foldcombat_mfr_age_sex_latent384_beta3p75_stageB_oof_ecdf"]
    beta6 = by_id.loc["latent384_beta6p5"]

    trend = f"""# Method-Branch Trend Interpretation

## A. Effective Regularization Branch

Branch A changed the reconstruction objective to channel-normalized off-diagonal loss while keeping the [1,0,2] latent384 beta3.75 setting. It increased the effective rate-distortion pressure from promoted beta*KLD/D `{promoted['beta_KLD_over_D_best_mean']:.6f}` to `{chmean['beta_KLD_over_D_best_mean']:.6f}`, but did not approach the ch1-only regime `{ch1['beta_KLD_over_D_best_mean']:.6f}`.

The downstream promoted-convention OOF-ECDF metrics were AUC `{chmean['adni_oof_ecdf_auc']:.6f}` and PR-AUC `{chmean['adni_oof_ecdf_pr_auc']:.6f}`, below the promoted [1,0,2] row `{promoted['adni_oof_ecdf_auc']:.6f}` / `{promoted['adni_oof_ecdf_pr_auc']:.6f}`. Philips CN FPR also worsened from `{promoted['philips_cn_fpr']:.6f}` to `{chmean['philips_cn_fpr']:.6f}`. This branch is best reported as an effective-regularization sensitivity, not a reopened primary candidate.

Beta6.5 also failed to reproduce the ch1-only effective regularization regime: beta*KLD/D was `{beta6['beta_KLD_over_D_best_mean']:.6f}`, with AUC `{beta6['adni_oof_ecdf_auc']:.6f}` and PR-AUC `{beta6['adni_oof_ecdf_pr_auc']:.6f}`. Together, beta scaling and channel-normalized loss do not provide a clean multichannel replacement for the promoted model.

## B. Foldwise Harmonization Branch

Branch B applied leakage-safe foldwise input ComBat by Manufacturer, preserving Age/Sex and excluding diagnosis. The harmonization branch reduced test latent scanner leakage from promoted `{promoted['scanner_leakage_latent_acc']:.6f}` to `{combat['scanner_leakage_latent_acc']:.6f}`, and the fold-level harmonization guards passed.

However, the promoted-convention Stage B row was AUC `{combat['adni_oof_ecdf_auc']:.6f}`, PR-AUC `{combat['adni_oof_ecdf_pr_auc']:.6f}`, BA `{combat['adni_oof_ecdf_ba']:.6f}`, and F1 `{combat['adni_oof_ecdf_f1']:.6f}`. Philips CN FPR worsened to `{combat['philips_cn_fpr']:.6f}`. The acquisition-nuisance reduction therefore did not translate into a promotable AD/CN operating point.

Fold-ComBat should be retained as a harmonization/deconfounding sensitivity result: it demonstrates that the nuisance axis can be reduced leakage-safely, but also that aggressive input harmonization can alter disease-ranking behavior and manufacturer-specific errors.
"""
    (OUT_DIR / "method_branch_trend_interpretation.md").write_text(trend, encoding="utf-8")

    rec = f"""# Final Recommendation

Decision: `promoted_model_remains_primary`.

Neither completed method branch reopens model selection.

- Primary remains `recover035_latent384_beta3p75_T80_h10000_p560_full5x5` [1,0,2], with OOF-ECDF AUC `{promoted['adni_oof_ecdf_auc']:.6f}` and PR-AUC `{promoted['adni_oof_ecdf_pr_auc']:.6f}`.
- ch1-only remains a parsimony sensitivity: it has higher ADNI AUC/PR-AUC (`{ch1['adni_oof_ecdf_auc']:.6f}` / `{ch1['adni_oof_ecdf_pr_auc']:.6f}`) but did not displace the promoted multichannel model in the final internal/external evidence synthesis.
- Branch A chmeanloss should be reported, if needed, as an effective-regularization sensitivity. It increased beta*KLD/D but did not improve AUC/PR-AUC and worsened Philips CN FPR.
- Branch B foldwise input ComBat should be reported as a harmonization sensitivity. It reduced scanner/manufacturer leakage substantially but failed the AUC/PR-AUC and Philips CN FPR promotion gates.

No OASIS scoring was run for these branches in this update, and no OASIS-based model selection was performed.
"""
    (OUT_DIR / "final_recommendation.md").write_text(rec, encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base = read_csv(BASE_MAP)
    base_cols = list(base.columns)

    chmean_row = build_method_row(
        base_cols,
        CHMEAN_AUDIT,
        model_id="chmeanloss_latent384_beta3p75_stageB_oof_ecdf",
        display_name="chmeanloss [1,0,2] latent384 beta3.75",
        decision_class="effective-regularization sensitivity",
        final_decision="not_promoted_effective_regularization_sensitivity",
        run_name="recover035_latent384_beta3p75_chmeanloss_T80_h10000_p560_full5x5",
        run_dir=str(ROOT / "recover035_latent384_beta3p75_chmeanloss_T80_h10000_p560_full5x5"),
        oasis_status="not_run_adni_gate_failed_effective_regularization_sensitivity",
    )
    combat_row = build_method_row(
        base_cols,
        FOLDCOMBAT_AUDIT,
        model_id="foldcombat_mfr_age_sex_latent384_beta3p75_stageB_oof_ecdf",
        display_name="foldcombat mfr+Age/Sex [1,0,2] latent384 beta3.75",
        decision_class="harmonization sensitivity",
        final_decision="not_promoted_harmonization_sensitivity",
        run_name="recover035_latent384_beta3p75_foldcombat_mfr_age_sex_T80_h10000_p560_full5x5",
        run_dir=str(ROOT / "recover035_latent384_beta3p75_foldcombat_mfr_age_sex_T80_h10000_p560_full5x5"),
        oasis_status="not_run_adni_gate_failed_harmonization_sensitivity_only",
    )

    drop_ids = {chmean_row["model_id"], combat_row["model_id"]}
    updated = base[~base["model_id"].isin(drop_ids)].copy()
    updated = pd.concat([updated, pd.DataFrame([chmean_row, combat_row])], ignore_index=True, sort=False)

    # Keep original columns first, then any additions.
    updated = updated[base_cols + [c for c in updated.columns if c not in base_cols]]
    write_table(updated, "full_model_evidence_map", max_rows=220)
    write_table(compact_table(updated), "model_decision_compact_table", max_rows=220)
    method_comp = method_branch_comparison(updated)
    write_table(method_comp, "method_branch_comparison", max_rows=80)

    mfr_errors = build_manufacturer_error_table()
    write_table(mfr_errors, "manufacturer_error_by_model", max_rows=160)

    write_trend_and_recommendation(updated, mfr_errors)

    (OUT_DIR / "README.md").write_text(
        "\n".join(
            [
                "# Final FULL-Model Evidence Map With Method Branches",
                "",
                "This read-only package updates the completed FULL-model evidence map with Branch A",
                "channel-normalized reconstruction loss and Branch B foldwise input ComBat harmonization.",
                "",
                "No training, OASIS scoring, threshold fitting, calibration fitting, tensor edits, metadata",
                "edits, or model artifact edits were performed.",
                "",
                "Final recommendation: `promoted_model_remains_primary`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (OUT_DIR / "command_log.json").write_text(
        json.dumps(
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "script": str(Path(__file__)),
                "base_map": str(BASE_MAP),
                "branch_a_audit": str(CHMEAN_AUDIT),
                "branch_b_audit": str(FOLDCOMBAT_AUDIT),
                "output_dir": str(OUT_DIR),
                "guardrails": [
                    "no_training",
                    "no_oasis_scoring",
                    "no_threshold_fitting",
                    "no_calibration_fitting",
                    "no_tensor_modification",
                    "no_metadata_modification",
                    "no_model_artifact_modification",
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
