#!/usr/bin/env python3
"""Update final FULL-model evidence map with ch12 beta2.75 current-loss result.

Read-only with respect to model artifacts. Writes a new evidence-map package.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "results/revision_bspc_2026/final_full_model_evidence_map_with_method_branches_20260607"
AUDIT = ROOT / "results/revision_bspc_2026/ch12_beta2p75_currentloss_completion_promotion_gate_audit_20260607"
OUT = ROOT / "results/revision_bspc_2026/final_full_model_evidence_map_with_ch12_beta2p75_20260607"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def to_md(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False) + "\n"
    except Exception:
        return df.to_string(index=False) + "\n"


def write_table(df: pd.DataFrame, stem: str) -> None:
    df.to_csv(OUT / f"{stem}.csv", index=False)
    (OUT / f"{stem}.md").write_text(to_md(df), encoding="utf-8")


def first_row(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if df.empty:
        raise RuntimeError(f"No rows in {path}")
    return df.iloc[0]


def load_primary_row() -> pd.Series:
    df = pd.read_csv(AUDIT / "primary_oof_rows.csv")
    row = df[df["primary_row"] == "oof_ecdf"]
    if row.empty:
        raise RuntimeError("Missing oof_ecdf primary row")
    return row.iloc[0]


def load_logitz_row() -> pd.Series:
    df = pd.read_csv(AUDIT / "primary_oof_rows.csv")
    row = df[df["primary_row"] == "oof_logitz"]
    if row.empty:
        raise RuntimeError("Missing oof_logitz primary row")
    return row.iloc[0]


def stagea_row(classifier_type: str) -> pd.Series:
    df = pd.read_csv(AUDIT / "stagea_pooled_metrics.csv")
    row = df[df["classifier_type"] == classifier_type]
    if row.empty:
        return pd.Series(dtype=object)
    return row.iloc[0]


def manufacturer_row(calib_method: str, manufacturer: str) -> pd.Series:
    df = pd.read_csv(AUDIT / "manufacturer_cn_fpr_ad_fnr_primary.csv")
    row = df[(df["calib_method"] == calib_method) & (df["manufacturer"] == manufacturer)]
    if row.empty:
        raise RuntimeError(f"Missing manufacturer row {calib_method}/{manufacturer}")
    return row.iloc[0]


def scanner_leakage(split: str) -> float:
    df = pd.read_csv(AUDIT / "scanner_leakage_summary.csv")
    row = df[df["split"] == split]
    if row.empty:
        return float("nan")
    return float(row.iloc[0]["latent_scanner_ba_mean"])


def build_new_row(columns: list[str]) -> Dict[str, Any]:
    ecdf = load_primary_row()
    logitz = load_logitz_row()
    rd = first_row(AUDIT / "rate_distortion_summary.csv")
    mi = first_row(AUDIT / "latent_mi_signal_nuisance_summary.csv")
    stg_logreg = stagea_row("logreg")
    stg_svm = stagea_row("svm")
    philips = manufacturer_row("oof_ecdf", "Philips")
    row: Dict[str, Any] = {col: pd.NA for col in columns}
    row.update(
        {
            "model_id": "ch12_latent384_beta2p75_currentloss_stageB_oof_ecdf",
            "display_name": "ch12 latent384 beta2.75 currentloss",
            "decision_class": "channel-pair beta sensitivity",
            "final_decision": "reject",
            "run_name": "recover035_ch12_latent384_beta2p75_currentloss_T80_h10000_p560_full5x5",
            "run_dir": "results/revision_bspc_2026/recover035_ch12_latent384_beta2p75_currentloss_T80_h10000_p560_full5x5",
            "channel_set_order": "[1, 2]",
            "selected_channel_names": '["Pearson_Full_FisherZ_Signed", "MI_KNN_Symmetric"]',
            "latent_dim": 384,
            "beta_vae": 2.75,
            "adni_oof_ecdf_auc": float(ecdf["auc"]),
            "adni_oof_ecdf_pr_auc": float(ecdf["pr_auc"]),
            "adni_oof_logitz_auc": float(logitz["auc"]),
            "adni_oof_logitz_pr_auc": float(logitz["pr_auc"]),
            "adni_raw_auc": 0.743952,
            "adni_raw_pr_auc": 0.507896,
            "adni_oof_ecdf_ba": float(ecdf["balanced_accuracy"]),
            "adni_oof_ecdf_sens": float(ecdf["sensitivity"]),
            "adni_oof_ecdf_spec": float(ecdf["specificity"]),
            "adni_oof_ecdf_f1": float(ecdf["f1"]),
            "adni_oof_logitz_ba": float(logitz["balanced_accuracy"]),
            "adni_oof_logitz_sens": float(logitz["sensitivity"]),
            "adni_oof_logitz_spec": float(logitz["specificity"]),
            "adni_oof_logitz_f1": float(logitz["f1"]),
            "adni_raw_ba": 0.692646,
            "adni_raw_sens": 0.731959,
            "adni_raw_spec": 0.653333,
            "adni_raw_f1": 0.522059,
            "philips_cn_fpr": float(philips["fpr_cn_pooled"]),
            "philips_cn_fp": int(philips["fp_cn_pooled"]),
            "philips_cn_n": int(philips["n_cn_pooled"]),
            "stageA_logreg_auc": float(stg_logreg.get("auc", pd.NA)),
            "stageA_logreg_pr_auc": float(stg_logreg.get("pr_auc", pd.NA)),
            "stageA_svm_auc": float(stg_svm.get("auc", pd.NA)),
            "stageA_svm_pr_auc": float(stg_svm.get("pr_auc", pd.NA)),
            "stageB_minus_stageA_logreg_auc": float(ecdf["auc"]) - float(stg_logreg.get("auc", pd.NA)),
            "stageB_minus_stageA_logreg_pr_auc": float(ecdf["pr_auc"]) - float(stg_logreg.get("pr_auc", pd.NA)),
            "scanner_leakage_train_latent_acc": scanner_leakage("trainDev"),
            "scanner_leakage_latent_acc": scanner_leakage("test"),
            "D_val_best_mean": float(rd["D_val_best"]),
            "R_val_bits_best_mean": float(rd["R_val_bits_best"]),
            "beta_KLD_over_D_best_mean": float(rd["beta_KLD_over_D_best"]),
            "bits_per_latent_dim_best_mean": float(rd["R_bits_per_latent_dim_best"]),
            "active_units_mean": float(mi["active_units"]),
            "total_correlation_nats_mean": float(mi["total_correlation_nats"]),
            "MI_Z_Y_nats_mean": float(mi["MI_Z_Y_nats"]),
            "MI_Z_Manufacturer_nats_mean": float(mi["MI_Z_Manufacturer_nats"]),
            "MI_Manufacturer_over_MI_Y_mean": float(mi["MI_Manufacturer_over_MI_Y"]),
            "oasis_status": "not_scored_ADNI_gate_failed",
        }
    )
    return row


def append_or_replace(df: pd.DataFrame, row: Dict[str, Any]) -> pd.DataFrame:
    model_id = row["model_id"]
    df = df[df["model_id"] != model_id].copy()
    return pd.concat([df, pd.DataFrame([row], columns=df.columns)], ignore_index=True)


def write_texts(full: pd.DataFrame) -> None:
    prior_trend = (SRC / "method_branch_trend_interpretation.md").read_text(encoding="utf-8")
    extra = """

## C. [1,2] Channel-Pair Beta2.75 Current-Loss Branch

The reviewer-driven [1,2] Pearson Full + MI-KNN branch lowered beta from `3.75`
to `2.75` under the current reconstruction loss to test whether weaker
regularization preserved more diagnostic signal. The promoted-convention
OOF-ECDF row was AUC `0.776529`, PR-AUC `0.556809`, BA `0.689158`, Sens
`0.721649`, Spec `0.656667`, and F1 `0.518519`.

This is below the promoted [1,0,2] reference and below the ch1-only reference.
Philips CN FPR worsened to `51/99 = 0.515152`; test latent scanner leakage was
`0.740222`, also worse than promoted. The observed beta*KLD/D was `0.033557`,
above promoted beta3.75 (`0.025711`) but still far below the ch1-only regime
(`0.065268`). OASIS was not scored because the ADNI promotion gate failed.

Conclusion: beta2.75 does not rescue the [1,2] channel pair as a primary or
sensitivity model; it remains a rejected channel-pair/beta-axis sensitivity.
"""
    (OUT / "method_branch_trend_interpretation.md").write_text(prior_trend.rstrip() + extra + "\n", encoding="utf-8")

    final = """# Final Recommendation

Decision: `promoted_model_remains_primary`.

Adding `ch12_latent384_beta2p75_currentloss_stageB_oof_ecdf` does not reopen model
selection. The branch failed the ADNI gate:

- AUC `0.776529` < promoted `0.795155`.
- PR-AUC `0.556809` < promoted `0.573934`.
- BA/F1/Sensitivity were worse than promoted.
- Philips CN FPR worsened to `51/99 = 0.515152`.
- Test latent scanner leakage worsened to `0.740222`.

The branch was not scored on OASIS because the ADNI promotion gate failed. The
primary model remains the promoted `[1,0,2]` latent384 beta3.75 model. ch1-only
remains a parsimony sensitivity; chmeanloss and foldcombat remain method
sensitivities; ch12 beta2.75 is rejected.

Guardrails: no training, no OASIS scoring, no threshold/calibration fitting, and
no tensor/metadata/model-artifact modification were performed for this update.
"""
    (OUT / "final_recommendation.md").write_text(final, encoding="utf-8")
    readme = f"""# Final FULL-Model Evidence Map With ch12 Beta2.75

Source evidence map: `{rel(SRC)}`.

Added row: `ch12_latent384_beta2p75_currentloss_stageB_oof_ecdf`, from
`{rel(AUDIT)}`.

This is a read-only evidence-map update. No model artifacts were modified.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    full = pd.read_csv(SRC / "full_model_evidence_map.csv")
    row = build_new_row(list(full.columns))
    full = append_or_replace(full, row)
    write_table(full, "full_model_evidence_map")

    compact_cols = list(pd.read_csv(SRC / "model_decision_compact_table.csv").columns)
    compact = full[compact_cols].copy()
    write_table(compact, "model_decision_compact_table")

    for name in ["manufacturer_error_by_model.csv", "method_branch_comparison.csv"]:
        src = SRC / name
        if src.exists():
            shutil.copy2(src, OUT / name)
            md = src.with_suffix(".md")
            if md.exists():
                shutil.copy2(md, OUT / md.name)
    write_texts(full)
    with (OUT / "command_log.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "created_utc": now_utc(),
                "source_evidence_map": rel(SRC),
                "source_audit": rel(AUDIT),
                "output_dir": rel(OUT),
                "added_model_id": row["model_id"],
                "guardrails": [
                    "no training",
                    "no OASIS scoring",
                    "no threshold/calibration fitting",
                    "no tensor modification",
                    "no metadata modification",
                    "no model artifact modification",
                ],
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")


if __name__ == "__main__":
    main()
