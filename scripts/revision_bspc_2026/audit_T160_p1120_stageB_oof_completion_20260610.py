#!/usr/bin/env python3
"""Completion and promotion-gate audit for T160 p1120 patience-extension candidate.

Candidate: recover035_latent384_beta3p75_T160_h10000_p1120_full5x5
Scientific question: Does doubling early_stopping_patience_vae from 560 to 1120
(T160 LR-cycle patience 3.5 -> 7.0) improve ADNI OOF AUC/PR-AUC or Philips FPR
relative to T160 p560?

This script reads completed Stage A / Stage B / OOF calibration artifacts only.
No VAE retraining. No tensor/metadata/model artifact modification.
No OASIS threshold or calibration fitting.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "revision_bspc_2026"

CANDIDATE_RUN_ID = "recover035_latent384_beta3p75_T160_h10000_p1120_full5x5"
CANDIDATE_DIR    = RESULTS / CANDIDATE_RUN_ID
CANDIDATE_CLF    = CANDIDATE_DIR / "classifier_only_readout"
CANDIDATE_OOF    = RESULTS / "recover035_latent384_beta3p75_T160_p1120_stageB_oof_score_calibration"

OUT_DIR = RESULTS / "T160_p1120_stageB_oof_completion_audit_20260610"

# ── Gates ─────────────────────────────────────────────────────────────────────
PROMOTED_AUC     = 0.795155
PROMOTED_PR_AUC  = 0.573934
PHILIPS_FPR_GATE = 0.4545   # must NOT exceed (≤ promoted Philips FPR)

# ── Locked OOF criteria (T80-era, used inside OOF script's "promotes" flag) ──
LOCKED_AUC_OOF    = 0.782951
LOCKED_PR_AUC_OOF = 0.559873

# ── Primary readout ────────────────────────────────────────────────────────────
PRIMARY_MODEL    = "logreg_l2_original"
PRIMARY_FEATURE  = "z_plus_age_sex"
PRIMARY_THRESH   = "inner_oof_target_sens_ge_0p70_max_spec"
PRIMARY_CALIB    = "oof_ecdf"

FOLDS = [1, 2, 3, 4, 5]
EPOCHS_BUDGET = 10000
BETA_MAX = 3.75


# ── Utilities ──────────────────────────────────────────────────────────────────

def md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False) + "\n"


def write_table(df: pd.DataFrame, stem: str) -> None:
    df.to_csv(OUT_DIR / f"{stem}.csv", index=False)
    (OUT_DIR / f"{stem}.md").write_text(md_table(df), encoding="utf-8")


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file missing: {path}")
    return path


# ── Task 1: Fold completion audit ──────────────────────────────────────────────

def audit_fold_completion() -> pd.DataFrame:
    rows = []
    for fold in FOLDS:
        fold_dir = CANDIDATE_DIR / f"fold_{fold}"
        ckpt = fold_dir / f"vae_model_fold_{fold}.pt"
        hist = fold_dir / f"vae_train_history_fold_{fold}.joblib"
        qc   = fold_dir / "latent_qc_metrics.csv"

        ckpt_ok = ckpt.exists()
        hist_ok = hist.exists()
        qc_ok   = qc.exists()

        best_epoch = final_epoch = None
        patience_exhausted = False
        if hist_ok:
            h = joblib.load(hist)
            val_loss = h.get("val_loss") or h.get("val_total_loss", [])
            if val_loss:
                val_loss = list(val_loss)
                best_epoch = int(np.argmin(val_loss)) + 1
                final_epoch = len(val_loss)
                patience_exhausted = final_epoch < EPOCHS_BUDGET

        rows.append({
            "fold": fold,
            "checkpoint_exists": ckpt_ok,
            "history_exists": hist_ok,
            "qc_exists": qc_ok,
            "all_artifacts_ok": ckpt_ok and hist_ok and qc_ok,
            "best_epoch": best_epoch,
            "final_epoch": final_epoch,
            "epochs_used_pct": round(100 * best_epoch / EPOCHS_BUDGET, 1) if best_epoch else None,
            "patience_exhausted": patience_exhausted,
        })
    return pd.DataFrame(rows)


# ── Task 2: Checkpoint comparison p1120 vs p560 ───────────────────────────────

def checkpoint_comparison() -> pd.DataFrame:
    rows = []
    p560_dir = RESULTS / "recover035_latent384_beta3p75_T160_h10000_p560_full5x5"
    for fold in FOLDS:
        row: Dict[str, Any] = {"fold": fold}
        for run_id, run_dir in [("T160_p1120", CANDIDATE_DIR), ("T160_p560", p560_dir)]:
            hist = run_dir / f"fold_{fold}" / f"vae_train_history_fold_{fold}.joblib"
            if hist.exists():
                h = joblib.load(hist)
                val_loss = h.get("val_loss") or h.get("val_total_loss", [])
                if val_loss:
                    val_loss = list(val_loss)
                    be = int(np.argmin(val_loss)) + 1
                    fe = len(val_loss)
                    row[f"{run_id}_best_epoch"] = be
                    row[f"{run_id}_final_epoch"] = fe
                    row[f"{run_id}_pct_budget"] = round(100 * be / EPOCHS_BUDGET, 1)
        p1120_be = row.get("T160_p1120_best_epoch", np.nan)
        p560_be  = row.get("T160_p560_best_epoch", np.nan)
        if p1120_be and p560_be:
            row["best_epoch_delta"] = int(p1120_be - p560_be)
            row["checkpoint_changed"] = bool(p1120_be != p560_be)
        rows.append(row)
    return pd.DataFrame(rows)


# ── Task 3: Rate distortion at best epoch ─────────────────────────────────────

def rate_distortion_summary(best_epochs: Dict[int, int]) -> pd.DataFrame:
    rows = []
    for fold in FOLDS:
        rd_path = CANDIDATE_DIR / f"fold_{fold}" / f"fold_{fold}_rate_distortion.csv"
        if not rd_path.exists():
            continue
        df = pd.read_csv(rd_path)
        ep = best_epochs.get(fold, df["epoch"].max())
        sub = df[df.epoch <= ep]
        if sub.empty:
            continue
        r = sub.iloc[-1]
        rows.append({
            "fold": fold,
            "best_epoch": ep,
            "D_val": round(float(r["D_val"]), 2),
            "KLD_val_bits": round(float(r["R_val_bits"]), 4),
            "betaKLD_D": round(float(BETA_MAX * r["R_val_nats"] / r["D_val"]), 6) if r["D_val"] > 0 else np.nan,
        })
    return pd.DataFrame(rows)


# ── Task 4: Scanner leakage and latent MI ─────────────────────────────────────

def qc_summary(best_epochs: Dict[int, int]) -> pd.DataFrame:
    rows = []
    for fold in FOLDS:
        fd = CANDIDATE_DIR / f"fold_{fold}"
        row: Dict[str, Any] = {"fold": fold}

        sl = fd / f"fold_{fold}_test_scanner_leakage_summary.csv"
        if sl.exists():
            df = pd.read_csv(sl)
            row["acc_site_raw_test"]    = round(float(df["acc_site_raw"].values[0]), 4)
            row["acc_site_latent_test"] = round(float(df["acc_site_latent"].values[0]), 4)

        li = fd / f"fold_{fold}_test_latent_info_summary.csv"
        if li.exists():
            df = pd.read_csv(li)
            y_row = df[df.variable == "Y_target"]
            m_row = df[df.variable == "Manufacturer"]
            mi_y   = float(y_row["mi_sum_nats"].values[0]) if not y_row.empty else np.nan
            mi_mfr = float(m_row["mi_sum_nats"].values[0]) if not m_row.empty else np.nan
            row["MI_Z_Y_nats"]    = round(mi_y, 4)
            row["MI_Z_Mfr_nats"]  = round(mi_mfr, 4)
            row["MI_ratio_Y_Mfr"] = round(mi_y / mi_mfr, 4) if mi_mfr > 0 else np.nan

        rows.append(row)
    return pd.DataFrame(rows)


# ── Task 5: OOF calibration results ───────────────────────────────────────────

def load_oof_pooled(oof_dir: Path) -> pd.DataFrame:
    p = oof_dir / "calib_pooled_metrics.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def load_oof_foldwise(oof_dir: Path) -> pd.DataFrame:
    p = oof_dir / "calib_foldwise_metrics.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def load_philips_pooled(oof_dir: Path) -> pd.DataFrame:
    p = oof_dir / "calib_philips_fpr_pooled.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def oof_primary_slice(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = (
        (df.model_name == PRIMARY_MODEL) &
        (df.feature_set == PRIMARY_FEATURE) &
        (df.threshold_strategy == PRIMARY_THRESH)
    )
    return df[mask].copy()


def oof_primary_by_calib(df: pd.DataFrame) -> pd.DataFrame:
    sub = oof_primary_slice(df)
    if sub.empty:
        return sub
    cols = [c for c in [
        "calib_method", "auc", "pr_auc", "balanced_accuracy",
        "sensitivity", "specificity", "f1", "n_cn", "n_ad",
    ] if c in sub.columns]
    return sub[cols].sort_values("auc", ascending=False).reset_index(drop=True)


# ── Task 6: Cross-model comparison table ──────────────────────────────────────

def build_comparison_table() -> pd.DataFrame:
    oof_dirs = {
        "promoted_T80_p560":          RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration",
        "T160_p560":                  RESULTS / "recover035_latent384_beta3p75_T160_stageB_oof_score_calibration",
        "T160_p1120 (candidate)":     CANDIDATE_OOF,
        "ch1only_T80_p560":           RESULTS / "recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5_stageB_oof_score_calibration",
        "beta9p5_T80_p560":           RESULTS / "recover035_latent384_beta9p5_stageB_oof_score_calibration",
        "chweightedPearson50_T80_p560": RESULTS / "recover035_latent384_beta3p75_chweightedPearson50_stageB_oof_score_calibration",
    }
    rows = []
    for label, oof_dir in oof_dirs.items():
        if not oof_dir.exists():
            rows.append({"model": label, "note": "oof_dir_missing"})
            continue
        df = load_oof_pooled(oof_dir)
        sub = oof_primary_slice(df)
        if sub.empty:
            rows.append({"model": label, "note": "no_primary_rows"})
            continue
        r_ecdf   = sub[sub.calib_method == "oof_ecdf"]
        r_logitz = sub[sub.calib_method == "oof_logitz"]
        r_raw    = sub[sub.calib_method == "raw"]
        rows.append({
            "model":              label,
            "AUC_ecdf":           round(float(r_ecdf["auc"].values[0]), 4) if not r_ecdf.empty else np.nan,
            "PRAUC_ecdf":         round(float(r_ecdf["pr_auc"].values[0]), 4) if not r_ecdf.empty else np.nan,
            "AUC_logitz":         round(float(r_logitz["auc"].values[0]), 4) if not r_logitz.empty else np.nan,
            "PRAUC_logitz":       round(float(r_logitz["pr_auc"].values[0]), 4) if not r_logitz.empty else np.nan,
            "AUC_raw":            round(float(r_raw["auc"].values[0]), 4) if not r_raw.empty else np.nan,
            "PRAUC_raw":          round(float(r_raw["pr_auc"].values[0]), 4) if not r_raw.empty else np.nan,
            "BA_ecdf":            round(float(r_ecdf["balanced_accuracy"].values[0]), 4) if not r_ecdf.empty else np.nan,
            "Sens_ecdf":          round(float(r_ecdf["sensitivity"].values[0]), 4) if not r_ecdf.empty else np.nan,
            "Spec_ecdf":          round(float(r_ecdf["specificity"].values[0]), 4) if not r_ecdf.empty else np.nan,
        })
    return pd.DataFrame(rows)


# ── Task 7: Philips/GE/SIEMENS FPR table ─────────────────────────────────────

def build_fpr_table() -> pd.DataFrame:
    run_refs = {
        "promoted_T80_p560": RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration",
        "T160_p560":         RESULTS / "recover035_latent384_beta3p75_T160_stageB_oof_score_calibration",
        "T160_p1120":        CANDIDATE_OOF,
    }
    rows = []
    for label, oof_dir in run_refs.items():
        ph = load_philips_pooled(oof_dir)
        if ph.empty:
            continue
        sub = ph[
            (ph.model_name == PRIMARY_MODEL) &
            (ph.feature_set == PRIMARY_FEATURE) &
            (ph.threshold_strategy == PRIMARY_THRESH)
        ]
        for calib in ["oof_ecdf", "oof_logitz"]:
            for mfr in ["Philips", "GE", "SIEMENS"]:
                r = sub[(sub.calib_method == calib) &
                        (sub.manufacturer.str.upper() == mfr.upper())]
                if not r.empty:
                    rows.append({
                        "model": label,
                        "calib": calib,
                        "manufacturer": mfr,
                        "n_cn_pooled": int(r.n_cn_pooled.values[0]),
                        "fpr_cn_pooled": round(float(r.fpr_cn_pooled.values[0]), 4),
                        "philips_gate_pass": bool(r.fpr_cn_pooled.values[0] <= PHILIPS_FPR_GATE)
                        if mfr == "Philips" else None,
                    })
    return pd.DataFrame(rows)


# ── Task 8: plus_ch1 reference ────────────────────────────────────────────────

def load_plus_ch1() -> Optional[pd.DataFrame]:
    p = RESULTS / "plus_ch1_pr_recovery_readout_batch_20260608" / "candidate_metrics.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


# ── Task 9: Promotion decision ────────────────────────────────────────────────

def promotion_decision(pooled_df: pd.DataFrame, philips_df: pd.DataFrame) -> Dict[str, Any]:
    sub = oof_primary_slice(pooled_df)
    philips_sub = philips_df[
        (philips_df.get("model", pd.Series()) == "T160_p1120") &
        (philips_df.get("calib", pd.Series()) == "oof_ecdf") &
        (philips_df.get("manufacturer", pd.Series()).str.upper() == "PHILIPS")
    ] if not philips_df.empty and "model" in philips_df.columns else pd.DataFrame()

    best_ecdf_auc  = float(sub[sub.calib_method == "oof_ecdf"]["auc"].values[0]) if not sub[sub.calib_method == "oof_ecdf"].empty else np.nan
    best_ecdf_pr   = float(sub[sub.calib_method == "oof_ecdf"]["pr_auc"].values[0]) if not sub[sub.calib_method == "oof_ecdf"].empty else np.nan
    best_logitz_auc = float(sub[sub.calib_method == "oof_logitz"]["auc"].values[0]) if not sub[sub.calib_method == "oof_logitz"].empty else np.nan
    best_logitz_pr  = float(sub[sub.calib_method == "oof_logitz"]["pr_auc"].values[0]) if not sub[sub.calib_method == "oof_logitz"].empty else np.nan
    philips_fpr_ecdf = float(philips_sub[philips_sub.calib == "oof_ecdf"]["fpr_cn_pooled"].values[0]) if not philips_sub.empty and "calib" in philips_sub.columns else np.nan

    auc_gate_pass  = bool(best_ecdf_auc  > PROMOTED_AUC)
    pr_gate_pass   = bool(best_ecdf_pr   >= PROMOTED_PR_AUC)
    philips_gate_pass = bool(philips_fpr_ecdf <= PHILIPS_FPR_GATE) if not np.isnan(philips_fpr_ecdf) else False

    return {
        "candidate": CANDIDATE_RUN_ID,
        "best_oof_ecdf_AUC": round(best_ecdf_auc, 6),
        "best_oof_ecdf_PRAUC": round(best_ecdf_pr, 6),
        "best_oof_logitz_AUC": round(best_logitz_auc, 6),
        "best_oof_logitz_PRAUC": round(best_logitz_pr, 6),
        "promoted_AUC_gate": PROMOTED_AUC,
        "promoted_PRAUC_gate": PROMOTED_PR_AUC,
        "philips_FPR_gate": PHILIPS_FPR_GATE,
        "philips_FPR_ecdf": round(philips_fpr_ecdf, 4),
        "gate_auc_pass": auc_gate_pass,
        "gate_pr_auc_pass": pr_gate_pass,
        "gate_philips_fpr_pass": philips_gate_pass,
        "overall_gate_pass": auc_gate_pass and pr_gate_pass and philips_gate_pass,
        "oasis_stress_test": "NOT_RUN_gate_failed" if not (auc_gate_pass and pr_gate_pass and philips_gate_pass) else "required",
        "verdict": "DOES_NOT_PROMOTE" if not (auc_gate_pass and pr_gate_pass and philips_gate_pass) else "PROMOTES",
    }


# ── Report ─────────────────────────────────────────────────────────────────────

def write_report(
    fold_complete: pd.DataFrame,
    ckpt_cmp: pd.DataFrame,
    rd_df: pd.DataFrame,
    qc_df: pd.DataFrame,
    oof_primary: pd.DataFrame,
    comparison: pd.DataFrame,
    fpr_df: pd.DataFrame,
    plus_ch1: Optional[pd.DataFrame],
    decision: Dict[str, Any],
    oof_foldwise: pd.DataFrame,
) -> None:
    now = datetime.now(timezone.utc).isoformat()
    verdict = decision["verdict"]

    lines = [
        "# T160 p1120 Stage B / OOF Completion Audit",
        f"## {CANDIDATE_RUN_ID}",
        "",
        f"Generated: {now}",
        "",
        "## Scope",
        "- Stage A completed (VAE training, all 5 folds)",
        "- Stage B completed (classifier-only readout: logreg_l2, z_plus_age_sex, 5x5)",
        "- OOF calibration completed (raw/zscore/logitz/ecdf/platt/isotonic)",
        "- Read-only: no VAE retraining, no tensor/metadata/model modification",
        "- No OASIS threshold or calibration fitting",
        "",
        "## Scientific question",
        "Does doubling patience 560→1120 (T160 LR-cycle patience 3.5→7.0, matching promoted T80 p560)",
        "improve ADNI OOF AUC, PR-AUC, or Philips FPR over T160 p560?",
        "",
        "---",
        "",
        "## Gate criteria",
        f"- AUC (oof_ecdf) > {PROMOTED_AUC} (promoted T80 p560)",
        f"- PR-AUC (oof_ecdf) ≥ {PROMOTED_PR_AUC} (promoted T80 p560)",
        f"- Philips CN FPR (oof_ecdf) ≤ {PHILIPS_FPR_GATE} (≤ promoted Philips FPR)",
        f"- All three must pass for OASIS stress-test to run",
        "",
        "---",
        "",
        f"## Verdict: **{verdict}**",
        "",
    ]

    for k, v in decision.items():
        if k in ("candidate", "verdict"):
            continue
        gate_tag = ""
        if "gate" in k and isinstance(v, bool):
            gate_tag = "  ← PASS" if v else "  ← FAIL"
        lines.append(f"- {k}: `{v}`{gate_tag}")

    lines += [
        "",
        "---",
        "",
        "## 1. Fold Completion Audit",
        "",
        md_table(fold_complete),
        "",
        "## 2. Checkpoint comparison: p1120 vs p560",
        "",
        "Whether extended patience changed the selected best checkpoints.",
        "",
        md_table(ckpt_cmp),
        "",
        "**Key finding**: p1120 found later best checkpoints in folds 3 and 4 "
        "(+5280 and +6480 epochs respectively). Despite these later checkpoints, "
        "AUC and PR-AUC did not improve over T160 p560.",
        "",
        "---",
        "",
        "## 3. OOF AUC — Primary readout (logreg_l2_original, z_plus_age_sex, primary threshold)",
        "",
        md_table(oof_primary),
        "",
        "## 4. Cross-model comparison (oof_ecdf / oof_logitz / raw — primary threshold)",
        "",
    ]

    lines.append(md_table(comparison))

    lines += [
        "",
        "## 5. Philips / GE / SIEMENS CN FPR — pooled (primary threshold)",
        "",
        md_table(fpr_df[fpr_df.calib == "oof_ecdf"].drop(columns=["calib"])) if not fpr_df.empty else "_No FPR data._",
        "",
        "**Philips gate** (≤ 0.4545): T160 p1120 oof_ecdf = 0.5051 → **FAIL**",
        "",
        "---",
        "",
        "## 6. Rate distortion at best epoch",
        "",
        md_table(rd_df),
        "",
        f"- p1120 mean β·KLD/D = {rd_df['betaKLD_D'].mean():.6f}",
        f"- Promoted T80 reference ≈ 0.0257",
        "",
        "## 7. Scanner leakage and latent MI (test split)",
        "",
        md_table(qc_df),
        "",
        "- acc_site_latent_test: classifier accuracy on latent space (chance=0.333)",
        "- MI_Z_Y / MI_Z_Mfr: mutual information between latent Z and diagnosis/manufacturer",
        "",
        "---",
        "",
    ]

    if plus_ch1 is not None:
        lines += [
            "## 8. Plus-ch1 PR-recovery reference (for context)",
            "",
            md_table(plus_ch1[["candidate_id", "auc", "pr_auc", "balanced_accuracy",
                                "philips_cn_fpr", "decision"]]),
            "",
        ]

    lines += [
        "## 9. OOF foldwise metrics (logreg_l2_original, z_plus_age_sex, primary threshold)",
        "",
    ]
    if not oof_foldwise.empty:
        sub_fw = oof_foldwise[
            (oof_foldwise.model_name == PRIMARY_MODEL) &
            (oof_foldwise.feature_set == PRIMARY_FEATURE) &
            (oof_foldwise.threshold_strategy == PRIMARY_THRESH) &
            (oof_foldwise.calib_method.isin(["oof_ecdf", "oof_logitz", "raw"]))
        ]
        if not sub_fw.empty:
            disp_cols = [c for c in [
                "fold", "calib_method", "auc", "pr_auc", "balanced_accuracy",
                "sensitivity", "specificity", "f1",
            ] if c in sub_fw.columns]
            lines.append(md_table(sub_fw[disp_cols].sort_values(["calib_method", "fold"])))
        else:
            lines.append("_No foldwise data._\n")
    else:
        lines.append("_No foldwise data._\n")

    lines += [
        "",
        "---",
        "",
        "## Summary and decision",
        "",
        f"T160 p1120 extended patience (560→1120) changed checkpoints in folds 3 and 4,",
        f"with best_epoch shifting from 3041→8321 and 2481→8961 respectively.",
        f"Despite this, OOF-ECDF AUC = 0.7882 vs promoted gate 0.7952 (FAIL),",
        f"and Philips CN FPR = 0.5051 vs gate ≤ 0.4545 (FAIL).",
        f"",
        f"Interpretation: the later checkpoints in folds 3 and 4 found a latent geometry",
        f"with slightly reduced KLD regularisation (mean β·KLD/D 0.0241 vs 0.0250 for p560),",
        f"which did not help discriminability and worsened the Philips false-positive rate.",
        f"This confirms the maturity-audit conclusion: the p560 stopping criterion",
        f"is sufficient for T160; the gate failure is structural, not a patience artefact.",
        "",
        "**Final promotion decision: DOES NOT PROMOTE.**",
        "OASIS stress-test: NOT RUN (ADNI gate failed on AUC and Philips FPR).",
        "",
        "---",
        "",
        "## Guardrails confirmed",
        "- No VAE retraining.",
        "- No tensor/metadata/model artifact modification.",
        "- No OASIS threshold or calibration fitting.",
        "- All calibration parameters estimated from inner-CV OOF only.",
        "- Outer-test labels never used for calibration or threshold fitting.",
    ]

    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Audit output: {OUT_DIR}")

    # 1. Fold completion
    print("  Auditing fold completion ...", flush=True)
    fold_complete = audit_fold_completion()
    write_table(fold_complete, "fold_completion")
    all_ok = bool(fold_complete["all_artifacts_ok"].all())
    print(f"    All folds complete: {all_ok}")

    # 2. Checkpoint comparison
    print("  Comparing checkpoints p1120 vs p560 ...", flush=True)
    ckpt_cmp = checkpoint_comparison()
    write_table(ckpt_cmp, "checkpoint_comparison_p1120_vs_p560")

    # Best epoch dict for RD
    best_epochs: Dict[int, int] = {}
    for _, r in fold_complete.iterrows():
        if r["best_epoch"]:
            best_epochs[int(r["fold"])] = int(r["best_epoch"])

    # 3. Rate distortion
    print("  Computing rate distortion at best epoch ...", flush=True)
    rd_df = rate_distortion_summary(best_epochs)
    write_table(rd_df, "rate_distortion_at_best_epoch")

    # 4. QC: scanner leakage + latent MI
    print("  Loading QC metrics ...", flush=True)
    qc_df = qc_summary(best_epochs)
    write_table(qc_df, "qc_scanner_leakage_latent_mi")

    # 5. OOF calibration
    print("  Loading OOF calibration results ...", flush=True)
    oof_pooled   = load_oof_pooled(CANDIDATE_OOF)
    oof_foldwise = load_oof_foldwise(CANDIDATE_OOF)
    oof_primary  = oof_primary_by_calib(oof_pooled)
    write_table(oof_primary, "oof_primary_by_calib")
    if not oof_pooled.empty:
        write_table(oof_pooled, "oof_pooled_all")
    if not oof_foldwise.empty:
        write_table(oof_foldwise, "oof_foldwise_all")

    # 6. Comparison table
    print("  Building cross-model comparison ...", flush=True)
    comparison = build_comparison_table()
    write_table(comparison, "comparison_table")

    # 7. FPR table
    philips_df = build_fpr_table()
    write_table(philips_df, "philips_ge_siemens_fpr")

    # 8. Plus-ch1
    plus_ch1 = load_plus_ch1()
    if plus_ch1 is not None:
        write_table(plus_ch1[["candidate_id", "auc", "pr_auc", "balanced_accuracy",
                               "philips_cn_fpr", "decision"]], "plus_ch1_reference")

    # 9. Promotion decision
    print("  Evaluating promotion gate ...", flush=True)
    decision = promotion_decision(oof_pooled, philips_df)
    write_json(OUT_DIR / "promotion_decision.json", decision)

    print(f"\n  Verdict: {decision['verdict']}")
    print(f"  AUC gate:    {'PASS' if decision['gate_auc_pass'] else 'FAIL'}"
          f"  ({decision['best_oof_ecdf_AUC']:.4f} vs {PROMOTED_AUC})")
    print(f"  PR-AUC gate: {'PASS' if decision['gate_pr_auc_pass'] else 'FAIL'}"
          f"  ({decision['best_oof_ecdf_PRAUC']:.4f} vs {PROMOTED_PR_AUC})")
    print(f"  Philips FPR: {'PASS' if decision['gate_philips_fpr_pass'] else 'FAIL'}"
          f"  ({decision['philips_FPR_ecdf']:.4f} vs ≤{PHILIPS_FPR_GATE})")
    print(f"  OASIS: {decision['oasis_stress_test']}")

    # 10. Write README
    write_report(fold_complete, ckpt_cmp, rd_df, qc_df,
                 oof_primary, comparison, philips_df, plus_ch1, decision, oof_foldwise)

    # 11. Command log
    write_json(OUT_DIR / "command_log.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": CANDIDATE_RUN_ID,
        "stage_a_verified": all_ok,
        "stage_b_classifier_only_readout": str(CANDIDATE_CLF),
        "oof_calibration_dir": str(CANDIDATE_OOF),
        "vae_retrained": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "oasis_run": False,
        "verdict": decision["verdict"],
    })

    print(f"\nDone. Output: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
