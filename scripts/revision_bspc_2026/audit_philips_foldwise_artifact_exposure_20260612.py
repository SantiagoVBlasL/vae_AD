#!/usr/bin/env python3
"""Read-only fold-wise artifact/protocol exposure audit for Philips CN errors.

This audit integrates promoted-model fold membership with Martin slice-timing
metadata flags. It does not retrain models, modify tensors or metadata, edit
predictions, refit thresholds, or exclude subjects.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "revision_bspc_2026"
OUT = RESULTS / "philips_foldwise_artifact_exposure_audit_20260612"
OUT.mkdir(parents=True, exist_ok=True)

MASTER_PATH = RESULTS / "promoted_model_master_database_20260610" / "promoted_model_master_database.csv"
SLICE_MERGED_PATH = RESULTS / "philips_fmri_slice_timing_audit_20260612" / "philips_cn_fmri_slice_timing_merged.csv"
SLICE_TABLE_PATH = RESULTS / "philips_fmri_slice_timing_audit_20260612" / "philips_cn_slice_order_subject_table.csv"
RUN_DIR = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
OOF_PATH = RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration" / "calib_predictions.csv"


def write_md(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_df(stem: str, df: pd.DataFrame, md_rows: int | None = 200) -> None:
    df.to_csv(OUT / f"{stem}.csv", index=False)
    if df.empty:
        md = "_No rows._\n"
    else:
        show = df if md_rows is None or len(df) <= md_rows else df.head(md_rows)
        md = show.to_markdown(index=False) + "\n"
        if md_rows is not None and len(df) > md_rows:
            md += f"\n_Only first {md_rows} of {len(df)} rows shown. See CSV for full table._\n"
    (OUT / f"{stem}.md").write_text(md, encoding="utf-8")


def norm_str(x: Any) -> str:
    if pd.isna(x):
        return "MISSING"
    s = str(x).strip()
    return s if s else "MISSING"


def as_bool(x: Any) -> bool:
    if pd.isna(x):
        return False
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    return str(x).strip().lower() in {"1", "true", "yes", "y", "t"}


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for p in [MASTER_PATH, SLICE_MERGED_PATH, SLICE_TABLE_PATH, OOF_PATH, RUN_DIR]:
        if not Path(p).exists():
            raise FileNotFoundError(p)
    master = pd.read_csv(MASTER_PATH, low_memory=False)
    slice_merged = pd.read_csv(SLICE_MERGED_PATH, low_memory=False)
    slice_table = pd.read_csv(SLICE_TABLE_PATH, low_memory=False)
    oof = pd.read_csv(OOF_PATH, low_memory=False)
    return master, slice_merged, slice_table, oof


def first_present(df: pd.DataFrame, cols: list[str], default: Any = np.nan) -> pd.Series:
    out = pd.Series(np.nan, index=df.index)
    for col in cols:
        if col in df.columns:
            out = out.where(out.notna(), df[col])
    return out.fillna(default)


def build_subject_flags(master: pd.DataFrame, slice_merged: pd.DataFrame, oof: pd.DataFrame) -> pd.DataFrame:
    base = master.copy()
    base["SubjectID"] = base["SubjectID"].astype(str)
    base["tensor_idx"] = pd.to_numeric(base["tensor_idx"], errors="coerce")
    base["diagnosis_group"] = base["ResearchGroup_Mapped"].map(norm_str)
    base["manufacturer_norm"] = first_present(base, ["Manufacturer_normalized", "Manufacturer"], "MISSING").map(norm_str)
    base["site3_norm"] = first_present(base, ["Site3", "SITE"], np.nan).map(lambda x: "MISSING" if pd.isna(x) else str(int(float(x))) if str(x).replace(".", "", 1).isdigit() else str(x))
    base["raw_tp_group_norm"] = first_present(base, ["raw_tp_group", "n_tp_label", "n_tp_raw"], "UNKNOWN").map(norm_str)
    base["ORIGPROT_norm"] = first_present(base, ["ORIGPROT", "ORIGPROT_phil"], "MISSING").map(norm_str)
    base["COLPROT_norm"] = first_present(base, ["COLPROT", "COLPROT_phil"], "MISSING").map(norm_str)
    base["Sex_norm"] = first_present(base, ["Sex", "PTGENDER"], "MISSING").map(norm_str)
    base["Age_num"] = pd.to_numeric(base.get("Age"), errors="coerce")
    base["y_score_final_num"] = pd.to_numeric(base.get("y_score_final"), errors="coerce")
    base["y_pred_num"] = pd.to_numeric(base.get("y_pred"), errors="coerce")
    base["y_true_num"] = pd.to_numeric(base.get("y_true"), errors="coerce")
    base["confusion_label_norm"] = base.get("confusion_label", pd.Series(index=base.index)).map(norm_str)

    oof_primary = oof[
        oof["model_name"].astype(str).eq("logreg_l2_original")
        & oof["feature_set"].astype(str).eq("z_plus_age_sex")
        & oof["calib_method"].astype(str).eq("oof_ecdf")
        & oof["threshold_strategy"].astype(str).eq("inner_oof_target_sens_ge_0p70_max_spec")
    ].copy()
    if oof_primary.empty:
        oof_primary = oof[
            oof["model_name"].astype(str).eq("logreg_l2_original")
            & oof["feature_set"].astype(str).eq("z_plus_age_sex")
            & oof["calib_method"].astype(str).eq("oof_ecdf")
        ].copy()
    oof_primary = oof_primary.drop_duplicates("SubjectID")
    oof_cols = ["SubjectID", "fold", "y_score", "y_pred", "threshold"]
    base = base.merge(oof_primary[oof_cols], on="SubjectID", how="left", suffixes=("", "_oof_ecdf"))
    base["outer_fold_final"] = pd.to_numeric(base.get("outer_fold"), errors="coerce").where(
        pd.to_numeric(base.get("outer_fold"), errors="coerce").notna(),
        pd.to_numeric(base.get("fold"), errors="coerce"),
    )
    base["y_score_final_num"] = base["y_score_final_num"].where(base["y_score_final_num"].notna(), pd.to_numeric(base.get("y_score"), errors="coerce"))
    base["y_pred_num"] = base["y_pred_num"].where(base["y_pred_num"].notna(), pd.to_numeric(base.get("y_pred_oof_ecdf"), errors="coerce"))

    sm_cols = [
        "SubjectID",
        "match_method",
        "match_confidence",
        "fmri_slice_order_class",
        "fmri_slice_order_len",
        "fmri_matches_dparsf_default",
        "stc_mismatch_risk",
        "fmri_PHASEDIR_norm",
        "fmri_model_norm",
        "fmri_scanner_model_norm",
        "fmri_software_version_norm",
        "fmri_MEANTSNR",
        "fmri_MEDTSNR",
        "fmri_SDTSNR",
    ]
    sm = slice_merged[[c for c in sm_cols if c in slice_merged.columns]].copy()
    if "SubjectID" in sm:
        sm["SubjectID"] = sm["SubjectID"].astype(str)
        sm = sm.drop_duplicates("SubjectID")
        base = base.merge(sm, on="SubjectID", how="left", suffixes=("", "_fmri"))

    base["philips_cn"] = base["manufacturer_norm"].str.upper().eq("PHILIPS") & base["diagnosis_group"].eq("CN")
    base["philips_fp"] = base["philips_cn"] & base["confusion_label_norm"].eq("FP")
    base["philips_tn"] = base["philips_cn"] & base["confusion_label_norm"].eq("TN")
    base["high_confidence_slice_timing_match"] = base.get("match_confidence", pd.Series(index=base.index)).eq("high")
    base["slice_order_class"] = base.get("fmri_slice_order_class", pd.Series(index=base.index)).map(norm_str)
    base["stc_mismatch_risk"] = base.get("stc_mismatch_risk", pd.Series(False, index=base.index)).map(as_bool)
    base["Site31_reverse_even_odd_48"] = base["philips_cn"] & base["site3_norm"].eq("31") & base["slice_order_class"].eq("reverse_even_odd_48") & base["high_confidence_slice_timing_match"]
    base["Site2_default_7of7_pattern"] = base["philips_cn"] & base["site3_norm"].eq("2")
    base["raw_tp_140"] = base["raw_tp_group_norm"].astype(str).str.contains("140", case=False, na=False)
    base["raw_tp_197"] = base["raw_tp_group_norm"].astype(str).str.contains("197", case=False, na=False)
    problem_site = base.get("philips_problem_site_flag", pd.Series(False, index=base.index)).map(as_bool)
    base["problem_site_flag"] = problem_site | (base["philips_cn"] & base["site3_norm"].isin(["2", "13", "301", "31", "18"]))
    base["PHASEDIR_AP"] = base.get("fmri_PHASEDIR_norm", pd.Series(index=base.index)).map(norm_str).eq("AP")
    base["PHASEDIR_PA"] = base.get("fmri_PHASEDIR_norm", pd.Series(index=base.index)).map(norm_str).eq("PA")
    model = base.get("fmri_model_norm", pd.Series(index=base.index)).map(norm_str)
    fallback_model = first_present(base, ["manufacturer_model_name", "scanner_model"], "MISSING").map(norm_str)
    base["scanner_model_flag_value"] = model.where(~model.eq("MISSING"), fallback_model)
    for cat in ["Intera", "GEMINI", "Achieva", "Ingenia", "Ingenuity"]:
        base[f"scanner_model_{cat}"] = base["scanner_model_flag_value"].eq(cat)
    base["scanner_model_missing"] = base["scanner_model_flag_value"].eq("MISSING")
    base["has_promoted_oof_score"] = base["y_score_final_num"].notna()

    keep = [
        "SubjectID",
        "tensor_idx",
        "diagnosis_group",
        "manufacturer_norm",
        "site3_norm",
        "raw_tp_group_norm",
        "ORIGPROT_norm",
        "COLPROT_norm",
        "Age_num",
        "Sex_norm",
        "outer_fold_final",
        "y_true_num",
        "y_score_final_num",
        "y_pred_num",
        "confusion_label_norm",
        "philips_cn",
        "philips_fp",
        "philips_tn",
        "high_confidence_slice_timing_match",
        "match_method",
        "match_confidence",
        "slice_order_class",
        "fmri_slice_order_len",
        "stc_mismatch_risk",
        "Site31_reverse_even_odd_48",
        "Site2_default_7of7_pattern",
        "raw_tp_140",
        "raw_tp_197",
        "problem_site_flag",
        "PHASEDIR_AP",
        "PHASEDIR_PA",
        "scanner_model_flag_value",
        "scanner_model_Intera",
        "scanner_model_GEMINI",
        "scanner_model_Achieva",
        "scanner_model_Ingenia",
        "scanner_model_Ingenuity",
        "scanner_model_missing",
        "fmri_MEANTSNR",
        "fmri_MEDTSNR",
        "fmri_SDTSNR",
        "has_promoted_oof_score",
    ]
    return base[[c for c in keep if c in base.columns]].copy()


def rows_for_tensor_indices(flags: pd.DataFrame, tensor_idx: np.ndarray, fold: int, split: str) -> pd.DataFrame:
    sub = flags[flags["tensor_idx"].isin(pd.to_numeric(pd.Series(tensor_idx), errors="coerce"))].copy()
    sub["outer_fold"] = fold
    sub["split"] = split
    return sub


def reconstruct_membership(flags: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for fold in range(1, 6):
        fdir = RUN_DIR / f"fold_{fold}"
        pool = np.load(fdir / "vae_training_pool_tensor_idx.npy", allow_pickle=False)
        actual_local = np.load(fdir / "vae_actual_train_idx_local_to_pool.npy", allow_pickle=False)
        val_local = np.load(fdir / "vae_internal_val_idx_local_to_pool.npy", allow_pickle=False)
        rows.append(rows_for_tensor_indices(flags, pool, fold, "vae_train_dev_pool"))
        rows.append(rows_for_tensor_indices(flags, pool[actual_local], fold, "vae_actual_train"))
        rows.append(rows_for_tensor_indices(flags, pool[val_local], fold, "vae_internal_val"))
        train_stageb = pd.read_csv(fdir / "train_dev_subjects_fold.csv")
        test_stageb = pd.read_csv(fdir / "test_subjects_fold.csv")
        rows.append(rows_for_tensor_indices(flags, train_stageb["tensor_idx"].to_numpy(), fold, "stageB_train"))
        rows.append(rows_for_tensor_indices(flags, test_stageb["tensor_idx"].to_numpy(), fold, "stageB_test_oof"))
    membership = pd.concat(rows, ignore_index=True)
    membership = membership.sort_values(["outer_fold", "split", "SubjectID"]).reset_index(drop=True)
    return membership


FLAG_COLS = [
    "stc_mismatch_risk",
    "Site31_reverse_even_odd_48",
    "Site2_default_7of7_pattern",
    "raw_tp_140",
    "raw_tp_197",
    "problem_site_flag",
    "PHASEDIR_AP",
    "PHASEDIR_PA",
    "scanner_model_Intera",
    "scanner_model_GEMINI",
    "scanner_model_Achieva",
    "scanner_model_Ingenia",
    "scanner_model_Ingenuity",
    "scanner_model_missing",
]


def exposure_counts(membership: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (fold, split), sub in membership.groupby(["outer_fold", "split"], dropna=False):
        row: dict[str, Any] = {"outer_fold": int(fold), "split": split, "N_total": len(sub)}
        for dx in ["CN", "AD", "MCI"]:
            row[f"N_dx_{dx}"] = int(sub["diagnosis_group"].eq(dx).sum())
        row["N_Philips"] = int(sub["manufacturer_norm"].str.upper().eq("PHILIPS").sum())
        row["N_Philips_CN"] = int(sub["philips_cn"].sum())
        for col in FLAG_COLS:
            row[f"N_{col}"] = int(sub[col].fillna(False).astype(bool).sum()) if col in sub else 0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["outer_fold", "split"])


def binary_fisher(sub: pd.DataFrame, flag: str, positive_mask: pd.Series, negative_mask: pd.Series, context: str, fold: int) -> dict[str, Any]:
    flag_values = sub[flag].fillna(False).astype(bool)
    a = int((positive_mask & flag_values).sum())
    b = int((positive_mask & ~flag_values).sum())
    c = int((negative_mask & flag_values).sum())
    d = int((negative_mask & ~flag_values).sum())
    n_pos = a + b
    n_neg = c + d
    positive_label = "AD+MCI" if "ADplusMCI" in context else "AD"
    out = {
        "outer_fold": fold,
        "context": context,
        "flag": flag,
        "positive_label": positive_label,
        "negative_label": "CN",
        "positive_N": n_pos,
        "negative_N": n_neg,
        "flag_positive_N": a,
        "flag_negative_N": c,
        "flag_rate_positive": a / n_pos if n_pos else np.nan,
        "flag_rate_negative": c / n_neg if n_neg else np.nan,
        "test": "Fisher exact 2x2",
        "p_raw": np.nan,
        "odds_ratio": np.nan,
        "low_count_caution": min(a, b, c, d) < 5,
    }
    if n_pos and n_neg and len({a + c, b + d}) > 1:
        try:
            odds, p = stats.fisher_exact([[a, b], [c, d]])
            out["p_raw"] = float(p)
            out["odds_ratio"] = float(odds)
        except Exception as exc:  # noqa: BLE001
            out["test"] = f"not_tested: {exc}"
    return out


def shortcut_risk(membership: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        stage = membership[(membership["outer_fold"].eq(fold)) & (membership["split"].eq("stageB_train"))].copy()
        stage = stage[stage["diagnosis_group"].isin(["CN", "AD"])]
        for flag in FLAG_COLS:
            rows.append(
                binary_fisher(
                    stage,
                    flag,
                    stage["diagnosis_group"].eq("AD"),
                    stage["diagnosis_group"].eq("CN"),
                    "stageB_train_CN_vs_AD",
                    fold,
                )
            )
        vae = membership[(membership["outer_fold"].eq(fold)) & (membership["split"].eq("vae_actual_train"))].copy()
        for flag in FLAG_COLS:
            rows.append(
                binary_fisher(
                    vae,
                    flag,
                    vae["diagnosis_group"].isin(["AD", "MCI"]),
                    vae["diagnosis_group"].eq("CN"),
                    "vae_actual_train_CN_vs_ADplusMCI",
                    fold,
                )
            )
    return pd.DataFrame(rows)


def stageb_test_error_by_flag(membership: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    test = membership[membership["split"].eq("stageB_test_oof")].copy()
    ph_cn = test[test["philips_cn"]].copy()
    for fold in range(1, 6):
        sub = ph_cn[ph_cn["outer_fold"].eq(fold)]
        for flag in FLAG_COLS:
            for val, label in [(True, "flag_true"), (False, "flag_false")]:
                s = sub[sub[flag].fillna(False).astype(bool).eq(val)]
                n = len(s)
                fp = int(s["confusion_label_norm"].eq("FP").sum())
                scores = pd.to_numeric(s["y_score_final_num"], errors="coerce")
                rows.append(
                    {
                        "outer_fold": fold,
                        "flag": flag,
                        "flag_value": label,
                        "N_philips_cn": n,
                        "N_FP": fp,
                        "FPR": fp / n if n else np.nan,
                        "score_mean": float(scores.mean()) if scores.notna().any() else np.nan,
                        "score_median": float(scores.median()) if scores.notna().any() else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def high_score_fp_table(membership: pd.DataFrame) -> pd.DataFrame:
    test = membership[membership["split"].eq("stageB_test_oof")].copy()
    hs = test[test["philips_fp"] & (pd.to_numeric(test["y_score_final_num"], errors="coerce") > 0.75)].copy()
    cols = [
        "outer_fold",
        "SubjectID",
        "tensor_idx",
        "diagnosis_group",
        "manufacturer_norm",
        "site3_norm",
        "raw_tp_group_norm",
        "ORIGPROT_norm",
        "COLPROT_norm",
        "Age_num",
        "Sex_norm",
        "y_score_final_num",
        "y_pred_num",
        "confusion_label_norm",
        "high_confidence_slice_timing_match",
        "match_confidence",
        "slice_order_class",
        "stc_mismatch_risk",
        "Site31_reverse_even_odd_48",
        "Site2_default_7of7_pattern",
        "raw_tp_140",
        "raw_tp_197",
        "problem_site_flag",
        "PHASEDIR_AP",
        "PHASEDIR_PA",
        "scanner_model_flag_value",
    ]
    return hs[[c for c in cols if c in hs.columns]].sort_values(["outer_fold", "y_score_final_num"], ascending=[True, False])


def value_set(df: pd.DataFrame, col: str) -> set[str]:
    return {norm_str(x) for x in df[col].dropna().tolist() if norm_str(x) != "MISSING"}


def train_test_overlap(membership: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fold in range(1, 6):
        vae_train = membership[(membership["outer_fold"].eq(fold)) & (membership["split"].eq("vae_actual_train"))]
        stage_train = membership[(membership["outer_fold"].eq(fold)) & (membership["split"].eq("stageB_train"))]
        test_fp = membership[(membership["outer_fold"].eq(fold)) & (membership["split"].eq("stageB_test_oof")) & (membership["philips_fp"])]
        vae_sets = {
            "site": value_set(vae_train, "site3_norm"),
            "model": value_set(vae_train, "scanner_model_flag_value"),
            "raw_tp": value_set(vae_train, "raw_tp_group_norm"),
            "origprot": value_set(vae_train, "ORIGPROT_norm"),
            "colprot": value_set(vae_train, "COLPROT_norm"),
        }
        stage_sets = {
            "site": value_set(stage_train, "site3_norm"),
            "model": value_set(stage_train, "scanner_model_flag_value"),
            "raw_tp": value_set(stage_train, "raw_tp_group_norm"),
            "origprot": value_set(stage_train, "ORIGPROT_norm"),
            "colprot": value_set(stage_train, "COLPROT_norm"),
        }
        for _, r in test_fp.iterrows():
            row: dict[str, Any] = {
                "outer_fold": fold,
                "SubjectID": r["SubjectID"],
                "Site3": r["site3_norm"],
                "scanner_model": r["scanner_model_flag_value"],
                "raw_tp_group": r["raw_tp_group_norm"],
                "ORIGPROT": r["ORIGPROT_norm"],
                "COLPROT": r["COLPROT_norm"],
                "y_score_final": r["y_score_final_num"],
                "stc_mismatch_risk": r["stc_mismatch_risk"],
                "Site31_reverse_even_odd_48": r["Site31_reverse_even_odd_48"],
                "Site2_default_7of7_pattern": r["Site2_default_7of7_pattern"],
            }
            for prefix, sets in [("vae_train", vae_sets), ("stageB_train", stage_sets)]:
                row[f"{prefix}_same_site_present"] = r["site3_norm"] in sets["site"]
                row[f"{prefix}_same_model_present"] = r["scanner_model_flag_value"] in sets["model"]
                row[f"{prefix}_same_raw_tp_present"] = r["raw_tp_group_norm"] in sets["raw_tp"]
                row[f"{prefix}_same_ORIGPROT_present"] = r["ORIGPROT_norm"] in sets["origprot"]
                row[f"{prefix}_same_COLPROT_present"] = r["COLPROT_norm"] in sets["colprot"]
                row[f"{prefix}_Site31_reverse_present"] = bool(vae_train["Site31_reverse_even_odd_48"].sum()) if prefix == "vae_train" else bool(stage_train["Site31_reverse_even_odd_48"].sum())
                row[f"{prefix}_Site2_pattern_present"] = bool(vae_train["Site2_default_7of7_pattern"].sum()) if prefix == "vae_train" else bool(stage_train["Site2_default_7of7_pattern"].sum())
                row[f"{prefix}_Intera_present"] = bool(vae_train["scanner_model_Intera"].sum()) if prefix == "vae_train" else bool(stage_train["scanner_model_Intera"].sum())
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_for_text(exposure: pd.DataFrame, shortcut: pd.DataFrame, overlap: pd.DataFrame, membership: pd.DataFrame) -> dict[str, Any]:
    test = membership[membership["split"].eq("stageB_test_oof")]
    ph_cn = test[test["philips_cn"]]
    return {
        "n_membership_rows": len(membership),
        "n_stageB_test_philips_cn": int(len(ph_cn)),
        "n_stageB_test_philips_fp": int(ph_cn["confusion_label_norm"].eq("FP").sum()),
        "n_stageB_test_stc_mismatch": int(ph_cn["stc_mismatch_risk"].sum()),
        "n_stageB_test_site31_reverse": int(ph_cn["Site31_reverse_even_odd_48"].sum()),
        "n_stageB_test_site2_pattern": int(ph_cn["Site2_default_7of7_pattern"].sum()),
        "n_high_score_philips_fp_gt0p75": int((ph_cn["philips_fp"] & (pd.to_numeric(ph_cn["y_score_final_num"], errors="coerce") > 0.75)).sum()),
        "n_overlap_rows": len(overlap),
    }


def executive_summary(summary: dict[str, Any], exposure: pd.DataFrame, shortcut: pd.DataFrame, test_by_flag: pd.DataFrame) -> str:
    top_shortcut = shortcut[
        shortcut["context"].eq("stageB_train_CN_vs_AD")
        & shortcut["p_raw"].notna()
        & (shortcut["flag_rate_positive"] != shortcut["flag_rate_negative"])
    ].sort_values("p_raw").head(12)
    site31_exp = exposure[["outer_fold", "split", "N_Site31_reverse_even_odd_48"]] if "N_Site31_reverse_even_odd_48" in exposure else pd.DataFrame()
    stc_test = test_by_flag[(test_by_flag["flag"].eq("stc_mismatch_risk")) & (test_by_flag["flag_value"].eq("flag_true"))]
    return f"""# 00 Executive Summary - Philips Foldwise Artifact Exposure Audit

**Generated**: {datetime.now(timezone.utc).isoformat()}

## Scope

Read-only fold-wise audit of whether Philips acquisition/protocol flags were
present in VAE training, VAE validation, StageB training, and StageB OOF test
splits for the promoted ADNI model.

## Core Counts

- Membership rows reconstructed across fold/split views: {summary['n_membership_rows']}
- Philips CN in StageB OOF test: {summary['n_stageB_test_philips_cn']}
- Philips CN FP in StageB OOF test: {summary['n_stageB_test_philips_fp']}
- Philips CN STC mismatch-risk in StageB OOF test: {summary['n_stageB_test_stc_mismatch']}
- Philips CN Site31 reverse-even-odd in StageB OOF test: {summary['n_stageB_test_site31_reverse']}
- Philips CN Site2 pattern in StageB OOF test: {summary['n_stageB_test_site2_pattern']}
- High-score Philips FP (`y_score_final > 0.75`): {summary['n_high_score_philips_fp_gt0p75']}

## Interpretation Preview

The audit supports an artifact/protocol exposure-risk framing rather than a
causal conclusion. Site31 reverse-even-odd subjects are visible as an objective
STC mismatch-risk stratum, but Site2 remains a high-FPR default-slice-order
stratum, so slice order cannot explain the full Philips shift. StageB shortcut
risk should be interpreted separately from VAE representation-learning risk.

## Site31 Reverse Exposure Snapshot

{site31_exp.to_markdown(index=False) if not site31_exp.empty else '_No Site31 reverse exposure rows._'}

## StageB Train Shortcut-Risk Flags With Smallest Raw P-Values

{top_shortcut.to_markdown(index=False) if not top_shortcut.empty else '_No tested shortcut rows._'}

## StageB Test Philips CN FPR for STC Mismatch-Risk

{stc_test.to_markdown(index=False) if not stc_test.empty else '_No STC mismatch test rows._'}
"""


def final_interpretation(summary: dict[str, Any], shortcut: pd.DataFrame, overlap: pd.DataFrame) -> str:
    stage_sig = shortcut[
        shortcut["context"].eq("stageB_train_CN_vs_AD")
        & shortcut["p_raw"].notna()
        & (shortcut["p_raw"] < 0.05)
        & (shortcut["flag_rate_positive"] != shortcut["flag_rate_negative"])
    ].sort_values("p_raw")
    vae_sig = shortcut[
        shortcut["context"].eq("vae_actual_train_CN_vs_ADplusMCI")
        & shortcut["p_raw"].notna()
        & (shortcut["p_raw"] < 0.05)
        & (shortcut["flag_rate_positive"] != shortcut["flag_rate_negative"])
    ].sort_values("p_raw")
    overlap_all = int(overlap[["vae_train_same_site_present", "stageB_train_same_site_present"]].all(axis=1).sum()) if not overlap.empty else 0
    return f"""# Final Interpretation

This audit is descriptive and read-only. It does not prove causality and does
not justify post-hoc subject exclusion, threshold changes, or model selection.

## VAE Representation-Learning Risk

VAE actual-train splits repeatedly contain Philips/site/rawTP/protocol strata,
including artifact/protocol flags that can be associated with CN versus
AD+MCI composition. This supports a **VAE artifact exposure risk** framing:
the diagnosis-agnostic VAE may encode protocol directions if those directions
are present in fold-local training data. The strongest VAE shortcut-risk rows
at p<0.05 are:

{vae_sig.head(15).to_markdown(index=False) if not vae_sig.empty else '_No VAE shortcut-risk rows had raw p<0.05._'}

## StageB Classifier Shortcut Risk

StageB train splits also contain protocol/site/model flags with CN-vs-AD
imbalance. This supports a **classifier shortcut risk** framing for flags that
are separable in StageB train. The strongest StageB shortcut-risk rows at
p<0.05 are:

{stage_sig.head(15).to_markdown(index=False) if not stage_sig.empty else '_No StageB shortcut-risk rows had raw p<0.05._'}

## Independent Test-Domain Shift

High-score Philips CN false positives include strata whose site/model/protocol
levels are already present in VAE and/or StageB train, but Site2 default-slice
order remains 7/7 FP in the slice-timing audit. This means slice-order mismatch
is not sufficient as a single explanation. The safer interpretation is a
multi-factor protocol-domain shift: site, raw timepoint group, scanner model,
age, and channel-level connectivity shifts.

Philips FP rows with both VAE-train and StageB-train same-site exposure:
`{overlap_all}` of `{len(overlap)}`.

## Evidence Classification

- A) Artifact learned in VAE train: **plausible exposure risk**, not proven.
- B) Classifier shortcut in StageB train: **plausible where flags are diagnosis-imbalanced**, not proven.
- C) Independent test-domain shift: **supported**, because Site2/default-slice-order and scanner/model/site strata remain problematic.
- D) Insufficient coverage: **also supported**, because only 40/99 Philips CN had high-confidence slice-timing matches.

The promoted model should remain unchanged.
"""


def martin_next_actions() -> str:
    return """# Martin Follow-Up Next Actions

1. Prioritize manual review of high-score Philips CN FP subjects in
   `high_score_philips_fp_artifact_table.csv`, especially those with Site2,
   Intera/GEMINI, 140TP, or Site31 reverse-even-odd flags.
2. Ask Martín to annotate unmatched Philips CN subjects where possible, because
   high-confidence slice-timing coverage is currently incomplete.
3. For Site31 reverse-even-odd subjects, verify whether DPARSF slice-timing
   correction used the odd/even default or a site-specific override.
4. For Site2, focus beyond slice order: scanner model, protocol/rawTP, ADNI
   phase, age, and tensor-channel shifts.
5. Treat any future exclusion or preprocessing correction as a new
   pre-specified sensitivity analysis, not as a post-hoc edit to the promoted
   model.
"""


def main() -> None:
    master, slice_merged, _slice_table, oof = load_inputs()
    flags = build_subject_flags(master, slice_merged, oof)
    flags.to_csv(OUT / "subject_artifact_flags_master.csv", index=False)

    membership = reconstruct_membership(flags)
    write_df("fold_membership_reconstruction", membership, md_rows=250)

    exposure = exposure_counts(membership)
    write_df("foldwise_artifact_exposure_counts", exposure, md_rows=None)

    shortcut = shortcut_risk(membership)
    write_df("stageB_train_shortcut_risk_by_fold", shortcut, md_rows=300)

    test_by_flag = stageb_test_error_by_flag(membership)
    write_df("stageB_test_philips_cn_error_by_flag", test_by_flag, md_rows=300)

    high_score = high_score_fp_table(membership)
    write_df("high_score_philips_fp_artifact_table", high_score, md_rows=None)

    overlap = train_test_overlap(membership)
    write_df("site_model_protocol_train_test_overlap", overlap, md_rows=None)

    summary = summarize_for_text(exposure, shortcut, overlap, membership)
    write_md(OUT / "00_EXECUTIVE_SUMMARY.md", executive_summary(summary, exposure, shortcut, test_by_flag))
    write_md(OUT / "final_interpretation.md", final_interpretation(summary, shortcut, overlap))
    write_md(OUT / "martin_followup_next_actions.md", martin_next_actions())
    write_json(
        OUT / "command_log.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "argv": ["scripts/revision_bspc_2026/audit_philips_foldwise_artifact_exposure_20260612.py"],
            "inputs": {
                "master": str(MASTER_PATH),
                "slice_merged": str(SLICE_MERGED_PATH),
                "slice_table": str(SLICE_TABLE_PATH),
                "run_dir": str(RUN_DIR),
                "oof_predictions": str(OOF_PATH),
            },
            "outputs": str(OUT.relative_to(PROJECT_ROOT)),
            "guardrails": {
                "read_only": True,
                "model_training": False,
                "tensor_edits": False,
                "metadata_edits": False,
                "prediction_edits": False,
                "threshold_refitting": False,
                "subject_exclusion": False,
                "modified_existing_artifacts": False,
            },
            "summary": summary,
        },
    )


if __name__ == "__main__":
    main()
