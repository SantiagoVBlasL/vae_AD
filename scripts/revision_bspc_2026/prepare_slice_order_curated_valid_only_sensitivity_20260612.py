#!/usr/bin/env python3
"""Preflight slice-order curated valid-only sensitivity for the promoted ADNI model.

This script is intentionally preflight-only. It creates derived mask files,
score-only retained-set summaries, a derived metadata CSV for the selected M1
exploratory sensitivity, and a guarded launcher. It does not train a model and
does not modify original tensors, metadata, predictions, thresholds, or promoted
artifacts.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "revision_bspc_2026"
OUT = RESULTS / "slice_order_curated_valid_only_sensitivity_20260612"

FULL_DB = (
    RESULTS
    / "full_database_for_martin_and_validity_preflight_20260612"
    / "promoted_model_full_database_for_martin_20260612.csv"
)
REFERENCE_CONFIG = (
    PROJECT_ROOT
    / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5.json"
)

RUN_NAME = "recover035_latent384_beta3p75_T80_h10000_p560_full5x5_sliceorderM1_validonly_sensitivity_20260612"
DERIVED_METADATA = OUT / "sliceorderM1_validonly_derived_metadata.csv"
M1_MANIFEST = OUT / "sliceorderM1_exclusion_manifest.csv"
CANDIDATE_CONFIG = OUT / f"adni_v5_1c_{RUN_NAME}.json"
GENERATED_LAUNCHER = OUT / f"run_{RUN_NAME}.py"
DRY_RUN_PASS = OUT / "DRY_RUN_PASS.txt"

EXPECTED_M1_SUBJECTS = [
    "031_S_4021",
    "031_S_4032",
    "031_S_4218",
    "031_S_4474",
    "031_S_4496",
    "018_S_6207",
    "018_S_6351",
    "301_S_6224",
    "301_S_6326",
    "301_S_6501",
]

MASKS = [
    "full_promoted_reference",
    "M0_confirmed_nondefault_only",
    "M1_pragmatic_cn_problem_site_unknown",
    "M2_broad_problem_site_unknown_all_dx",
    "M2_optional_broader_problem_sites_2_13_18_31_301",
    "M3_martin_literal_known_correct_only",
]

PROBLEM_SITES_M1_M2 = {"18", "301"}
PROBLEM_SITES_OPTIONAL = {"2", "13", "18", "31", "301"}
SEED = 42
OUTER_FOLDS = 5
VAE_VAL_SPLIT = 0.2


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def md_table(df: pd.DataFrame, max_rows: int = 80) -> str:
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


def write_df(stem: str, df: pd.DataFrame, max_rows: int = 80) -> None:
    df.to_csv(OUT / f"{stem}.csv", index=False)
    (OUT / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def norm_str(s: pd.Series) -> pd.Series:
    return s.fillna("MISSING").astype(str).str.strip().replace({"": "MISSING", "nan": "MISSING", "NaN": "MISSING"})


def bool_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    s = df[col]
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    return norm_str(s).str.lower().isin({"true", "1", "yes", "y"})


def site_col(df: pd.DataFrame) -> pd.Series:
    return norm_str(df["Site3_final"]).str.replace(r"\.0$", "", regex=True)


def is_philips(df: pd.DataFrame) -> pd.Series:
    return norm_str(df["Manufacturer_final"]).str.lower().eq("philips")


def is_cn(df: pd.DataFrame) -> pd.Series:
    return norm_str(df["diagnosis_group"]).eq("CN")


def is_dx(df: pd.DataFrame, dx: str) -> pd.Series:
    return norm_str(df["diagnosis_group"]).eq(dx)


def add_masks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_site3_norm"] = site_col(out)
    out["_manufacturer_norm"] = norm_str(out["Manufacturer_final"])
    out["_diagnosis_norm"] = norm_str(out["diagnosis_group"])
    out["_slice_norm"] = norm_str(out["slice_order_class"])
    out["_matches_default"] = bool_col(out, "matches_dparsf_default")

    phil = is_philips(out)
    cn = is_cn(out)
    missing_slice = out["_slice_norm"].eq("MISSING")
    reverse = out["_slice_norm"].eq("reverse_even_odd_48")

    exclude_m0 = phil & reverse
    exclude_m1 = exclude_m0 | (phil & cn & out["_site3_norm"].isin(PROBLEM_SITES_M1_M2) & missing_slice)
    exclude_m2 = exclude_m0 | (phil & out["_site3_norm"].isin(PROBLEM_SITES_M1_M2) & missing_slice)
    exclude_m2_optional = exclude_m0 | (phil & out["_site3_norm"].isin(PROBLEM_SITES_OPTIONAL) & missing_slice)
    exclude_m3 = phil & ~out["_matches_default"]

    out["retained_full_promoted_reference"] = True
    out["retained_M0_confirmed_nondefault_only"] = ~exclude_m0
    out["retained_M1_pragmatic_cn_problem_site_unknown"] = ~exclude_m1
    out["retained_M2_broad_problem_site_unknown_all_dx"] = ~exclude_m2
    out["retained_M2_optional_broader_problem_sites_2_13_18_31_301"] = ~exclude_m2_optional
    out["retained_M3_martin_literal_known_correct_only"] = ~exclude_m3

    reasons: dict[str, list[str]] = {}
    for idx, row in out.iterrows():
        rs = []
        if bool(exclude_m0.loc[idx]):
            rs.append("Philips_reverse_even_odd_48_confirmed_nondefault")
        if bool((phil & cn & out["_site3_norm"].isin(PROBLEM_SITES_M1_M2) & missing_slice).loc[idx]):
            rs.append("Philips_CN_Site18_or_301_missing_slice_order")
        if bool((phil & out["_site3_norm"].isin(PROBLEM_SITES_M1_M2) & missing_slice & ~cn).loc[idx]):
            rs.append("Philips_nonCN_Site18_or_301_missing_slice_order")
        if bool((phil & out["_site3_norm"].isin(PROBLEM_SITES_OPTIONAL) & missing_slice).loc[idx]):
            rs.append("Philips_optional_problem_site_2_13_18_31_301_missing_slice_order")
        if bool(exclude_m3.loc[idx]):
            rs.append("Philips_not_known_default_dparsf_slice_order")
        reasons[str(row["SubjectID"])] = rs
    out["_mask_reason_all"] = out["SubjectID"].astype(str).map(lambda sid: ";".join(reasons.get(sid, [])) or "retained")
    return out


def slice_coverage(df: pd.DataFrame) -> pd.DataFrame:
    phil = df[is_philips(df)].copy()
    rows: list[dict[str, Any]] = []

    def summarize(sub: pd.DataFrame, scope: str, diagnosis: str = "ALL", site: str = "ALL") -> None:
        known = ~norm_str(sub["slice_order_class"]).eq("MISSING")
        high = bool_col(sub, "high_confidence_slice_timing_match")
        rows.append({
            "scope": scope,
            "diagnosis": diagnosis,
            "Site3": site,
            "N": len(sub),
            "slice_known": int(known.sum()),
            "slice_missing": int((~known).sum()),
            "default_odd_even_48": int(norm_str(sub["slice_order_class"]).eq("default_odd_even_48").sum()),
            "reverse_even_odd_48": int(norm_str(sub["slice_order_class"]).eq("reverse_even_odd_48").sum()),
            "high_confidence_N": int(high.sum()),
            "high_confidence_known": int((high & known).sum()),
            "high_confidence_default": int((high & norm_str(sub["slice_order_class"]).eq("default_odd_even_48")).sum()),
            "high_confidence_reverse": int((high & norm_str(sub["slice_order_class"]).eq("reverse_even_odd_48")).sum()),
        })

    summarize(phil, "all_philips")
    for dx, sub in phil.groupby("_diagnosis_norm", dropna=False):
        summarize(sub, "by_diagnosis", diagnosis=str(dx))
    for site, sub in phil.groupby("_site3_norm", dropna=False):
        summarize(sub, "by_site", site=str(site))
    for (dx, site), sub in phil.groupby(["_diagnosis_norm", "_site3_norm"], dropna=False):
        summarize(sub, "by_diagnosis_site", diagnosis=str(dx), site=str(site))
    return pd.DataFrame(rows)


def excluded_subjects(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mask in MASKS[1:]:
        retained_col = f"retained_{mask}"
        sub = df.loc[~df[retained_col]].copy()
        for _, r in sub.iterrows():
            reason = r["_mask_reason_all"]
            if mask == "M0_confirmed_nondefault_only":
                reason = "Philips_reverse_even_odd_48_confirmed_nondefault"
            elif mask == "M1_pragmatic_cn_problem_site_unknown":
                parts = [p for p in str(r["_mask_reason_all"]).split(";") if p in {
                    "Philips_reverse_even_odd_48_confirmed_nondefault",
                    "Philips_CN_Site18_or_301_missing_slice_order",
                }]
                reason = ";".join(parts)
            elif mask == "M2_broad_problem_site_unknown_all_dx":
                parts = [p for p in str(r["_mask_reason_all"]).split(";") if p in {
                    "Philips_reverse_even_odd_48_confirmed_nondefault",
                    "Philips_CN_Site18_or_301_missing_slice_order",
                    "Philips_nonCN_Site18_or_301_missing_slice_order",
                }]
                reason = ";".join(parts)
            elif mask == "M3_martin_literal_known_correct_only":
                reason = "Philips_not_known_default_dparsf_slice_order"
            rows.append({
                "mask": mask,
                "SubjectID": r["SubjectID"],
                "diagnosis_group": r["diagnosis_group"],
                "Manufacturer": r["Manufacturer_final"],
                "Site3": r["Site3_final"],
                "raw_tp_group": r["raw_tp_group_final"],
                "slice_order_class": r["slice_order_class"],
                "matches_dparsf_default": r["matches_dparsf_default"],
                "high_confidence_slice_timing_match": r["high_confidence_slice_timing_match"],
                "y_score_final": r.get("y_score_final", np.nan),
                "confusion_label": r.get("confusion_label", ""),
                "reason": reason,
            })
    return pd.DataFrame(rows)


def retained_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mask in MASKS:
        retained = df[f"retained_{mask}"]
        for status, sub in [("retained", df[retained]), ("excluded", df[~retained])]:
            rows.append({"mask": mask, "status": status, "stratification": "total", "group": "ALL", "N": len(sub)})
            for strat, col in [
                ("diagnosis", "_diagnosis_norm"),
                ("Manufacturer", "_manufacturer_norm"),
                ("Site3", "_site3_norm"),
                ("raw_tp_group", "raw_tp_group_final"),
            ]:
                for group, n in norm_str(sub[col]).value_counts(dropna=False).sort_index().items():
                    rows.append({"mask": mask, "status": status, "stratification": strat, "group": str(group), "N": int(n)})
            phil = sub[is_philips(sub)]
            for group, n in norm_str(phil["diagnosis_group"]).value_counts(dropna=False).sort_index().items():
                rows.append({"mask": mask, "status": status, "stratification": "Philips_diagnosis", "group": str(group), "N": int(n)})
    return pd.DataFrame(rows)


def choose_stratification(clf: pd.DataFrame) -> tuple[np.ndarray, str, int]:
    y_label = clf["_diagnosis_norm"].map({"CN": 0, "AD": 1}).to_numpy()
    key = clf["_diagnosis_norm"].astype(str) + "_" + clf["_manufacturer_norm"].astype(str)
    vc = key.value_counts()
    if vc.empty or int(vc.min()) < OUTER_FOLDS:
        return y_label, "diagnosis_only_fallback", int(vc.min()) if not vc.empty else 0
    return key.to_numpy(), "diagnosis_plus_manufacturer", int(vc.min())


def split_train_val(vae_pool: pd.DataFrame, fold: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if len(vae_pool) <= 10:
        return vae_pool.copy(), vae_pool.iloc[0:0].copy(), "too_small_no_val"
    key = vae_pool["_diagnosis_norm"].astype(str) + "_" + vae_pool["_manufacturer_norm"].astype(str)
    if int(key.value_counts().min()) < 2:
        key = vae_pool["_diagnosis_norm"].astype(str)
        note = "diagnosis_only"
    else:
        note = "diagnosis_plus_manufacturer"
    train_idx, val_idx = train_test_split(
        np.arange(len(vae_pool)),
        test_size=VAE_VAL_SPLIT,
        random_state=SEED + (fold - 1) + 10,
        shuffle=True,
        stratify=key,
    )
    return vae_pool.iloc[train_idx].copy(), vae_pool.iloc[val_idx].copy(), note


def fold_counts(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    feasibility: list[dict[str, Any]] = []
    for mask in MASKS:
        retained = df[f"retained_{mask}"]
        ret = df[retained & df["_diagnosis_norm"].isin(["CN", "AD", "MCI"])].copy()
        clf = ret[ret["_diagnosis_norm"].isin(["CN", "AD"])].copy()
        if clf["_diagnosis_norm"].nunique() < 2:
            feasibility.append({"mask": mask, "fold": "ALL", "status": "FAIL", "reason": "classifier_pool_lacks_CN_or_AD"})
            continue
        strat_y, strat_note, min_stratum = choose_stratification(clf)
        y = clf["_diagnosis_norm"].map({"CN": 0, "AD": 1}).to_numpy()
        cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=SEED)
        mask_ok = True
        mask_reasons = []
        for fold, (tr_idx, te_idx) in enumerate(cv.split(np.arange(len(clf)), strat_y), start=1):
            train = clf.iloc[tr_idx].copy()
            test = clf.iloc[te_idx].copy()
            test_ids = set(test["SubjectID"].astype(str))
            vae_pool = ret[~ret["SubjectID"].astype(str).isin(test_ids)].copy()
            try:
                vae_train, vae_val, vae_strat_note = split_train_val(vae_pool, fold)
                val_failed = False
            except Exception as exc:
                vae_train, vae_val = vae_pool.copy(), vae_pool.iloc[0:0].copy()
                vae_strat_note = f"failed:{exc}"
                val_failed = True
                mask_ok = False
                mask_reasons.append(f"fold_{fold}_vae_val_split_failed")
            for split_name, sub in [
                ("stageB_train", train),
                ("stageB_test_oof", test),
                ("vae_train_dev_pool", vae_pool),
                ("vae_actual_train", vae_train),
                ("vae_internal_val", vae_val),
            ]:
                row: dict[str, Any] = {
                    "mask": mask,
                    "fold": fold,
                    "split": split_name,
                    "N": len(sub),
                    "N_CN": int((sub["_diagnosis_norm"] == "CN").sum()),
                    "N_AD": int((sub["_diagnosis_norm"] == "AD").sum()),
                    "N_MCI": int((sub["_diagnosis_norm"] == "MCI").sum()),
                    "N_GE": int((sub["_manufacturer_norm"] == "GE").sum()),
                    "N_Philips": int((sub["_manufacturer_norm"] == "Philips").sum()),
                    "N_SIEMENS": int((sub["_manufacturer_norm"] == "SIEMENS").sum()),
                    "stratification_note": strat_note if split_name.startswith("stageB") else vae_strat_note,
                    "min_outer_stratum": min_stratum,
                    "vae_val_split_failed": val_failed if split_name.startswith("vae") else False,
                }
                rows.append(row)
            checks = {
                "stageB_train_has_CN_AD": train["_diagnosis_norm"].isin(["CN", "AD"]).groupby(train["_diagnosis_norm"]).size().reindex(["CN", "AD"], fill_value=0).min() > 0,
                "stageB_test_has_CN_AD": test["_diagnosis_norm"].isin(["CN", "AD"]).groupby(test["_diagnosis_norm"]).size().reindex(["CN", "AD"], fill_value=0).min() > 0,
                "vae_pool_has_CN_AD_MCI": all((vae_pool["_diagnosis_norm"] == dx).any() for dx in ["CN", "AD", "MCI"]),
                "train_has_all_mfr": set(train["_manufacturer_norm"]) >= {"GE", "Philips", "SIEMENS"},
                "test_has_all_mfr": set(test["_manufacturer_norm"]) >= {"GE", "Philips", "SIEMENS"},
            }
            bad = [k for k, v in checks.items() if not bool(v)]
            if bad:
                mask_ok = False
                mask_reasons.extend([f"fold_{fold}_{b}" for b in bad])
            feasibility.append({
                "mask": mask,
                "fold": fold,
                "status": "PASS" if not bad and not val_failed else "FAIL",
                "reason": "PASS" if not bad and not val_failed else ";".join(bad + (["vae_val_split_failed"] if val_failed else [])),
                **{k: bool(v) for k, v in checks.items()},
            })
        feasibility.append({
            "mask": mask,
            "fold": "ALL",
            "status": "PASS" if mask_ok else "FAIL",
            "reason": "PASS" if mask_ok else ";".join(sorted(set(mask_reasons))),
        })
    return pd.DataFrame(rows), pd.DataFrame(feasibility)


def confusion_metrics(y: np.ndarray, score: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    if len(np.unique(y)) == 2:
        out["AUC"] = float(roc_auc_score(y, score))
        out["PR_AUC"] = float(average_precision_score(y, score))
    else:
        out["AUC"] = np.nan
        out["PR_AUC"] = np.nan
    out["BA"] = float(balanced_accuracy_score(y, pred)) if len(y) else np.nan
    out["Sensitivity"] = float(recall_score(y, pred, pos_label=1, zero_division=0)) if len(y) else np.nan
    out["Specificity"] = float(recall_score(y, pred, pos_label=0, zero_division=0)) if len(y) else np.nan
    out["F1"] = float(f1_score(y, pred, zero_division=0)) if len(y) else np.nan
    out["Brier"] = float(brier_score_loss(y, score)) if len(y) else np.nan
    out["N"] = int(len(y))
    out["TN"] = int(((y == 0) & (pred == 0)).sum())
    out["FP"] = int(((y == 0) & (pred == 1)).sum())
    out["FN"] = int(((y == 1) & (pred == 0)).sum())
    out["TP"] = int(((y == 1) & (pred == 1)).sum())
    return out


def score_only(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    fpr_rows = []
    for mask in MASKS:
        sub = df[
            df[f"retained_{mask}"]
            & df["_diagnosis_norm"].isin(["CN", "AD"])
            & df["y_score_final"].notna()
            & df["y_pred"].notna()
        ].copy()
        y = sub["_diagnosis_norm"].map({"CN": 0, "AD": 1}).to_numpy(dtype=int)
        score = pd.to_numeric(sub["y_score_final"], errors="coerce").to_numpy(dtype=float)
        pred = pd.to_numeric(sub["y_pred"], errors="coerce").to_numpy(dtype=int)
        row = {"mask": mask, **confusion_metrics(y, score, pred)}
        row["note"] = "locked promoted OOF scores recomputed on retained subjects; not a new model"
        rows.append(row)
        cn = sub[sub["_diagnosis_norm"].eq("CN")].copy()
        for mfr, msub in cn.groupby("_manufacturer_norm", dropna=False):
            den = len(msub)
            fp = int((pd.to_numeric(msub["y_pred"], errors="coerce") == 1).sum())
            fpr_rows.append({
                "mask": mask,
                "Manufacturer": str(mfr),
                "CN_N": den,
                "FP": fp,
                "FPR": fp / den if den else np.nan,
            })
        phil_cn = cn[cn["_manufacturer_norm"].eq("Philips")]
        den = len(phil_cn)
        fp = int((pd.to_numeric(phil_cn["y_pred"], errors="coerce") == 1).sum())
        fpr_rows.append({
            "mask": mask,
            "Manufacturer": "Philips_CN_overall",
            "CN_N": den,
            "FP": fp,
            "FPR": fp / den if den else np.nan,
        })
    return pd.DataFrame(rows), pd.DataFrame(fpr_rows)


def create_derived_metadata(df: pd.DataFrame, ref_cfg: dict[str, Any]) -> pd.DataFrame:
    metadata_path = Path(ref_cfg["paths"]["metadata_path"])
    meta = pd.read_csv(metadata_path)
    meta["SubjectID"] = meta["SubjectID"].astype(str).str.strip()
    excluded = set(df.loc[~df["retained_M1_pragmatic_cn_problem_site_unknown"], "SubjectID"].astype(str))
    meta["sliceorder_m1_excluded"] = meta["SubjectID"].isin(excluded)
    meta["sliceorder_m1_exclusion_reason"] = meta["SubjectID"].map(
        df.set_index("SubjectID")["_mask_reason_all"].to_dict()
    ).where(meta["sliceorder_m1_excluded"], "")
    meta["sliceorder_m1_vae_eligible"] = np.where(meta["sliceorder_m1_excluded"], "", "eligible")
    meta["ResearchGroup_Mapped_original"] = meta["ResearchGroup_Mapped"]
    meta.loc[meta["sliceorder_m1_excluded"], "ResearchGroup_Mapped"] = "EXCLUDED_SLICEORDER_M1"
    meta.loc[meta["sliceorder_m1_excluded"], "training_ready"] = False
    meta.loc[meta["sliceorder_m1_excluded"], "exclude_from_supervised"] = True
    if "supervised_exclusion_reason" not in meta.columns:
        meta["supervised_exclusion_reason"] = ""
    meta["supervised_exclusion_reason"] = meta["supervised_exclusion_reason"].fillna("").astype(str)
    meta.loc[meta["sliceorder_m1_excluded"], "supervised_exclusion_reason"] = (
        meta.loc[meta["sliceorder_m1_excluded"], "supervised_exclusion_reason"]
        + ";sliceorderM1_validonly_sensitivity_exclusion"
    ).str.strip(";")
    return meta


def create_config(ref_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = json.loads(json.dumps(ref_cfg))
    cfg["run_name"] = RUN_NAME
    cfg["description"] = (
        "Exploratory QC sensitivity: same promoted [1,0,2] latent384 beta3.75 T80 p560 FULL 5x5, "
        "but using a derived metadata CSV that excludes the M1 Philips CN slice-order/problem-site subjects "
        "from both VAE and StageB pools. This is not a primary replacement model."
    )
    cfg["paths"]["metadata_path"] = str(DERIVED_METADATA.relative_to(PROJECT_ROOT))
    cfg["paths"]["output_dir"] = f"results/revision_bspc_2026/{RUN_NAME}"
    cfg["paths"]["big_disk_output_dir"] = f"/media/diego/Datos/vae_AD_results/revision_bspc_2026/{RUN_NAME}"
    cfg["paths"]["split_preview_csv"] = f"results/revision_bspc_2026/{RUN_NAME}_split_preview.csv"
    cfg["paths"]["split_preview_summary_csv"] = f"results/revision_bspc_2026/{RUN_NAME}_split_preview_summary.csv"
    cfg["slice_order_validonly_sensitivity"] = {
        "mask": "M1_pragmatic_cn_problem_site_unknown",
        "derived_metadata": str(DERIVED_METADATA.relative_to(PROJECT_ROOT)),
        "exclusion_manifest": str(M1_MANIFEST.relative_to(PROJECT_ROOT)),
        "exploratory_qc_sensitivity": True,
        "not_primary_replacement_without_external_validation": True,
    }
    return cfg


def launcher_source() -> str:
    return f'''#!/usr/bin/env python3
"""Guarded launcher for {RUN_NAME}.

Default behavior is dry-run/preflight only. Real training requires
--confirm-training and an existing DRY_RUN_PASS marker from the preflight
package. This launcher uses a derived metadata CSV and does not modify original
metadata/tensors/promoted artifacts.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path({str(PROJECT_ROOT)!r})
REFERENCE_CONFIG = PROJECT_ROOT / {str(REFERENCE_CONFIG.relative_to(PROJECT_ROOT))!r}
CONFIG = PROJECT_ROOT / {str(CANDIDATE_CONFIG.relative_to(PROJECT_ROOT))!r}
DRY_RUN_PASS = PROJECT_ROOT / {str(DRY_RUN_PASS.relative_to(PROJECT_ROOT))!r}
RUN_NAME = {RUN_NAME!r}
EXPECTED_M1_N = {len(EXPECTED_M1_SUBJECTS)}
STALE_NAMES = {{"run_config.json", "classifier_only_readout", "latent_cache"}}
STALE_PREFIXES = ("fold_", "all_folds_metrics", "summary_metrics")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def append_arg(cmd: list[str], name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            cmd.append(f"--{{name}}")
        return
    cmd.append(f"--{{name}}")
    if isinstance(value, list):
        cmd.extend(str(v) for v in value)
    else:
        cmd.append(str(value))


def validate_config(ref: dict[str, Any], cfg: dict[str, Any]) -> None:
    if cfg.get("run_name") != RUN_NAME:
        raise RuntimeError(f"Unexpected run_name: {{cfg.get('run_name')}}")
    ref_params = dict(ref["parameters"])
    cfg_params = dict(cfg["parameters"])
    diffs = {{k: (ref_params.get(k), cfg_params.get(k)) for k in sorted(set(ref_params) | set(cfg_params)) if ref_params.get(k) != cfg_params.get(k)}}
    if diffs:
        raise RuntimeError(f"Scientific parameter diffs are not allowed for M1 sensitivity: {{diffs}}")
    for key in ["global_tensor_path", "training_script"]:
        if ref["paths"].get(key) != cfg["paths"].get(key):
            raise RuntimeError(f"Unexpected path diff for {{key}}")
    if "sliceorderM1_validonly_derived_metadata.csv" not in str(cfg["paths"].get("metadata_path", "")):
        raise RuntimeError("metadata_path must point to the derived M1 metadata CSV.")
    if RUN_NAME not in str(cfg["paths"].get("output_dir", "")):
        raise RuntimeError("output_dir must contain the M1 run name.")
    if cfg.get("selected_channel_names") != ref.get("selected_channel_names"):
        raise RuntimeError("selected_channel_names changed unexpectedly.")
    if cfg["parameters"].get("channels_to_use") != [1, 0, 2]:
        raise RuntimeError("channels_to_use must remain [1,0,2].")


def stage_a_command(cfg: dict[str, Any], dry_run: bool) -> list[str]:
    python_exe = cfg.get("python_executable") or sys.executable
    cmd = [
        python_exe,
        str(resolve(cfg["paths"]["training_script"])),
        "--global_tensor_path", str(resolve(cfg["paths"]["global_tensor_path"])),
        "--metadata_path", str(resolve(cfg["paths"]["metadata_path"])),
        "--output_dir", str(resolve(cfg["paths"]["output_dir"])),
    ]
    for key, value in cfg["parameters"].items():
        append_arg(cmd, key, value)
    cmd.extend([
        "--vae_required_metadata_cols",
        "ResearchGroup_Mapped",
        "Manufacturer",
        "Age",
        "Sex",
        "sliceorder_m1_vae_eligible",
    ])
    cmd.append("--vae_abort_if_val_split_fails")
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def stale_markers(outdir: Path) -> list[str]:
    if not outdir.exists():
        return []
    return sorted(p.name for p in outdir.iterdir() if p.name in STALE_NAMES or any(p.name.startswith(x) for x in STALE_PREFIXES))


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Preflight only; no training.")
    parser.add_argument("--confirm-training", action="store_true", help="Required for real training.")
    args = parser.parse_args()
    if not args.dry_run and not args.confirm_training:
        raise SystemExit("Refusing to launch training without --confirm-training. Use --dry-run for preflight.")
    if args.confirm_training and not DRY_RUN_PASS.exists():
        raise SystemExit(f"Refusing real training: dry-run PASS marker is missing: {{DRY_RUN_PASS}}")
    ref = load_json(REFERENCE_CONFIG)
    cfg = load_json(CONFIG)
    validate_config(ref, cfg)
    outdir = resolve(cfg["paths"]["output_dir"])
    stale = stale_markers(outdir)
    if stale:
        raise SystemExit(f"Refusing to continue because stale output markers exist in {{outdir}}: {{stale}}")
    cmd = stage_a_command(cfg, dry_run=args.dry_run)
    print(shlex.join(cmd))
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
'''


def training_feasibility_text(df: pd.DataFrame, counts_fold: pd.DataFrame, feasibility: pd.DataFrame) -> str:
    lines = ["# Training Feasibility by Mask", ""]
    for mask in MASKS:
        ret = df[df[f"retained_{mask}"]]
        dx = ret["_diagnosis_norm"].value_counts().to_dict()
        mfr = ret["_manufacturer_norm"].value_counts().to_dict()
        ph = ret[ret["_manufacturer_norm"].eq("Philips")]
        ph_dx = ph["_diagnosis_norm"].value_counts().to_dict()
        all_status = feasibility[(feasibility["mask"].eq(mask)) & (feasibility["fold"].astype(str).eq("ALL"))]
        status = all_status["status"].iloc[0] if not all_status.empty else "NOT_EVALUATED"
        reason = all_status["reason"].iloc[0] if not all_status.empty else ""
        lines.extend([
            f"## {mask}",
            "",
            f"- Status: `{status}`",
            f"- Reason: `{reason}`",
            f"- Retained diagnosis counts: `{dx}`",
            f"- Retained manufacturer counts: `{mfr}`",
            f"- Retained Philips diagnosis counts: `{ph_dx}`",
            "",
        ])
    lines.extend([
        "## Final feasibility interpretation",
        "",
        "`M3_martin_literal_known_correct_only` is not a safe training mask at this snapshot.",
        "Philips slice-order annotations are incomplete and concentrated in the Philips CN audit subset;",
        "literal known-correct filtering would discard most Philips AD/MCI and distort the cohort.",
        "",
        "`M1_pragmatic_cn_problem_site_unknown` is fold-feasible as an exploratory QC sensitivity.",
        "It excludes exactly the currently identified Philips CN subjects with confirmed non-default",
        "slice order or missing slice order at the targeted problem sites 18/301. It must not replace",
        "the promoted primary model unless independently validated and reported as pre-specified.",
    ])
    return "\n".join(lines) + "\n"


def mask_definitions_text() -> str:
    return """# Slice-Order Curation Mask Definitions

All masks are exploratory QC-sensitivity masks. They do not modify the promoted
model, original metadata, tensors, predictions, thresholds, or fold files.

## M0_confirmed_nondefault_only

Exclude Philips subjects with `slice_order_class == reverse_even_odd_48`.
This is the narrow confirmed non-default DPARSF mismatch-risk mask.

## M1_pragmatic_cn_problem_site_unknown

Exclude:

- all M0 subjects;
- Philips CN subjects from Site18 with missing slice-order class;
- Philips CN subjects from Site301 with missing slice-order class.

This is the selected exploratory sensitivity mask because it is narrow,
CN-focused, and fold-feasible in the current annotation snapshot.

## M2_broad_problem_site_unknown_all_dx

Exclude all M0 subjects and all Philips subjects, any diagnosis, from Site18 or
Site301 when slice order is missing. This is reported for preflight comparison
only.

## M2_optional_broader_problem_sites_2_13_18_31_301

Optional broader descriptive mask: exclude all M0 subjects and all Philips
subjects from Sites 2, 13, 18, 31, or 301 when slice order is missing. Do not
train this unless explicitly requested.

## M3_martin_literal_known_correct_only

Keep all GE and Siemens subjects, but keep Philips subjects only when
`matches_dparsf_default == True`. This reflects Martín's literal request, but
is preflight-only here because Philips annotations are incomplete and mostly
available for Philips CN.
"""


def executive_summary(
    df: pd.DataFrame,
    score_df: pd.DataFrame,
    feasibility: pd.DataFrame,
    dry_returncode: int,
) -> str:
    phil = df[is_philips(df)]
    m1_ex = df.loc[~df["retained_M1_pragmatic_cn_problem_site_unknown"], "SubjectID"].astype(str).tolist()
    m3 = df[df["retained_M3_martin_literal_known_correct_only"]]
    m3_ph = m3[is_philips(m3)]
    m1_status = feasibility[(feasibility["mask"].eq("M1_pragmatic_cn_problem_site_unknown")) & (feasibility["fold"].astype(str).eq("ALL"))]["status"].iloc[0]
    return f"""# Executive Summary

This package is a preflight/dry-run only for a slice-order curated valid-only
sensitivity branch. No training was launched.

Promoted full database rows: `{len(df)}`.

Philips slice-order coverage:

- Total Philips subjects: `{len(phil)}`
- Philips with known slice_order_class: `{int((~phil['_slice_norm'].eq('MISSING')).sum())}`
- Philips default_odd_even_48: `{int(phil['_slice_norm'].eq('default_odd_even_48').sum())}`
- Philips reverse_even_odd_48: `{int(phil['_slice_norm'].eq('reverse_even_odd_48').sum())}`

Selected M1 exploratory sensitivity:

- Fold feasibility: `{m1_status}`
- Excluded subjects: `{len(m1_ex)}`
- Excluded SubjectIDs: `{', '.join(m1_ex)}`

M3 Martín-literal known-correct-only mask:

- Retained Philips subjects: `{len(m3_ph)}`
- Retained Philips diagnosis counts: `{m3_ph['_diagnosis_norm'].value_counts().to_dict()}`
- Feasibility: not recommended for training at this snapshot because Philips
  AD/MCI slice-order annotation coverage is incomplete.

Launcher dry-run return code: `{dry_returncode}`.
Dry-run PASS marker: `{'created' if dry_returncode == 0 and DRY_RUN_PASS.exists() else 'not_created'}`.

The M1 branch is an exploratory QC sensitivity only. It must not replace the
promoted model unless independently validated and reported transparently.
"""


def final_recommendation_text(df: pd.DataFrame, feasibility: pd.DataFrame) -> str:
    m1_ex = df.loc[~df["retained_M1_pragmatic_cn_problem_site_unknown"]].copy()
    m3 = df[df["retained_M3_martin_literal_known_correct_only"]]
    m3_ph = m3[is_philips(m3)]
    m1_status = feasibility[(feasibility["mask"].eq("M1_pragmatic_cn_problem_site_unknown")) & (feasibility["fold"].astype(str).eq("ALL"))]["status"].iloc[0]
    return f"""# Final Recommendation

## Is M3_martin_literal_known_correct_only feasible?

No. M3 is not recommended for training in the current annotation snapshot.
It keeps GE and Siemens all, but keeps only Philips scans with known DPARSF
default slice order. Philips slice-order annotation coverage is incomplete and
currently concentrated in the Philips CN audit subset, so Philips AD/MCI
representation is not reliable enough for a full retrain.

Retained Philips diagnosis counts under M3:

`{m3_ph['_diagnosis_norm'].value_counts().to_dict()}`

## Is M1 feasible as exploratory QC sensitivity?

Yes, M1 fold feasibility status is `{m1_status}`. It is feasible only as an
exploratory QC sensitivity, not as a replacement primary model.

## What exactly is excluded by M1?

M1 excludes `{len(m1_ex)}` Philips CN subjects:

`{', '.join(m1_ex['SubjectID'].astype(str).tolist())}`

These are the currently identified Philips CN subjects with confirmed
non-default reverse-even-odd slice order or missing slice-order annotations at
the targeted problem sites 18/301.

## Why this must not replace the primary model

The M1 exclusion rule was derived after observing Philips false-positive
structure. If retraining improves Philips CN FPR, part of that improvement may
be mechanical because the excluded set contains known high-risk Philips CN
subjects. The result must be reported as a pre-specified exploratory QC
sensitivity and requires independent validation before it can affect the
primary model claim.

## What to report to Martín and Diego

- The full promoted model remains locked.
- Literal known-correct Philips-only retraining is not feasible until Philips
  AD/MCI slice-order annotation coverage is completed.
- A narrow M1 sensitivity is dry-run feasible and excludes exactly the listed
  10 Philips CN subjects.
- The launcher is generated but guarded: real training requires explicit
  `--confirm-training` and the dry-run PASS marker.
"""


def create_outputs(df: pd.DataFrame, args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ref_cfg = read_json(REFERENCE_CONFIG)

    coverage = slice_coverage(df)
    exclusions = excluded_subjects(df)
    counts = retained_counts(df)
    counts_fold, feasibility = fold_counts(df)
    score_df, fpr_df = score_only(df)

    write_df("philips_slice_order_coverage", coverage, max_rows=200)
    (OUT / "mask_definitions.md").write_text(mask_definitions_text(), encoding="utf-8")
    write_df("excluded_subjects_by_mask", exclusions, max_rows=300)
    write_df("retained_counts_by_mask", counts, max_rows=400)
    write_df("retained_counts_by_fold", counts_fold, max_rows=300)
    write_df("locked_model_score_only_sensitivity", score_df, max_rows=80)
    write_df("philips_cn_fpr_by_mask", fpr_df, max_rows=120)
    (OUT / "training_feasibility_by_mask.md").write_text(training_feasibility_text(df, counts_fold, feasibility), encoding="utf-8")

    derived = create_derived_metadata(df, ref_cfg)
    derived.to_csv(DERIVED_METADATA, index=False)
    m1_manifest = exclusions[exclusions["mask"].eq("M1_pragmatic_cn_problem_site_unknown")].copy()
    m1_manifest.to_csv(M1_MANIFEST, index=False)

    actual_m1 = sorted(m1_manifest["SubjectID"].astype(str).tolist())
    expected_m1 = sorted(EXPECTED_M1_SUBJECTS)
    if actual_m1 != expected_m1:
        raise RuntimeError(f"M1 excluded subjects differ from expected. expected={expected_m1}, actual={actual_m1}")

    cand_cfg = create_config(ref_cfg)
    write_json(CANDIDATE_CONFIG, cand_cfg)
    GENERATED_LAUNCHER.write_text(launcher_source(), encoding="utf-8")
    GENERATED_LAUNCHER.chmod(0o755)

    py = cand_cfg.get("python_executable") or sys.executable
    compile_proc = subprocess.run(
        [py, "-m", "py_compile", str(GENERATED_LAUNCHER)],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if compile_proc.returncode != 0:
        raise RuntimeError(f"Generated launcher py_compile failed:\n{compile_proc.stderr}")

    dry_cmd = [py, str(GENERATED_LAUNCHER), "--dry-run"]
    dry_proc = subprocess.run(dry_cmd, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    (OUT / "launcher_dryrun_stdout.txt").write_text(dry_proc.stdout, encoding="utf-8")
    (OUT / "launcher_dryrun_stderr.txt").write_text(dry_proc.stderr, encoding="utf-8")
    if dry_proc.returncode == 0:
        DRY_RUN_PASS.write_text(
            f"PASS created_utc={datetime.now(timezone.utc).isoformat()}\ncommand={shlex.join(dry_cmd)}\n",
            encoding="utf-8",
        )

    launch_lines = [
        "# Slice-order M1 valid-only sensitivity commands",
        "",
        "# Dry-run validation:",
        shlex.join(dry_cmd),
        "",
        "# Real training is blocked unless explicitly confirmed and DRY_RUN_PASS.txt exists:",
        shlex.join([py, str(GENERATED_LAUNCHER), "--confirm-training"]),
        "",
        "# After Stage A completion, run the established Stage B classifier-only readout and OOF calibration,",
        "# adapting run-dir/output-dir to the M1 output path. Do not run these during preflight.",
        f"# Expected Stage A output_dir: {cand_cfg['paths']['output_dir']}",
        f"# Derived metadata: {DERIVED_METADATA.relative_to(PROJECT_ROOT)}",
        f"# Exclusion manifest: {M1_MANIFEST.relative_to(PROJECT_ROOT)}",
    ]
    (OUT / "launch_commands.txt").write_text("\n".join(launch_lines) + "\n", encoding="utf-8")

    (OUT / "00_EXECUTIVE_SUMMARY.md").write_text(executive_summary(df, score_df, feasibility, dry_proc.returncode), encoding="utf-8")
    (OUT / "final_recommendation.md").write_text(final_recommendation_text(df, feasibility), encoding="utf-8")

    write_json(
        OUT / "command_log.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "argv": sys.argv,
            "guardrails": {
                "preflight_only": True,
                "model_training": False,
                "original_metadata_edits": False,
                "tensor_edits": False,
                "prediction_edits": False,
                "threshold_refitting": False,
                "promoted_artifact_edits": False,
            },
            "inputs": {
                "full_database": str(FULL_DB),
                "reference_config": str(REFERENCE_CONFIG),
            },
            "outputs": str(OUT.relative_to(PROJECT_ROOT)),
            "generated_launcher": str(GENERATED_LAUNCHER.relative_to(PROJECT_ROOT)),
            "generated_config": str(CANDIDATE_CONFIG.relative_to(PROJECT_ROOT)),
            "derived_metadata": str(DERIVED_METADATA.relative_to(PROJECT_ROOT)),
            "dry_run_command": shlex.join(dry_cmd),
            "dry_run_returncode": int(dry_proc.returncode),
            "dry_run_pass_file": str(DRY_RUN_PASS.relative_to(PROJECT_ROOT)) if DRY_RUN_PASS.exists() else None,
            "m1_excluded_subjects": actual_m1,
        },
    )

    if dry_proc.returncode != 0:
        raise RuntimeError(f"Launcher dry-run failed with return code {dry_proc.returncode}. See launcher_dryrun_stderr.txt")


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Kept for explicit preflight semantics; this script never launches training.")
    args = parser.parse_args()
    if not FULL_DB.exists():
        raise FileNotFoundError(FULL_DB)
    if not REFERENCE_CONFIG.exists():
        raise FileNotFoundError(REFERENCE_CONFIG)
    df = pd.read_csv(FULL_DB)
    required = [
        "SubjectID", "diagnosis_group", "Manufacturer_final", "Site3_final",
        "slice_order_class", "matches_dparsf_default", "high_confidence_slice_timing_match",
        "y_score_final", "y_pred",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Full database is missing required columns: {missing}")
    df = add_masks(df)
    create_outputs(df, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
