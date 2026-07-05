#!/usr/bin/env python3
"""Read-only audit joining Martin's fMRI slice-timing file to Philips CN errors.

The audit integrates /media/diego/Datos/fmri_slice_timing.csv with the promoted
model master database and evaluates slice-order/acquisition metadata against
Philips CN FP/TN labels. It does not train models, modify tensors, edit
metadata, change predictions, refit thresholds, or exclude subjects.
"""

from __future__ import annotations

import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "revision_bspc_2026"
OUT = RESULTS / "philips_fmri_slice_timing_audit_20260612"
OUT.mkdir(parents=True, exist_ok=True)

SLICE_PATH = Path("/media/diego/Datos/fmri_slice_timing.csv")
MASTER_PATH = RESULTS / "promoted_model_master_database_20260610" / "promoted_model_master_database.csv"
FPR_PATH = RESULTS / "local_martin_philips_source_audit_20260611" / "site_protocol_qc_fpr_table.csv"
TEST_PATH = RESULTS / "local_martin_philips_source_audit_20260611" / "acquisition_qc_fp_vs_tn_tests.csv"

DEFAULT_ODD_EVEN_48 = list(range(1, 49, 2)) + list(range(2, 49, 2))
REVERSE_EVEN_ODD_48 = list(range(48, 0, -2)) + list(range(47, 0, -2))

COLORS = {
    "FP": "#D55E00",
    "TN": "#0072B2",
    "default_odd_even_48": "#0072B2",
    "reverse_even_odd_48": "#CC79A7",
    "other": "#999999",
}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_df(stem: str, df: pd.DataFrame) -> None:
    df.to_csv(OUT / f"{stem}.csv", index=False)
    if df.empty:
        (OUT / f"{stem}.md").write_text("_No rows._\n", encoding="utf-8")
    else:
        (OUT / f"{stem}.md").write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")


def write_md(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def savefig(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def rid_from_subject(sid: Any) -> float:
    m = re.match(r"^\d+_S_(\d+)$", str(sid).strip())
    return float(int(m.group(1))) if m else np.nan


def parse_date(x: Any) -> pd.Timestamp:
    if pd.isna(x):
        return pd.NaT
    return pd.to_datetime(str(x), errors="coerce")


def parse_sliceord(x: Any) -> list[int]:
    if pd.isna(x):
        return []
    return [int(v) for v in re.findall(r"-?\d+", str(x))]


def classify_slice_order(vals: list[int]) -> str:
    if vals == DEFAULT_ODD_EVEN_48:
        return "default_odd_even_48"
    if vals == REVERSE_EVEN_ODD_48:
        return "reverse_even_odd_48"
    if not vals:
        return "missing"
    return f"other_{len(vals)}"


def normalize_string(x: Any) -> str:
    if pd.isna(x):
        return "MISSING"
    s = str(x).strip()
    return s if s else "MISSING"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for p in [SLICE_PATH, MASTER_PATH, FPR_PATH, TEST_PATH]:
        if not p.exists():
            raise FileNotFoundError(p)
    st = pd.read_csv(SLICE_PATH, low_memory=False)
    master = pd.read_csv(MASTER_PATH, low_memory=False)
    fpr = pd.read_csv(FPR_PATH)
    tests = pd.read_csv(TEST_PATH)
    return st, master, fpr, tests


def add_slice_features(st: pd.DataFrame) -> pd.DataFrame:
    out = st.copy()
    out["RID"] = pd.to_numeric(out["RID"], errors="coerce")
    out["IMAGEUID"] = pd.to_numeric(out["IMAGEUID"], errors="coerce")
    out["SCANDATE_parsed"] = out["SCANDATE"].apply(parse_date)
    out["slice_order_list"] = out["SLICEORD"].apply(parse_sliceord)
    out["slice_order_len"] = out["slice_order_list"].apply(len)
    out["slice_order_class"] = out["slice_order_list"].apply(classify_slice_order)
    out["matches_dparsf_default"] = out["slice_order_class"].eq("default_odd_even_48")
    out["PHASEDIR_norm"] = out["PHASEDIR"].apply(normalize_string)
    out["model_norm"] = out.get("MANUFACTURERSMODELNAME", out.get("ScannerModel", pd.Series(index=out.index))).apply(normalize_string)
    out["scanner_model_norm"] = out.get("ScannerModel", pd.Series(index=out.index)).apply(normalize_string)
    out["software_version_norm"] = out.get("SoftwareVersion", pd.Series(index=out.index)).apply(normalize_string)
    out["manufacturer_norm"] = out.get("MANUFACTURER", out.get("ScannerManufacturer", pd.Series(index=out.index))).apply(normalize_string)
    return out


def source_inventory(st_raw: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {"section": "shape", "item": "rows_raw_file", "value": len(st_raw)},
        {"section": "shape", "item": "columns_raw_file", "value": st_raw.shape[1]},
        {"section": "shape", "item": "columns_after_derived_audit_fields", "value": st.shape[1]},
        {"section": "identity", "item": "unique_RID", "value": int(st["RID"].nunique(dropna=True))},
        {"section": "identity", "item": "unique_IMAGEUID", "value": int(st["IMAGEUID"].nunique(dropna=True))},
        {"section": "coverage", "item": "SLICETIMING_NFQ_nonmissing", "value": int(st["SLICETIMING_NFQ"].notna().sum()) if "SLICETIMING_NFQ" in st else 0},
        {"section": "coverage", "item": "SLICETIMING_NFQ_fraction", "value": float(st["SLICETIMING_NFQ"].notna().mean()) if "SLICETIMING_NFQ" in st else 0.0},
    ]
    for col, section in [
        ("MANUFACTURER", "MANUFACTURER_counts"),
        ("MANUFACTURERSMODELNAME", "MANUFACTURERSMODELNAME_counts"),
        ("slice_order_class", "SLICEORD_class_counts"),
        ("PHASEDIR_norm", "PHASEDIR_counts"),
        ("NumberVolumes", "NumberVolumes_counts"),
        ("SlicesPerVolume", "SlicesPerVolume_counts"),
    ]:
        if col not in st.columns:
            continue
        vc = st[col].fillna("MISSING").astype(str).value_counts(dropna=False)
        for k, v in vc.items():
            rows.append({"section": section, "item": k, "value": int(v)})
    return pd.DataFrame(rows)


def philips_cn_master(master: pd.DataFrame) -> pd.DataFrame:
    if "is_philips_cn" in master.columns:
        ph = master[master["is_philips_cn"].fillna(False).astype(bool)].copy()
    else:
        ph = master[
            master["Manufacturer"].astype(str).str.upper().eq("PHILIPS")
            & master["ResearchGroup_Mapped"].astype(str).str.upper().eq("CN")
        ].copy()
    ph = ph.drop_duplicates("SubjectID").reset_index(drop=True)
    ph["RID_derived"] = ph["SubjectID"].apply(rid_from_subject)
    ph["ImageID_num"] = pd.to_numeric(ph["ImageID"], errors="coerce")
    ph["master_date"] = ph["acquisition_date"].apply(parse_date)
    if "EXAMDATE" in ph.columns:
        exam_date = ph["EXAMDATE"].apply(parse_date)
        ph["master_date"] = ph["master_date"].where(ph["master_date"].notna(), exam_date)
    return ph


def pick_best_status(rows: pd.DataFrame) -> pd.Series:
    if rows.empty:
        raise ValueError("Cannot pick from empty rows")
    tmp = rows.copy()
    tmp["_status_rank"] = tmp.get("STATUS", pd.Series(index=tmp.index)).astype(str).str.lower().map({"final": 0}).fillna(1)
    tmp = tmp.sort_values(["_status_rank", "SCANDATE_parsed", "IMAGEUID"], na_position="last")
    return tmp.iloc[0]


def join_subjects(ph: pd.DataFrame, st: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    cov_rows: list[dict[str, Any]] = []
    for _, subj in ph.iterrows():
        sid = subj["SubjectID"]
        image = subj["ImageID_num"]
        rid = subj["RID_derived"]
        date = subj["master_date"]
        match = None
        method = "none"
        confidence = "none"
        candidates_image = st[st["IMAGEUID"].eq(image)] if np.isfinite(image) else st.iloc[0:0]
        candidates_date = st[(st["RID"].eq(rid)) & (st["SCANDATE_parsed"].eq(date))] if np.isfinite(rid) and pd.notna(date) else st.iloc[0:0]
        candidates_rid = st[st["RID"].eq(rid)] if np.isfinite(rid) else st.iloc[0:0]
        if not candidates_image.empty:
            match = pick_best_status(candidates_image)
            method = "IMAGEUID_exact"
            confidence = "high"
        elif not candidates_date.empty:
            match = pick_best_status(candidates_date)
            method = "RID_date_exact"
            confidence = "high"
        elif not candidates_rid.empty:
            tmp = candidates_rid.copy()
            if pd.notna(date):
                tmp["_date_abs_days"] = (tmp["SCANDATE_parsed"] - date).abs().dt.days
                tmp = tmp.sort_values(["_date_abs_days", "SCANDATE_parsed", "IMAGEUID"], na_position="last")
            match = tmp.iloc[0]
            method = "RID_only_nearest"
            confidence = "low"
        base = subj.to_dict()
        base.update({"match_method": method, "match_confidence": confidence})
        if match is not None:
            for col in st.columns:
                if col == "slice_order_list":
                    base[f"fmri_{col}"] = " ".join(str(x) for x in match[col])
                else:
                    base[f"fmri_{col}"] = match[col]
            if confidence == "low" and pd.notna(date) and pd.notna(match.get("SCANDATE_parsed", pd.NaT)):
                base["rid_nearest_date_abs_days"] = abs((match["SCANDATE_parsed"] - date).days)
            else:
                base["rid_nearest_date_abs_days"] = np.nan
        else:
            base["rid_nearest_date_abs_days"] = np.nan
        rows.append(base)
        cov_rows.append(
            {
                "SubjectID": sid,
                "RID": rid,
                "ImageID": image,
                "master_date": date,
                "n_IMAGEUID_exact_candidates": int(len(candidates_image)),
                "n_RID_date_exact_candidates": int(len(candidates_date)),
                "n_RID_only_candidates": int(len(candidates_rid)),
                "match_method": method,
                "match_confidence": confidence,
            }
        )
    merged = pd.DataFrame(rows)
    for col in ["slice_order_class", "slice_order_len", "matches_dparsf_default", "PHASEDIR_norm", "model_norm", "scanner_model_norm", "software_version_norm"]:
        fcol = f"fmri_{col}"
        if fcol not in merged.columns:
            merged[fcol] = np.nan
    merged["stc_mismatch_risk"] = merged["match_confidence"].eq("high") & ~merged["fmri_matches_dparsf_default"].fillna(False).astype(bool)
    return merged, pd.DataFrame(cov_rows)


def fpr_table(df: pd.DataFrame, group_col: str, label: str, min_n: int = 1) -> pd.DataFrame:
    rows = []
    for group, sub in df.groupby(group_col, dropna=False):
        n = len(sub)
        if n < min_n:
            continue
        fp = int(sub["confusion_label"].astype(str).str.upper().eq("FP").sum())
        rows.append(
            {
                "stratification": label,
                "group": "MISSING" if pd.isna(group) else str(group),
                "N": int(n),
                "N_FP": fp,
                "FPR": fp / n if n else np.nan,
                "low_n_caution": bool(n < 10),
            }
        )
    return pd.DataFrame(rows).sort_values(["stratification", "FPR", "N"], ascending=[True, False, False])


def fisher_or_chi2(df: pd.DataFrame, var: str) -> dict[str, Any]:
    sub = df[["confusion_label", var]].dropna().copy()
    sub = sub[sub["confusion_label"].isin(["FP", "TN"])]
    if sub.empty or sub[var].nunique() < 2:
        return {"variable": var, "type": "categorical", "test": "not_tested", "p_raw": np.nan, "effect": "insufficient_levels"}
    ct = pd.crosstab(sub["confusion_label"], sub[var])
    if ct.shape == (2, 2):
        odds, p = stats.fisher_exact(ct.values)
        return {"variable": var, "type": "categorical", "test": "Fisher exact", "p_raw": float(p), "effect": f"OR={odds:.4g}", "levels": list(ct.columns)}
    chi2, p, _dof, _exp = stats.chi2_contingency(ct.values)
    return {"variable": var, "type": "categorical", "test": "Chi-square descriptive low-N aware", "p_raw": float(p), "effect": f"chi2={chi2:.4g}", "levels": list(ct.columns)}


def mw_test(df: pd.DataFrame, var: str) -> dict[str, Any]:
    fp = pd.to_numeric(df.loc[df["confusion_label"].eq("FP"), var], errors="coerce").dropna().to_numpy()
    tn = pd.to_numeric(df.loc[df["confusion_label"].eq("TN"), var], errors="coerce").dropna().to_numpy()
    row = {
        "variable": var,
        "type": "continuous",
        "test": "Mann-Whitney",
        "FP_n": len(fp),
        "TN_n": len(tn),
        "FP_median": float(np.median(fp)) if len(fp) else np.nan,
        "TN_median": float(np.median(tn)) if len(tn) else np.nan,
        "p_raw": np.nan,
        "CLES_FP_gt_TN": np.nan,
    }
    if len(fp) >= 3 and len(tn) >= 3:
        _u, p = stats.mannwhitneyu(fp, tn, alternative="two-sided")
        row["p_raw"] = float(p)
        row["CLES_FP_gt_TN"] = float(np.mean(fp[:, None] > tn[None, :]))
    return row


def fp_tn_tests(high: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for var in ["fmri_MEANTSNR", "fmri_MEDTSNR", "fmri_SDTSNR"]:
        rows.append(mw_test(high, var))
    for var in [
        "fmri_slice_order_class",
        "fmri_matches_dparsf_default",
        "fmri_PHASEDIR_norm",
        "fmri_model_norm",
        "fmri_scanner_model_norm",
        "fmri_software_version_norm",
        "Site3",
        "raw_tp_group",
    ]:
        rows.append(fisher_or_chi2(high, var))
    return pd.DataFrame(rows)


def subject_table(merged: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "SubjectID",
        "ImageID",
        "RID_derived",
        "Site3",
        "raw_tp_group",
        "Age",
        "y_score_final",
        "y_pred",
        "confusion_label",
        "fmri_SLICEORD",
        "fmri_slice_order_class",
        "fmri_slice_order_len",
        "fmri_matches_dparsf_default",
        "stc_mismatch_risk",
        "fmri_PHASEDIR_norm",
        "fmri_MEANTSNR",
        "fmri_MEDTSNR",
        "fmri_SDTSNR",
        "fmri_MANUFACTURERSMODELNAME",
        "fmri_ScannerModel",
        "fmri_SoftwareVersion",
        "fmri_RepetitionTime",
        "fmri_REPETITIONTIME",
        "fmri_EchoTime",
        "fmri_ECHOTIME",
        "fmri_NumberVolumes",
        "fmri_SlicesPerVolume",
        "fmri_SliceThickness",
        "match_method",
        "match_confidence",
        "rid_nearest_date_abs_days",
    ]
    out = merged[[c for c in cols if c in merged.columns]].copy()
    out["_fp_sort"] = out["confusion_label"].astype(str).str.upper().eq("FP")
    out = out.sort_values(["stc_mismatch_risk", "_fp_sort", "y_score_final"], ascending=[False, False, False]).drop(columns="_fp_sort")
    return out


def question_answers(merged: pd.DataFrame, high: pd.DataFrame, fpr: pd.DataFrame) -> str:
    mismatch = high[high["stc_mismatch_risk"]].copy()
    site31 = high[pd.to_numeric(high["Site3"], errors="coerce").eq(31)]
    site18 = high[pd.to_numeric(high["Site3"], errors="coerce").eq(18)]
    site2 = high[pd.to_numeric(high["Site3"], errors="coerce").eq(2)]
    phasedir = fpr[fpr["stratification"].eq("PHASEDIR")]
    model = fpr[fpr["stratification"].eq("MANUFACTURERSMODELNAME")]

    def list_subjects(df: pd.DataFrame) -> str:
        if df.empty:
            return "- None"
        lines = []
        for _, r in df.sort_values(["Site3", "SubjectID"]).iterrows():
            lines.append(
                f"- `{r['SubjectID']}`: {r['confusion_label']}, Site3={r.get('Site3')}, "
                f"slice_order={r.get('fmri_slice_order_class')}, PHASEDIR={r.get('fmri_PHASEDIR_norm')}, ImageID={r.get('ImageID')}"
            )
        return "\n".join(lines)

    lines = [
        "# Martin Questions Answered",
        "",
        "## A/B/C. Philips CN classifier/evaluation subjects with non-default slice order",
        "",
        list_subjects(mismatch),
        "",
        f"High-confidence matched Philips CN subjects: `{len(high)}` of 99.",
        f"High-confidence STC mismatch-risk subjects: `{len(mismatch)}`.",
        "",
        "## D. Are Site 31 and Site 18 affected?",
        "",
        f"- Site 31 high-confidence N={len(site31)}; non-default slice-order N={int(site31['stc_mismatch_risk'].sum()) if not site31.empty else 0}.",
        f"- Site 18 high-confidence N={len(site18)}; non-default slice-order N={int(site18['stc_mismatch_risk'].sum()) if not site18.empty else 0}.",
        "",
        "## E. Does slice-order mismatch explain Site 2 7/7 FP?",
        "",
        f"- Site 2 high-confidence N={len(site2)}, FP={int(site2['confusion_label'].eq('FP').sum()) if not site2.empty else 0}.",
        f"- Site 2 non-default slice-order N={int(site2['stc_mismatch_risk'].sum()) if not site2.empty else 0}.",
        "- If Site 2 is default but 7/7 FP, slice order alone cannot explain the full Philips shift.",
        "",
        "## F. Does PHASEDIR AP/PA explain FP/TN?",
        "",
        phasedir.to_markdown(index=False) if not phasedir.empty else "No PHASEDIR strata available.",
        "",
        "PHASEDIR is descriptive here; low-N and site/protocol coupling limit causal interpretation.",
        "",
        "## G. Does scanner model explain FP/TN, or is it confounded with site/rawTP?",
        "",
        model.to_markdown(index=False) if not model.empty else "No scanner-model strata available.",
        "",
        "Scanner model is interpreted as confounded with site, ADNI phase, and raw timepoint group unless a sufficiently balanced within-site analysis is available.",
    ]
    return "\n".join(lines)


def make_figures(high: pd.DataFrame, fpr: pd.DataFrame) -> None:
    # Fig 1
    fig1_df = fpr[fpr["stratification"].eq("slice_order_class")].copy()
    fig1_df.to_csv(OUT / "fig1_slice_order_fpr_high_confidence_plotted_values.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    if not fig1_df.empty:
        fig1_df = fig1_df.sort_values("FPR")
        colors = [COLORS.get(g, COLORS["other"]) for g in fig1_df["group"]]
        ax.barh(fig1_df["group"], fig1_df["FPR"], color=colors)
        for i, r in enumerate(fig1_df.itertuples(index=False)):
            ax.text(r.FPR + 0.015, i, f"{r.FPR:.2f} ({r.N_FP}/{r.N})", va="center")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("False-positive rate")
    ax.set_title("Philips CN FPR by slice-order class (high-confidence matches)")
    ax.grid(axis="x", color="#DDDDDD")
    savefig(fig, "fig1_slice_order_fpr_high_confidence")

    # Fig 2
    fig2_df = pd.concat(
        [
            fpr[fpr["stratification"].eq("PHASEDIR")].assign(panel="PHASEDIR"),
            fpr[fpr["stratification"].eq("MANUFACTURERSMODELNAME")].assign(panel="Scanner model"),
        ],
        ignore_index=True,
    )
    fig2_df.to_csv(OUT / "fig2_phasedir_model_fpr_high_confidence_plotted_values.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, panel in zip(axes, ["PHASEDIR", "Scanner model"]):
        sub = fig2_df[fig2_df["panel"].eq(panel)].sort_values("FPR")
        ax.barh(sub["group"], sub["FPR"], color="#0072B2")
        for i, r in enumerate(sub.itertuples(index=False)):
            ax.text(r.FPR + 0.015, i, f"{r.FPR:.2f} ({r.N_FP}/{r.N})", va="center", fontsize=9)
        ax.set_xlim(0, 1.05)
        ax.set_title(panel)
        ax.grid(axis="x", color="#DDDDDD")
    fig.suptitle("PHASEDIR and scanner-model FPR (high-confidence Philips CN)")
    savefig(fig, "fig2_phasedir_model_fpr_high_confidence")

    # Fig 3
    matrix = pd.crosstab(high["Site3"].astype(str), high["fmri_slice_order_class"].astype(str))
    matrix.to_csv(OUT / "fig3_site_slice_order_matrix_plotted_values.csv")
    fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(matrix))), constrained_layout=True)
    im = ax.imshow(matrix.values, aspect="auto", cmap="Blues")
    ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns, rotation=30, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]), matrix.index)
    ax.set_xlabel("Slice-order class")
    ax.set_ylabel("Site3")
    ax.set_title("Site3 x slice-order class counts")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix.values[i, j]), ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8, label="N")
    savefig(fig, "fig3_site_slice_order_matrix")

    # Fig 4
    plot = high[high["confusion_label"].isin(["FP", "TN"])].copy()
    plot.to_csv(OUT / "fig4_score_by_slice_order_plotted_values.csv", index=False)
    classes = sorted(plot["fmri_slice_order_class"].dropna().unique())
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    rng = np.random.default_rng(20260612)
    xpos = {c: i for i, c in enumerate(classes)}
    for label, color, offset in [("TN", "#0072B2", -0.12), ("FP", "#D55E00", 0.12)]:
        sub = plot[plot["confusion_label"].eq(label)]
        x = sub["fmri_slice_order_class"].map(xpos).astype(float).to_numpy() + offset + rng.normal(0, 0.035, len(sub))
        ax.scatter(x, sub["y_score_final"], color=color, label=label, alpha=0.82, edgecolor="white", linewidth=0.4)
    ax.set_xticks(range(len(classes)), classes, rotation=25, ha="right")
    ax.set_ylabel("Promoted-model score")
    ax.set_title("Philips CN score by slice-order class")
    ax.grid(axis="y", color="#DDDDDD")
    ax.legend(title="OOF label")
    savefig(fig, "fig4_score_by_slice_order")


def caption_files() -> None:
    write_md(
        OUT / "fig1_slice_order_fpr_high_confidence_caption.md",
        "FPR by slice-order class among high-confidence Philips CN matches. Descriptive only; low-N strata should not be interpreted causally.",
    )
    write_md(
        OUT / "fig2_phasedir_model_fpr_high_confidence_caption.md",
        "FPR by PHASEDIR and scanner model among high-confidence Philips CN matches. These fields may be confounded with site, phase, and raw timepoint group.",
    )
    write_md(
        OUT / "fig3_site_slice_order_matrix_caption.md",
        "Count heatmap for Site3 by slice-order class among high-confidence Philips CN matches. Used to locate objective protocol strata for follow-up review.",
    )
    write_md(
        OUT / "fig4_score_by_slice_order_caption.md",
        "Promoted-model score distribution by slice-order class and FP/TN label among high-confidence Philips CN matches. Descriptive; thresholds were not refit.",
    )


def executive_summary(inv: pd.DataFrame, coverage: pd.DataFrame, merged: pd.DataFrame, fpr: pd.DataFrame, tests: pd.DataFrame) -> str:
    high = merged[merged["match_confidence"].eq("high")]
    mismatch = high[high["stc_mismatch_risk"]]
    fp = int(high["confusion_label"].eq("FP").sum())
    tn = int(high["confusion_label"].eq("TN").sum())
    top_fpr = fpr[fpr["stratification"].isin(["slice_order_class", "raw_tp_group", "Site3"])].head(20)
    return f"""# 00 Executive Summary - Philips fMRI Slice Timing Audit

**Generated**: {datetime.now(timezone.utc).isoformat()}

## Scope

Read-only integration of Martín's `/media/diego/Datos/fmri_slice_timing.csv`
with the promoted model master database for 99 Philips CN subjects.

## Join Result

- Philips CN subjects: 99
- High-confidence matches: {len(high)}
- Low-confidence RID-only matches: {int(merged['match_confidence'].eq('low').sum())}
- Unmatched: {int(merged['match_confidence'].eq('none').sum())}
- High-confidence FP/TN: FP={fp}, TN={tn}
- High-confidence non-default DPARSF slice-order mismatch-risk subjects: {len(mismatch)}

## Interpretation Preview

Slice-order evidence is objective for high-confidence matches, but remains
descriptive. If non-default high-confidence matches are concentrated in a site,
that is an STC mismatch risk, not proof of causality. If high-FPR sites use the
default odd/even 48-slice order, slice order alone cannot explain the Philips
CN shift.

## Top FPR Strata

{top_fpr.to_markdown(index=False) if not top_fpr.empty else '_No rows._'}
"""


def final_interpretation(merged: pd.DataFrame, fpr: pd.DataFrame) -> str:
    high = merged[merged["match_confidence"].eq("high")]
    mismatch = high[high["stc_mismatch_risk"]]
    site2 = high[pd.to_numeric(high["Site3"], errors="coerce").eq(2)]
    site31 = high[pd.to_numeric(high["Site3"], errors="coerce").eq(31)]
    site18 = high[pd.to_numeric(high["Site3"], errors="coerce").eq(18)]
    site2_default = int(site2["fmri_matches_dparsf_default"].fillna(False).sum()) if not site2.empty else 0
    return f"""# Final Interpretation - Philips fMRI Slice Timing Audit

This audit is read-only and descriptive. It does not retrain models, edit
tensors or metadata, refit thresholds, modify predictions, or exclude subjects.

## Slice-Order Risk

High-confidence non-default slice-order matches: `{len(mismatch)}`.

If Site 31 contains `reverse_even_odd_48` while the DPARSF preprocessing default
was `default_odd_even_48`, this is an objective slice-timing-correction mismatch
risk. It should be described as a risk and a plausible contributor, not as a
proven cause.

Site 31: N={len(site31)}, non-default N={int(site31['stc_mismatch_risk'].sum()) if not site31.empty else 0}.
Site 18: N={len(site18)}, non-default N={int(site18['stc_mismatch_risk'].sum()) if not site18.empty else 0}.

Site 2: N={len(site2)}, default-slice-order N={site2_default}, FP={int(site2['confusion_label'].eq('FP').sum()) if not site2.empty else 0}.
If Site 2 is default slice order but 7/7 FP, slice order alone cannot explain
the full Philips shift.

## Conservative Scientific Interpretation

The dominant explanation remains multi-factorial and descriptive: age, ADNI
phase/raw timepoint group, site, scanner model/protocol metadata, possible
slice-timing-correction mismatch in specific strata, and the previously observed
channel-level MI/Pearson shift. None of these findings by itself justifies
post-hoc subject exclusion or model selection.
"""


def main() -> None:
    st_raw, master, fpr_prior, tests_prior = load_inputs()
    st = add_slice_features(st_raw)
    inv = source_inventory(st_raw, st)
    write_df("fmri_slice_timing_source_inventory", inv)

    ph = philips_cn_master(master)
    merged, coverage = join_subjects(ph, st)
    coverage_summary = pd.concat(
        [
            pd.DataFrame(
                [
                    {"match_confidence": k, "N": int(v)}
                    for k, v in merged["match_confidence"].value_counts(dropna=False).items()
                ]
            ),
            pd.DataFrame(
                [
                    {"match_confidence": f"method:{k}", "N": int(v)}
                    for k, v in merged["match_method"].value_counts(dropna=False).items()
                ]
            ),
        ],
        ignore_index=True,
    )
    coverage_detail = pd.concat([coverage_summary, coverage], ignore_index=True, sort=False)
    write_df("philips_cn_join_coverage", coverage_detail)
    merged.to_csv(OUT / "philips_cn_fmri_slice_timing_merged.csv", index=False)

    subj = subject_table(merged)
    write_df("philips_cn_slice_order_subject_table", subj)

    high = merged[merged["match_confidence"].eq("high")].copy()
    high = high[high["confusion_label"].isin(["FP", "TN"])].copy()
    fpr_tables = pd.concat(
        [
            fpr_table(high, "fmri_slice_order_class", "slice_order_class"),
            fpr_table(high, "fmri_matches_dparsf_default", "matches_dparsf_default"),
            fpr_table(high, "fmri_PHASEDIR_norm", "PHASEDIR"),
            fpr_table(high, "fmri_model_norm", "MANUFACTURERSMODELNAME"),
            fpr_table(high, "fmri_scanner_model_norm", "ScannerModel"),
            fpr_table(high, "fmri_software_version_norm", "SoftwareVersion"),
            fpr_table(high, "Site3", "Site3"),
            fpr_table(high, "raw_tp_group", "raw_tp_group"),
        ],
        ignore_index=True,
    )
    write_df("fpr_by_slice_order_phase_model_site", fpr_tables)

    tests = fp_tn_tests(high)
    write_df("fp_tn_tests_high_confidence", tests)

    make_figures(high, fpr_tables)
    caption_files()

    write_md(OUT / "martin_questions_answered.md", question_answers(merged, high, fpr_tables))
    write_md(OUT / "final_interpretation.md", final_interpretation(merged, fpr_tables))
    write_md(OUT / "00_EXECUTIVE_SUMMARY.md", executive_summary(inv, coverage_detail, merged, fpr_tables, tests))

    command_log = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
        "inputs": {
            "fmri_slice_timing": str(SLICE_PATH),
            "master_db": str(MASTER_PATH),
            "prior_fpr": str(FPR_PATH),
            "prior_tests": str(TEST_PATH),
        },
        "output_dir": str(OUT.relative_to(PROJECT_ROOT)),
        "guardrails": {
            "read_only": True,
            "model_training": False,
            "tensor_edits": False,
            "metadata_edits": False,
            "prediction_edits": False,
            "threshold_refitting": False,
            "subject_exclusion": False,
        },
        "match_counts": merged["match_confidence"].value_counts(dropna=False).to_dict(),
        "argv": sys.argv,
    }
    write_json(OUT / "command_log.json", command_log)


if __name__ == "__main__":
    main()
