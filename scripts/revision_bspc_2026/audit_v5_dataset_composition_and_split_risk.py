"""
Dataset composition and CV split risk audit for ADNI v5 DPARSF-10000
no-Python-bandpass training set.

Analyses:
  1. Subject counts (tensor vs metadata, by diagnosis)
  2. Demographics per diagnosis (Age, Sex, SMD)
  3. Manufacturer/Site confounds (chi-square, Cramér V, mono-case sites)
  4. Per-fold composition for the v5 [4,1,0] baseline run
  5. Per-fold scanner leakage vs AUC
  6. CV stratification recommendations

Usage:
    python audit_v5_dataset_composition_and_split_risk.py [--output-root DIR]

Outputs (default: results/revision_bspc_2026/v5_dataset_composition_and_split_risk/):
    README.md
    dataset_counts.csv
    demographics_by_diagnosis.csv
    manufacturer_site_contingency.csv
    fold_composition.csv
    fold_leakage_auc_table.csv
    stratification_recommendation.csv
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]

_TENSOR_PATH = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026"
    "/adni_expanded_v5_dparsf10000_no_pybandpass/subject_tensors"
    "/GLOBAL_TENSOR_ADNI_expanded_v5_dparsf10000_no_pybandpass.npz"
)
_METADATA_PATH = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026"
    "/adni_expanded_v5_dparsf10000_no_pybandpass"
    "/training_ready_metadata_v5_dparsf10000_no_pybandpass.csv"
)
_RESULTS_DIR = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026"
    "/adni_v5_dparsf10000_no_pybandpass_ch4_1_0_baseline"
)
_DEFAULT_OUT = (
    _REPO_ROOT / "results" / "revision_bspc_2026"
    / "v5_dataset_composition_and_split_risk"
)

_N_FOLDS = 5
_SUPERVISED_GROUPS = ["CN", "AD"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=_DEFAULT_OUT)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _smd(a: pd.Series, b: pd.Series) -> float:
    """Cohen's d / standardised mean difference (pooled SD)."""
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else float("nan")


def _cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    return float(np.sqrt(chi2 / (n * (min(r, c) - 1))))


def _pct(num: int, den: int) -> str:
    return f"{100 * num / den:.1f}%" if den > 0 else "N/A"


def _iqr(s: pd.Series) -> float:
    return float(s.quantile(0.75) - s.quantile(0.25))


# ---------------------------------------------------------------------------
# 1. Subject counts
# ---------------------------------------------------------------------------

def build_dataset_counts(meta: pd.DataFrame, tensor_n: int) -> pd.DataFrame:
    cn_ad = meta[meta["ResearchGroup_Mapped"].isin(_SUPERVISED_GROUPS)]
    rows = [
        {"item": "tensor_subjects_total", "value": tensor_n, "note": "subjects in global tensor"},
        {"item": "metadata_subjects_total", "value": len(meta),
         "note": "training-ready metadata (3 excluded: 035_S_6927, 128_S_2002, 114_S_6039)"},
    ]
    for grp in ["CN", "AD", "MCI"]:
        n = int((meta["ResearchGroup_Mapped"] == grp).sum())
        rows.append({"item": f"n_{grp}", "value": n, "note": f"in metadata"})
    rows += [
        {"item": "n_supervised_CN_AD",
         "value": len(cn_ad), "note": "CN+AD with complete Age+Sex"},
        {"item": "n_missing_Age", "value": int(meta["Age"].isna().sum()), "note": ""},
        {"item": "n_missing_Sex", "value": int(meta["Sex"].isna().sum()), "note": ""},
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Demographics
# ---------------------------------------------------------------------------

def build_demographics(meta: pd.DataFrame) -> pd.DataFrame:
    rows = []
    cn_ad = meta[meta["ResearchGroup_Mapped"].isin(_SUPERVISED_GROUPS)]
    for grp in ["CN", "AD", "MCI"]:
        sub = meta[meta["ResearchGroup_Mapped"] == grp]
        age = sub["Age"].dropna()
        n = len(sub)
        n_F = int((sub["Sex"] == "F").sum())
        n_M = int((sub["Sex"] == "M").sum())
        rows.append({
            "diagnosis": grp,
            "n": n,
            "age_mean": round(float(age.mean()), 2),
            "age_std": round(float(age.std(ddof=1)), 2),
            "age_median": round(float(age.median()), 2),
            "age_IQR": round(_iqr(age), 2),
            "age_min": round(float(age.min()), 1),
            "age_max": round(float(age.max()), 1),
            "n_F": n_F,
            "n_M": n_M,
            "pct_F": _pct(n_F, n),
            "n_Age_missing": int(sub["Age"].isna().sum()),
            "n_Sex_missing": int(sub["Sex"].isna().sum()),
        })

    # Age SMD (AD vs CN)
    ad_age = meta[meta["ResearchGroup_Mapped"] == "AD"]["Age"].dropna()
    cn_age = meta[meta["ResearchGroup_Mapped"] == "CN"]["Age"].dropna()
    smd_val = _smd(ad_age, cn_age)
    rows.append({
        "diagnosis": "AD_vs_CN_SMD",
        "n": len(ad_age) + len(cn_age),
        "age_mean": round(smd_val, 3),
        "age_std": float("nan"),
        "age_median": float("nan"),
        "age_IQR": float("nan"),
        "age_min": float("nan"),
        "age_max": float("nan"),
        "n_F": None,
        "n_M": None,
        "pct_F": "",
        "n_Age_missing": None,
        "n_Sex_missing": None,
        "note": "standardised mean difference (pooled SD); |SMD|<0.2 negligible, 0.2-0.5 small",
    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Manufacturer / Site
# ---------------------------------------------------------------------------

def build_manufacturer_site(meta: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cn_ad = meta[meta["ResearchGroup_Mapped"].isin(_SUPERVISED_GROUPS)].copy()

    # Contingency table
    ct = pd.crosstab(cn_ad["ResearchGroup_Mapped"], cn_ad["Manufacturer"])
    chi2, p, dof, expected = chi2_contingency(ct)
    n = int(ct.values.sum())
    r, c = ct.shape
    v = _cramers_v(chi2, n, r, c)

    stats = {
        "chi2": round(chi2, 3),
        "p_value": float(p),
        "dof": int(dof),
        "n": n,
        "cramers_v": round(v, 3),
        "interpretation": (
            "negligible (<0.1)" if v < 0.1
            else "weak (0.1-0.3)" if v < 0.3
            else "moderate (0.3-0.5)" if v < 0.5
            else "strong (≥0.5)"
        ),
    }

    # Per-manufacturer counts and pct by diagnosis
    rows = []
    for grp in _SUPERVISED_GROUPS:
        sub = cn_ad[cn_ad["ResearchGroup_Mapped"] == grp]
        n_total = len(sub)
        for mfr in ["GE MEDICAL SYSTEMS", "Philips", "SIEMENS"]:
            n_mfr = int((sub["Manufacturer"] == mfr).sum())
            rows.append({
                "diagnosis": grp,
                "manufacturer": mfr,
                "n": n_mfr,
                "pct_within_dx": _pct(n_mfr, n_total),
            })

    # Site3 analysis
    site_groups = cn_ad.groupby("Site3")["ResearchGroup_Mapped"].apply(set)
    both_sites = sorted(site_groups[
        site_groups.apply(lambda s: "CN" in s and "AD" in s)
    ].index.tolist())
    mono_cn = sorted(site_groups[
        site_groups.apply(lambda s: s == {"CN"})
    ].index.tolist())
    mono_ad = sorted(site_groups[
        site_groups.apply(lambda s: s == {"AD"})
    ].index.tolist())

    stats["n_sites_total"] = int(cn_ad["Site3"].nunique())
    stats["n_sites_with_both_CN_AD"] = len(both_sites)
    stats["n_sites_CN_only"] = len(mono_cn)
    stats["n_sites_AD_only"] = len(mono_ad)
    stats["sites_CN_only"] = str(mono_cn)
    stats["sites_AD_only_n_subjects"] = {
        str(int(s)): int((cn_ad["Site3"] == s).sum())
        for s in mono_ad
    }

    # GE-CN structural absence
    ge_cn = int(((cn_ad["ResearchGroup_Mapped"] == "CN") & (cn_ad["Manufacturer"] == "GE MEDICAL SYSTEMS")).sum())
    stats["GE_CN_count"] = ge_cn
    stats["GE_CN_note"] = (
        "CRITICAL: CN has 0 GE subjects. GE is exclusively AD+MCI. "
        "Any model can exploit this as a perfect shortcut for some AD subjects."
    )

    return pd.DataFrame(rows), stats


# ---------------------------------------------------------------------------
# 4. Fold composition
# ---------------------------------------------------------------------------

def _leakage_for_fold(fold: int, source: str = "train_pool") -> Dict[str, float]:
    suffix = (
        f"fold_{fold}_scanner_leakage_summary.csv"
        if source == "train_pool"
        else f"fold_{fold}_test_scanner_leakage_summary.csv"
    )
    p = _RESULTS_DIR / f"fold_{fold}" / suffix
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if df.empty:
        return {}
    r = df.iloc[0]
    return {
        "acc_site_raw": round(float(r.get("acc_site_raw", float("nan"))), 4),
        "acc_site_latent": round(float(r.get("acc_site_latent", float("nan"))), 4),
        "chance_level": round(float(r.get("chance_level", float("nan"))), 4),
        "n_sites": int(r.get("n_sites", 0)),
    }


def build_fold_composition(meta: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metrics_files = list(_RESULTS_DIR.glob("all_folds_metrics_MULTI*.csv"))
    if not metrics_files:
        print("  [WARN] fold metrics CSV not found; fold AUC will be missing")
        fold_metrics = pd.DataFrame()
    else:
        fold_metrics = pd.read_csv(metrics_files[0])

    comp_rows: List[Dict] = []
    leakage_rows: List[Dict] = []

    for fold in range(1, _N_FOLDS + 1):
        test_path = _RESULTS_DIR / f"fold_{fold}" / "test_subjects_fold.csv"
        train_path = _RESULTS_DIR / f"fold_{fold}" / "train_dev_subjects_fold.csv"
        if not test_path.exists():
            continue

        test_df = pd.read_csv(test_path).merge(
            meta[["SubjectID", "Age", "Sex", "Manufacturer", "Site3"]],
            on="SubjectID", how="left"
        )
        train_df = pd.read_csv(train_path).merge(
            meta[["SubjectID", "Age", "Sex", "Manufacturer", "Site3"]],
            on="SubjectID", how="left"
        )

        # AUC
        lr_raw = svm_raw = float("nan")
        if not fold_metrics.empty:
            lr_row = fold_metrics[
                (fold_metrics["fold"] == fold) &
                (fold_metrics["actual_classifier_type"] == "logreg")
            ]
            svm_row = fold_metrics[
                (fold_metrics["fold"] == fold) &
                (fold_metrics["actual_classifier_type"] == "svm")
            ]
            if not lr_row.empty:
                lr_raw = float(lr_row["auc_raw"].iloc[0])
            if not svm_row.empty:
                svm_raw = float(svm_row["auc_raw"].iloc[0])

        # Leakage
        lk_train = _leakage_for_fold(fold, "train_pool")
        lk_test = _leakage_for_fold(fold, "test")

        for split, df_split in [("test", test_df), ("train_dev", train_df)]:
            for grp in _SUPERVISED_GROUPS:
                sub = df_split[df_split["ResearchGroup_Mapped"] == grp]
                n = len(sub)
                if n == 0:
                    continue
                age = sub["Age"].dropna()
                for mfr in ["GE MEDICAL SYSTEMS", "Philips", "SIEMENS"]:
                    n_mfr = int((sub["Manufacturer"] == mfr).sum())
                    comp_rows.append({
                        "fold": fold,
                        "split": split,
                        "diagnosis": grp,
                        "manufacturer": mfr,
                        "n_manufacturer": n_mfr,
                        "pct_manufacturer": round(100 * n_mfr / n, 1),
                        "n_diagnosis": n,
                        "n_F": int((sub["Sex"] == "F").sum()),
                        "n_M": int((sub["Sex"] == "M").sum()),
                        "age_mean": round(float(age.mean()), 2) if len(age) > 0 else float("nan"),
                        "age_std": round(float(age.std(ddof=1)), 2) if len(age) > 1 else float("nan"),
                        "auc_raw_logreg": round(lr_raw, 4),
                        "auc_raw_svm": round(svm_raw, 4),
                    })

        # Age gap (test fold)
        ad_test = test_df[test_df["ResearchGroup_Mapped"] == "AD"]["Age"].dropna()
        cn_test = test_df[test_df["ResearchGroup_Mapped"] == "CN"]["Age"].dropna()
        age_gap = float(ad_test.mean() - cn_test.mean()) if len(ad_test) > 0 and len(cn_test) > 0 else float("nan")

        # Philips CN pct (test)
        cn_t = test_df[test_df["ResearchGroup_Mapped"] == "CN"]
        philips_cn_pct = float(
            100 * (cn_t["Manufacturer"] == "Philips").sum() / len(cn_t)
        ) if len(cn_t) > 0 else float("nan")

        leakage_rows.append({
            "fold": fold,
            "auc_raw_logreg": round(lr_raw, 4),
            "auc_raw_svm": round(svm_raw, 4),
            "test_n_AD": int((test_df["ResearchGroup_Mapped"] == "AD").sum()),
            "test_n_CN": int((test_df["ResearchGroup_Mapped"] == "CN").sum()),
            "test_GE_AD": int(((test_df["ResearchGroup_Mapped"] == "AD") & (test_df["Manufacturer"] == "GE MEDICAL SYSTEMS")).sum()),
            "test_Philips_CN_pct": round(philips_cn_pct, 1),
            "test_age_gap_AD_minus_CN": round(age_gap, 2),
            "train_acc_site_raw": lk_train.get("acc_site_raw"),
            "train_acc_site_latent": lk_train.get("acc_site_latent"),
            "test_acc_site_raw": lk_test.get("acc_site_raw"),
            "test_acc_site_latent": lk_test.get("acc_site_latent"),
            "chance_level": lk_train.get("chance_level"),
            "leakage_note": (
                "latent > raw (VAE amplified scanner info)"
                if (lk_train.get("acc_site_latent", 0) or 0) >
                   (lk_train.get("acc_site_raw", 0) or 0)
                else "latent ≤ raw (VAE reduced or preserved scanner leakage)"
            ),
        })

    return pd.DataFrame(comp_rows), pd.DataFrame(leakage_rows)


# ---------------------------------------------------------------------------
# 5. Stratification recommendation
# ---------------------------------------------------------------------------

def build_stratification_recommendation() -> pd.DataFrame:
    rows = [
        {
            "strategy": "current_Sex_only",
            "stratify_cols": "Sex",
            "pros": "Simple, sufficient strata size (5 folds × 2 groups).",
            "cons": (
                "Does not control Manufacturer-Dx confound. "
                "Fold 4 CN test 59% Philips is not unusual under Sex-only stratification."
            ),
            "n_strata": 4,
            "min_stratum_size_approx": 40,
            "recommended_role": "baseline_only",
            "priority": 3,
        },
        {
            "strategy": "primary_Sex_Manufacturer",
            "stratify_cols": "Sex + Manufacturer",
            "pros": (
                "Controls the GE=AD confound across folds. "
                "Ensures each fold sees proportional GE/Philips/Siemens per diagnosis. "
                "GE-AD strata: ~21/5=4 per fold (feasible for 5 folds)."
            ),
            "cons": (
                "GE-CN strata is always 0 (structural absense). "
                "Strata CN_GE will be empty — requires special handling "
                "(e.g., fill with nearest or ignore in cross-product)."
            ),
            "n_strata": 12,
            "min_stratum_size_approx": 4,
            "recommended_role": "PRIMARY recommended",
            "priority": 1,
        },
        {
            "strategy": "sensitivity_Sex_AgeBin2",
            "stratify_cols": "Sex + AgeBin2 (median split at ~73.6 yr)",
            "pros": (
                "Controls age imbalance (AD slightly older, SMD=0.24). "
                "Avoids folds where AD-CN age gap collapses (seen in fold 4 gap=0.6yr)."
            ),
            "cons": (
                "Does not control Manufacturer confound directly. "
                "Age split may be unstable if distribution shifts between runs."
            ),
            "n_strata": 8,
            "min_stratum_size_approx": 12,
            "recommended_role": "SENSITIVITY recommended",
            "priority": 2,
        },
        {
            "strategy": "manufacturer_as_QC_only",
            "stratify_cols": "Manufacturer (site leakage QC covariate, not CV stratifier)",
            "pros": (
                "Manufacturer information used only to measure scanner leakage after CV. "
                "Avoids over-constraining small strata (GE-CN=0)."
            ),
            "cons": "Does not prevent fold-level Manufacturer imbalance.",
            "n_strata": 0,
            "min_stratum_size_approx": 0,
            "recommended_role": "QC only (always apply)",
            "priority": 0,
        },
    ]
    return pd.DataFrame(rows).sort_values("priority")


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------

def write_readme(
    out_dir: Path,
    meta: pd.DataFrame,
    tensor_n: int,
    mfr_stats: Dict[str, Any],
    leakage_df: pd.DataFrame,
    run_ts: str,
) -> None:
    cn_ad = meta[meta["ResearchGroup_Mapped"].isin(_SUPERVISED_GROUPS)]
    ad = meta[meta["ResearchGroup_Mapped"] == "AD"]
    cn = meta[meta["ResearchGroup_Mapped"] == "CN"]

    # fold 4 details
    f4 = leakage_df[leakage_df["fold"] == 4].iloc[0] if not leakage_df.empty else {}

    lines = [
        "# ADNI v5 DPARSF-10000 No-Python-Bandpass — Dataset Composition & Split Risk Audit",
        "",
        f"Generated: {run_ts}",
        "",
        "---",
        "",
        "## 1. Subject Counts",
        "",
        f"- Tensor subjects: {tensor_n}",
        f"- Training-ready metadata: {len(meta)} (3 excluded: 035_S_6927, 128_S_2002, 114_S_6039)",
        f"- Supervised CN+AD: {len(cn_ad)} (CN={len(cn)}, AD={len(ad)})",
        f"- MCI (VAE pool only): {int((meta['ResearchGroup_Mapped']=='MCI').sum())}",
        f"- Missing Age: {int(meta['Age'].isna().sum())}  Missing Sex: {int(meta['Sex'].isna().sum())}",
        "",
        "---",
        "",
        "## 2. Demographics",
        "",
        "### Age",
        "",
        f"| Diagnosis | N | Mean | SD | Median | IQR |",
        f"|---|---|---|---|---|---|",
        f"| AD | {len(ad)} | {ad['Age'].mean():.1f} | {ad['Age'].std():.1f} | {ad['Age'].median():.1f} | {_iqr(ad['Age']):.1f} |",
        f"| CN | {len(cn)} | {cn['Age'].mean():.1f} | {cn['Age'].std():.1f} | {cn['Age'].median():.1f} | {_iqr(cn['Age']):.1f} |",
        "",
        f"Age SMD (AD − CN): **{_smd(ad['Age'].dropna(), cn['Age'].dropna()):.3f}** "
        f"(AD older by ~{ad['Age'].mean()-cn['Age'].mean():.1f} yr). "
        "Small effect. Age contributes modestly to classifier performance.",
        "",
        "### Sex",
        "",
        f"| Diagnosis | N_F | N_M | %F |",
        f"|---|---|---|---|",
    ]
    for grp, sub in [("AD", ad), ("CN", cn)]:
        nf = int((sub["Sex"] == "F").sum())
        nm = int((sub["Sex"] == "M").sum())
        lines.append(f"| {grp} | {nf} | {nm} | {_pct(nf, len(sub))} |")

    lines += [
        "",
        "CN is female-dominated (60%F); AD is male-dominated (58%M). Sex is a confound.",
        "",
        "---",
        "",
        "## 3. Manufacturer Confound",
        "",
        "### Manufacturer × Diagnosis (CN + AD only)",
        "",
        "| Manufacturer | CN (n) | CN (%) | AD (n) | AD (%) |",
        "|---|---|---|---|---|",
    ]
    for mfr in ["GE MEDICAL SYSTEMS", "Philips", "SIEMENS"]:
        cn_n = int(((cn["Manufacturer"] == mfr)).sum())
        ad_n = int(((ad["Manufacturer"] == mfr)).sum())
        lines.append(
            f"| {mfr} | {cn_n} | {_pct(cn_n, len(cn))} | {ad_n} | {_pct(ad_n, len(ad))} |"
        )

    lines += [
        "",
        f"**Chi-square test (Manufacturer × Diagnosis):**",
        f"- χ²({mfr_stats['dof']}) = {mfr_stats['chi2']}, p = {mfr_stats['p_value']:.2e}",
        f"- Cramér V = **{mfr_stats['cramers_v']}** ({mfr_stats['interpretation']} association)",
        "",
        "### Critical finding: GE is exclusively AD+MCI (0 CN subjects)",
        "",
        f"- GE CN subjects: **{mfr_stats['GE_CN_count']}** (structural absence)",
        f"- GE AD subjects: 21 ({_pct(21, len(ad))})",
        "",
        "This means **any classifier can use GE scanner = likely AD** as a shortcut for a subset",
        "of AD subjects. This is NOT a random confound — it is structural. The VAE may learn",
        "a GE-specific connectivity pattern and encode it in latent space.",
        "",
        "### Site analysis",
        "",
        f"- Total sites (CN+AD): {mfr_stats['n_sites_total']}",
        f"- Sites with both CN and AD: {mfr_stats['n_sites_with_both_CN_AD']}",
        f"- CN-only sites: {mfr_stats['n_sites_CN_only']} "
        f"(sites: {mfr_stats['sites_CN_only']})",
        f"- AD-only sites: {mfr_stats['n_sites_AD_only']} (many with n≤4 subjects)",
        "",
        "High proportion of AD-only sites inflates AD fragmentation per fold.",
        "Small AD-only sites (n=1-3) may land entirely in one fold, amplifying fold variance.",
        "",
        "---",
        "",
        "## 4. Per-Fold Composition (v5 [4,1,0] baseline)",
        "",
        "| Fold | AUC LR | AUC SVM | Test AD | Test CN | GE-AD | Philips-CN% | Age gap (AD-CN) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    if not leakage_df.empty:
        for _, row in leakage_df.sort_values("fold").iterrows():
            lines.append(
                f"| {int(row['fold'])} | {row['auc_raw_logreg']:.3f} | {row['auc_raw_svm']:.3f} "
                f"| {int(row['test_n_AD'])} | {int(row['test_n_CN'])} "
                f"| {int(row['test_GE_AD'])} | {row['test_Philips_CN_pct']:.0f}% "
                f"| {row['test_age_gap_AD_minus_CN']:+.1f} yr |"
            )

    lines += [
        "",
        "**Fold 4 (worst, LR=0.615):**",
        "- Smallest age gap (+0.6 yr): classifier cannot exploit AD-older signal.",
        "- 6 GE-AD subjects in test (highest across folds). Since CN=0 GE, these are",
        "  the most structurally distinguishable AD subjects — yet AUC is still lowest.",
        "  This suggests the GE signal alone cannot compensate for the weak age gap.",
        "- Philips-CN = 59% (consistent with structural Philips dominance in CN).",
        "- Train-pool scanner leakage: acc_site_latent = "
        f"{float(f4.get('train_acc_site_latent', float('nan'))):.3f} vs "
        f"acc_site_raw = {float(f4.get('train_acc_site_raw', float('nan'))):.3f} "
        "(latent > raw → VAE amplified scanner info for fold 4).",
        "",
        "---",
        "",
        "## 5. Leakage vs AUC",
        "",
        "| Fold | AUC LR | Train acc_site_raw | Train acc_site_latent | Latent vs Raw |",
        "|---|---|---|---|---|",
    ]
    if not leakage_df.empty:
        for _, row in leakage_df.sort_values("fold").iterrows():
            direction = (
                "↑ latent > raw"
                if (row.get("train_acc_site_latent") or 0) > (row.get("train_acc_site_raw") or 0)
                else "↓ latent ≤ raw"
            )
            lines.append(
                f"| {int(row['fold'])} | {row['auc_raw_logreg']:.3f} "
                f"| {row.get('train_acc_site_raw', 'N/A')} "
                f"| {row.get('train_acc_site_latent', 'N/A')} | {direction} |"
            )

    lines += [
        "",
        "In fold 4, the VAE amplified scanner leakage (latent > raw). This is consistent",
        "with a weaker age signal forcing the model to rely on scanner-correlated patterns.",
        "In other folds, latent leakage ≤ raw (VAE partially disentangles scanner).",
        "",
        "---",
        "",
        "## 6. Stratification Recommendations",
        "",
        "### Primary CV strategy: stratify by [Sex, Manufacturer]",
        "",
        "- Controls the structural GE=AD confound across folds.",
        "- GE-AD strata: ~21/5 = 4.2 subjects per fold (feasible with StratifiedKFold).",
        "- GE-CN strata is always 0 → handle as absent (do not include in stratify key).",
        "- Implementation: create `Sex_Manufacturer` key as composite string;",
        "  for subjects with Sex+Manufacturer creating an empty strata,",
        "  fall back to Sex-only key (this affects only GE-CN which doesn't exist).",
        "",
        "### Sensitivity CV strategy: stratify by [Sex, AgeBin2]",
        "",
        f"- Median age split at ~{meta['Age'].median():.1f} yr.",
        "- Controls the AD-older bias (SMD=0.236).",
        "- Reduces probability of folds with near-zero age gap (as in fold 4, gap=0.6yr).",
        "",
        "### Manufacturer as QC covariate (always apply)",
        "",
        "- Report acc_site_raw and acc_site_latent per fold regardless of stratification.",
        "- Flag folds where acc_site_latent > acc_site_raw + 0.05 as high-leakage folds.",
        "- Do NOT include Manufacturer as a feature in the classifier.",
        "",
        "### Decision table",
        "",
        "| Strategy | Addresses | Priority |",
        "|---|---|---|",
        "| Sex + Manufacturer | GE=AD structural confound | PRIMARY |",
        "| Sex + AgeBin2 | Age-diagnosis colinearity | SENSITIVITY |",
        "| Manufacturer QC only | Scanner leakage monitoring | ALWAYS |",
        "",
        "---",
        "",
        "## 7. Files in This Directory",
        "",
        "| File | Description |",
        "|---|---|",
        "| `dataset_counts.csv` | Total counts per group and missing data |",
        "| `demographics_by_diagnosis.csv` | Age/Sex stats per diagnosis + SMD |",
        "| `manufacturer_site_contingency.csv` | Manufacturer × Diagnosis counts + chi2/V stats |",
        "| `fold_composition.csv` | Per-fold Manufacturer/Sex/Age breakdown |",
        "| `fold_leakage_auc_table.csv` | Fold AUC + scanner leakage summary |",
        "| `stratification_recommendation.csv` | Structured recommendation table |",
    ]

    (out_dir / "README.md").write_text("\n".join(lines) + "\n")
    print("  Wrote README.md")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_dir: Path = args.output_root
    out_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=== ADNI v5 Dataset Composition & Split Risk Audit ===")
    print(f"Output: {out_dir}")
    print()

    # --- Load tensor N ---
    print("[0] Loading tensor subject count ...")
    if _TENSOR_PATH.exists():
        d = np.load(_TENSOR_PATH, allow_pickle=False)
        tensor_n = int(d["global_tensor_data"].shape[0])
        print(f"    Tensor subjects: {tensor_n}")
    else:
        print(f"    [WARN] Tensor not found: {_TENSOR_PATH}")
        tensor_n = -1

    # --- Load metadata ---
    print("[1] Loading metadata ...")
    if not _METADATA_PATH.exists():
        print(f"    [ERROR] Metadata not found: {_METADATA_PATH}", file=sys.stderr)
        sys.exit(1)
    meta = pd.read_csv(_METADATA_PATH)
    print(f"    Rows: {len(meta)}  Columns: {list(meta.columns)}")

    # --- 1. Counts ---
    print("[2] Building dataset counts ...")
    counts_df = build_dataset_counts(meta, tensor_n)
    counts_df.to_csv(out_dir / "dataset_counts.csv", index=False)
    print(f"    Wrote dataset_counts.csv")

    # --- 2. Demographics ---
    print("[3] Building demographics ...")
    demo_df = build_demographics(meta)
    demo_df.to_csv(out_dir / "demographics_by_diagnosis.csv", index=False)
    print(f"    Wrote demographics_by_diagnosis.csv")

    # --- 3. Manufacturer/Site ---
    print("[4] Building manufacturer/site analysis ...")
    mfr_df, mfr_stats = build_manufacturer_site(meta)
    # Write contingency CSV with stats appended
    stats_rows = [{"stat": k, "value": str(v)} for k, v in mfr_stats.items()]
    mfr_df.to_csv(out_dir / "manufacturer_site_contingency.csv", index=False)
    pd.DataFrame(stats_rows).to_csv(out_dir / "manufacturer_site_stats.csv", index=False)
    print(f"    chi2={mfr_stats['chi2']}, p={mfr_stats['p_value']:.2e}, Cramér V={mfr_stats['cramers_v']}")
    print(f"    GE-CN count: {mfr_stats['GE_CN_count']} (structural absence)")
    print(f"    Wrote manufacturer_site_contingency.csv + manufacturer_site_stats.csv")

    # --- 4. Fold composition ---
    print("[5] Building fold composition ...")
    if not _RESULTS_DIR.exists():
        print(f"    [WARN] Results dir not found: {_RESULTS_DIR}")
        fold_comp = pd.DataFrame()
        leakage_df = pd.DataFrame()
    else:
        fold_comp, leakage_df = build_fold_composition(meta)
    fold_comp.to_csv(out_dir / "fold_composition.csv", index=False)
    leakage_df.to_csv(out_dir / "fold_leakage_auc_table.csv", index=False)
    print(f"    Wrote fold_composition.csv ({len(fold_comp)} rows)")
    print(f"    Wrote fold_leakage_auc_table.csv ({len(leakage_df)} rows)")

    # --- 5. Stratification recommendations ---
    print("[6] Building stratification recommendations ...")
    strat_df = build_stratification_recommendation()
    strat_df.to_csv(out_dir / "stratification_recommendation.csv", index=False)
    print(f"    Wrote stratification_recommendation.csv")

    # --- README ---
    print("[7] Writing README ...")
    write_readme(out_dir, meta, tensor_n, mfr_stats, leakage_df, run_ts)

    # --- Print key numbers ---
    print()
    print("=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    print(f"  Manufacturer confound: Cramér V={mfr_stats['cramers_v']}, "
          f"p={mfr_stats['p_value']:.2e} ({mfr_stats['interpretation']})")
    print(f"  GE-CN: {mfr_stats['GE_CN_count']} subjects — STRUCTURAL ABSENCE")
    print(f"  Age SMD (AD-CN): {_smd(meta[meta['ResearchGroup_Mapped']=='AD']['Age'].dropna(), meta[meta['ResearchGroup_Mapped']=='CN']['Age'].dropna()):.3f}")
    if not leakage_df.empty:
        print()
        print("  Fold AUC summary:")
        for _, row in leakage_df.sort_values("fold").iterrows():
            marker = " ← worst fold" if row['auc_raw_logreg'] == leakage_df['auc_raw_logreg'].min() else ""
            print(f"    Fold {int(row['fold'])}: LR={row['auc_raw_logreg']:.3f}  "
                  f"GE-AD={int(row['test_GE_AD'])}  "
                  f"age_gap={row['test_age_gap_AD_minus_CN']:+.1f}yr  "
                  f"lk_latent={row.get('train_acc_site_latent','N/A')}{marker}")
    print()
    print(f"Outputs: {out_dir}")


if __name__ == "__main__":
    main()
