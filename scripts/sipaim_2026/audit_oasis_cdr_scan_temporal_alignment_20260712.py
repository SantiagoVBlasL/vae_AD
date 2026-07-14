#!/usr/bin/env python3
"""OASIS CDR-to-scan temporal alignment audit.

Read-only. No training, no calibration, no threshold fitting, no OASIS-based
promotion decisions. Builds three CDR-to-scan mappings (nearest within +/-180d,
nearest within +/-365d, latest-at-or-before capped at 365d) from the official
longitudinal OASIS3 CDR table, reports unmapped scans separately, and
recomputes CDR-severity association metrics (pairwise AUC, bootstrap CI, raw
Spearman, Age/Sex-adjusted partial rank association) for each mapping.

CDR is treated throughout as a severity indicator, not an AD-etiology label:
CDR>=1 is reported as "CDR>=1 (high severity)", never as "AD".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, spearmanr, pearsonr

PROJECT_ROOT = Path("/home/diego/proyectos/vae_AD")
OUT_DIR = PROJECT_ROOT / "results/sipaim_2026/final_blocker_resolution_20260712"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SUBJ_TABLE = PROJECT_ROOT / "results/sipaim_2026/oasis_label_sensitivity_20260710/oasis_label_sensitivity_subject_table.csv"
CDR_TABLE = Path("/media/diego/Datos/vae_AD_data/OASIS3/metadata_raw/imported/OASIS3_UDSb4_cdr.csv")

BOOT_N = 10000
BOOT_SEED = 20260710


def log(msg: str) -> None:
    print(msg, flush=True)


# ── 1. Load current-180 subjects with scan day, Age, Sex, frozen score ──────
subj = pd.read_csv(SUBJ_TABLE)
subj["OASISID"] = subj["subject_id"].str.replace("sub-", "", regex=False)
subj["scan_day"] = subj["experiment_id"].str.extract(r"_d(\d+)$").astype(float)
assert subj["scan_day"].notna().all(), "some current-180 subjects have unparseable scan day"
subj = subj[["OASISID", "subject_id", "session_id", "experiment_id", "scan_day", "Age", "Sex", "score_locked"]].copy()
subj["sex_male"] = (subj["Sex"] == "M").astype(int)
log(f"Loaded {len(subj)} current-180 subjects with scan_day parsed from experiment_id.")

# ── 2. Load full longitudinal CDR table, restrict to current-180 subjects ──
cdr = pd.read_csv(CDR_TABLE, usecols=["OASISID", "OASIS_session_label", "days_to_visit", "CDRTOT", "dx1"])
cdr = cdr.dropna(subset=["CDRTOT", "days_to_visit"]).copy()
cdr["days_to_visit"] = pd.to_numeric(cdr["days_to_visit"], errors="coerce")
cdr = cdr[cdr["OASISID"].isin(subj["OASISID"])].copy()
log(f"Loaded {len(cdr)} CDR visit-rows across {cdr['OASISID'].nunique()} of the 180 subjects.")


def pick_nearest_within(visits: pd.DataFrame, scan_day: float, window: float):
    v = visits.assign(abs_gap=(visits["days_to_visit"] - scan_day).abs())
    v = v[v["abs_gap"] <= window]
    if v.empty:
        return None
    v = v.sort_values(["abs_gap", "days_to_visit"], ascending=[True, True])
    return v.iloc[0]


def pick_latest_before(visits: pd.DataFrame, scan_day: float, max_gap: float):
    v = visits[(visits["days_to_visit"] <= scan_day) & ((scan_day - visits["days_to_visit"]) <= max_gap)]
    if v.empty:
        return None
    v = v.sort_values("days_to_visit", ascending=False)
    return v.iloc[0]


rows = []
for _, s in subj.iterrows():
    visits = cdr[cdr["OASISID"] == s["OASISID"]]
    scan_day = s["scan_day"]
    rec = dict(OASISID=s["OASISID"], subject_id=s["subject_id"], session_id=s["session_id"],
               experiment_id=s["experiment_id"], scan_day=scan_day, Age=s["Age"], Sex=s["Sex"],
               sex_male=s["sex_male"], score_locked=s["score_locked"], n_cdr_visits_on_record=len(visits))

    nearest_any = pick_nearest_within(visits, scan_day, np.inf) if len(visits) else None
    rec["nearest_any_day_gap"] = float(nearest_any["abs_gap"]) if nearest_any is not None else np.nan

    for label, picker, arg in [
        ("A_nearest_180d", pick_nearest_within, 180.0),
        ("B_nearest_365d", pick_nearest_within, 365.0),
        ("C_latest_before_365d", pick_latest_before, 365.0),
    ]:
        v = picker(visits, scan_day, arg)
        if v is None:
            rec[f"{label}_CDRTOT"] = np.nan
            rec[f"{label}_day_gap"] = np.nan
            rec[f"{label}_dx1"] = None
            rec[f"{label}_mapped"] = False
        else:
            gap = float(v["days_to_visit"] - scan_day)  # signed: negative = CDR visit before scan
            rec[f"{label}_CDRTOT"] = float(v["CDRTOT"])
            rec[f"{label}_day_gap"] = gap
            rec[f"{label}_dx1"] = v["dx1"]
            rec[f"{label}_mapped"] = True
    rows.append(rec)

align = pd.DataFrame(rows)
for label in ["A_nearest_180d", "B_nearest_365d", "C_latest_before_365d"]:
    def grp(v):
        if pd.isna(v):
            return "unmapped"
        if v == 0:
            return "CDR 0"
        if v == 0.5:
            return "CDR 0.5"
        return "CDR >= 1 (high severity)"
    align[f"{label}_cdr_group"] = align[f"{label}_CDRTOT"].apply(grp)

align.to_csv(OUT_DIR / "oasis_cdr_scan_alignment.csv", index=False)
log(f"Wrote {OUT_DIR / 'oasis_cdr_scan_alignment.csv'} ({len(align)} rows)")

# ── 3. Unmapped-scan report ─────────────────────────────────────────────────
unmapped_rows = []
for label in ["A_nearest_180d", "B_nearest_365d", "C_latest_before_365d"]:
    um = align[~align[f"{label}_mapped"]]
    for _, r in um.iterrows():
        unmapped_rows.append(dict(
            mapping=label, subject_id=r["subject_id"], OASISID=r["OASISID"], scan_day=r["scan_day"],
            n_cdr_visits_on_record=r["n_cdr_visits_on_record"],
            nearest_available_cdr_day_gap=r["nearest_any_day_gap"],
            reason="no CDR visit within mapping window" if r["n_cdr_visits_on_record"] > 0 else "no CDR visit on record at all",
        ))
unmapped_df = pd.DataFrame(unmapped_rows)
unmapped_df.to_csv(OUT_DIR / "_unmapped_scans.csv", index=False)
log(f"Unmapped scan counts: {unmapped_df.groupby('mapping').size().to_dict() if len(unmapped_df) else {}}")

# ── 4. Metrics per mapping ──────────────────────────────────────────────────
def bootstrap_auc_ci(y, s, rng, n_boot=BOOT_N):
    n = len(y)
    aucs = []
    idx_pool = np.arange(n)
    for _ in range(n_boot):
        idx = rng.choice(idx_pool, size=n, replace=True)
        yt = y[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, s[idx]))
    aucs = np.array(aucs)
    return float(aucs.mean()), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)), int(len(aucs))


def partial_spearman(score, cdr_ordinal, age, sex_male):
    """Partial Spearman of score vs CDR ordinal, controlling for Age and Sex,
    via rank-residual regression (Pearson correlation of residualized ranks)."""
    r_score = rankdata(score)
    r_cdr = rankdata(cdr_ordinal)
    X = np.column_stack([age, sex_male]).astype(float)
    reg_score = LinearRegression().fit(X, r_score)
    resid_score = r_score - reg_score.predict(X)
    reg_cdr = LinearRegression().fit(X, r_cdr)
    resid_cdr = r_cdr - reg_cdr.predict(X)
    rho, pval = pearsonr(resid_score, resid_cdr)
    return float(rho), float(pval)


metrics_rows = []
group_rows = []
for label in ["A_nearest_180d", "B_nearest_365d", "C_latest_before_365d"]:
    mapped = align[align[f"{label}_mapped"]].copy()
    n_mapped = len(mapped)
    n_unmapped = len(align) - n_mapped

    counts = mapped[f"{label}_cdr_group"].value_counts().to_dict()
    for grp_name, n in counts.items():
        group_rows.append(dict(mapping=label, cdr_group=grp_name, n=n))
    group_rows.append(dict(mapping=label, cdr_group="unmapped", n=n_unmapped))

    # Pairwise AUC: CDR0 vs CDR>=1 (high severity), CDR0.5 excluded
    pw = mapped[mapped[f"{label}_cdr_group"].isin(["CDR 0", "CDR >= 1 (high severity)"])]
    y_pw = (pw[f"{label}_cdr_group"] == "CDR >= 1 (high severity)").astype(int).to_numpy()
    s_pw = pw["score_locked"].to_numpy()
    auc_pw = roc_auc_score(y_pw, s_pw) if len(np.unique(y_pw)) == 2 else np.nan
    rng = np.random.default_rng(BOOT_SEED)
    boot_mean, ci_lo, ci_hi, n_valid = bootstrap_auc_ci(y_pw, s_pw, rng) if len(np.unique(y_pw)) == 2 else (np.nan,) * 3 + (0,)

    # Raw Spearman: score vs full CDRTOT ordinal (0, 0.5, 1, 2, 3), all mapped subjects
    rho_raw, p_raw = spearmanr(mapped["score_locked"], mapped[f"{label}_CDRTOT"])

    # Age/Sex-adjusted partial rank association
    rho_partial, p_partial = partial_spearman(
        mapped["score_locked"].to_numpy(), mapped[f"{label}_CDRTOT"].to_numpy(),
        mapped["Age"].to_numpy(), mapped["sex_male"].to_numpy(),
    )

    metrics_rows.append(dict(
        mapping=label, n_mapped=n_mapped, n_unmapped=n_unmapped,
        n_cdr0=int((mapped[f'{label}_cdr_group'] == 'CDR 0').sum()),
        n_cdr05=int((mapped[f'{label}_cdr_group'] == 'CDR 0.5').sum()),
        n_cdr_ge1=int((mapped[f'{label}_cdr_group'] == 'CDR >= 1 (high severity)').sum()),
        n_pairwise_cdr0_vs_ge1=len(pw),
        pairwise_auc=auc_pw, pairwise_auc_boot_mean=boot_mean,
        pairwise_auc_ci_low=ci_lo, pairwise_auc_ci_high=ci_hi, n_boot_valid=n_valid,
        spearman_rho_raw=rho_raw, spearman_pvalue_raw=p_raw,
        partial_spearman_rho_age_sex_adj=rho_partial, partial_spearman_pvalue_age_sex_adj=p_partial,
        mean_abs_day_gap=float(mapped[f"{label}_day_gap"].abs().mean()),
        max_abs_day_gap=float(mapped[f"{label}_day_gap"].abs().max()),
    ))

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(OUT_DIR / "cdr_temporal_sensitivity_metrics.csv", index=False)
group_df = pd.DataFrame(group_rows)
log(metrics_df.to_string(index=False))
log(f"Wrote {OUT_DIR / 'cdr_temporal_sensitivity_metrics.csv'}")

with open(OUT_DIR / "_cdr_group_counts_by_mapping.json", "w") as f:
    json.dump(group_rows, f, indent=2)

log("Task 1 done.")
