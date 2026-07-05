#!/usr/bin/env python3
"""Read-only ADNI/OASIS age-sex and VAE-conditioning risk audit."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results/revision_bspc_2026/adni_oasis_age_sex_conditioning_risk_audit"

ADNI_METADATA = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/"
    "training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv"
)
OASIS_PILOT = ROOT / "results/revision_bspc_2026/oasis_tanda_2026_05_25_audit/subject_session_manifest.csv"
OASIS_NEXT = ROOT / "results/revision_bspc_2026/oasis_next_batch_selection_audit/selected_ideal_60CN_60AD.csv"
OASIS_SPLIT = ROOT / "results/revision_bspc_2026/oasis_60cn_60ad_calibration_test_protocol/split_calibration_test.csv"


def to_markdown(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.write_text(df.to_markdown(index=index) + "\n", encoding="utf-8")


def write_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT_DIR / f"{name}.csv", index=False)
    to_markdown(df, OUT_DIR / f"{name}.md", index=False)


def normalize_adni(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
    out["cohort"] = "ADNI_locked_v5_1b"
    out["cohort_group"] = "ADNI"
    out["diagnosis"] = out["ResearchGroup_Mapped"]
    out["age"] = pd.to_numeric(out["Age"], errors="coerce")
    out["sex_raw"] = out["Sex"].astype(str)
    out["sex_group"] = out["Sex"].astype(str).str.upper().replace({"FEMALE": "F", "MALE": "M"})
    out["subject_id"] = out["SubjectID"].astype(str)
    out["site_code"] = out["SubjectID"].astype(str).str.extract(r"^(\d{3})_S_")[0]
    out["Manufacturer"] = out["Manufacturer"].astype(str)
    return out[
        [
            "cohort",
            "cohort_group",
            "subject_id",
            "diagnosis",
            "age",
            "sex_raw",
            "sex_group",
            "Manufacturer",
            "site_code",
        ]
    ]


def normalize_oasis(df: pd.DataFrame, cohort: str, subset_col: str | None = None) -> pd.DataFrame:
    out = df[df["diagnosis"].isin(["CN", "AD_DEMENTIA"])].copy()
    if subset_col and subset_col in out.columns:
        out["cohort"] = cohort + "_" + out[subset_col].astype(str)
    else:
        out["cohort"] = cohort
    out["cohort_group"] = cohort
    out["age"] = pd.to_numeric(out["age_at_MR"], errors="coerce")
    out["sex_raw"] = out["sex"].astype(str)
    # Keep OASIS numeric sex codes explicit. A local authoritative data
    # dictionary was not present in the files consumed by this audit.
    out["sex_group"] = "OASIS_code_" + out["sex_raw"].str.replace(r"\.0$", "", regex=True)
    out["site_code"] = out.get("session_id", pd.Series(index=out.index, dtype=object)).astype(str)
    out["Manufacturer"] = out.get("Manufacturer", pd.Series(index=out.index, dtype=object)).astype(str)
    return out[
        [
            "cohort",
            "cohort_group",
            "subject_id",
            "diagnosis",
            "age",
            "sex_raw",
            "sex_group",
            "Manufacturer",
            "site_code",
        ]
    ]


def iqr(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return float(vals.quantile(0.75) - vals.quantile(0.25)) if len(vals) else np.nan


def pooled_smd(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce").dropna()
    y = pd.to_numeric(b, errors="coerce").dropna()
    if len(x) < 2 or len(y) < 2:
        return np.nan
    pooled = math.sqrt((float(x.var(ddof=1)) + float(y.var(ddof=1))) / 2.0)
    return float((x.mean() - y.mean()) / pooled) if pooled > 0 else np.nan


def sex_counts_string(series: pd.Series) -> str:
    counts = series.fillna("MISSING").astype(str).value_counts().sort_index()
    return ";".join([f"{idx}={int(val)}" for idx, val in counts.items()])


def build_summary(combined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (cohort, diagnosis), sub in combined.groupby(["cohort", "diagnosis"], dropna=False):
        ages = pd.to_numeric(sub["age"], errors="coerce").dropna()
        row = {
            "cohort": cohort,
            "diagnosis": diagnosis,
            "n": int(len(sub)),
            "age_mean": float(ages.mean()) if len(ages) else np.nan,
            "age_sd": float(ages.std(ddof=1)) if len(ages) > 1 else np.nan,
            "age_median": float(ages.median()) if len(ages) else np.nan,
            "age_q1": float(ages.quantile(0.25)) if len(ages) else np.nan,
            "age_q3": float(ages.quantile(0.75)) if len(ages) else np.nan,
            "age_iqr": iqr(sub["age"]),
            "age_min": float(ages.min()) if len(ages) else np.nan,
            "age_max": float(ages.max()) if len(ages) else np.nan,
            "sex_counts": sex_counts_string(sub["sex_group"]),
            "sex_raw_counts": sex_counts_string(sub["sex_raw"]),
            "manufacturer_counts": sex_counts_string(sub["Manufacturer"]),
        }
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["cohort", "diagnosis"]).reset_index(drop=True)
    summary["age_smd_AD_minus_CN_within_cohort"] = np.nan
    for cohort, sub in combined.groupby("cohort", dropna=False):
        cn = sub[sub["diagnosis"] == "CN"]["age"]
        ad_label = "AD" if "AD" in set(sub["diagnosis"]) else "AD_DEMENTIA"
        ad = sub[sub["diagnosis"] == ad_label]["age"]
        smd = pooled_smd(ad, cn)
        summary.loc[summary["cohort"].eq(cohort), "age_smd_AD_minus_CN_within_cohort"] = smd
    return summary


def diagnosis_tests(combined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    try:
        from scipy import stats
    except Exception as exc:  # pragma: no cover
        stats = None
        rows.append(
            {
                "cohort": "environment",
                "test_name": "scipy_import",
                "status": "failed",
                "statistic": np.nan,
                "p_value": np.nan,
                "effect": np.nan,
                "details": str(exc),
            }
        )

    for cohort, sub in combined.groupby("cohort", dropna=False):
        labels = set(sub["diagnosis"])
        ad_label = "AD" if "AD" in labels else "AD_DEMENTIA"
        cn_age = pd.to_numeric(sub.loc[sub["diagnosis"].eq("CN"), "age"], errors="coerce").dropna()
        ad_age = pd.to_numeric(sub.loc[sub["diagnosis"].eq(ad_label), "age"], errors="coerce").dropna()
        if len(cn_age) and len(ad_age):
            rows.append(
                {
                    "cohort": cohort,
                    "test_name": "age_descriptive_AD_minus_CN",
                    "status": "ok",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "effect": float(ad_age.mean() - cn_age.mean()),
                    "details": f"CN n={len(cn_age)} mean={cn_age.mean():.3f}; {ad_label} n={len(ad_age)} mean={ad_age.mean():.3f}",
                }
            )
        if stats is not None and len(cn_age) > 1 and len(ad_age) > 1:
            t_stat, t_p = stats.ttest_ind(ad_age, cn_age, equal_var=False, nan_policy="omit")
            u_stat, u_p = stats.mannwhitneyu(ad_age, cn_age, alternative="two-sided")
            ks_stat, ks_p = stats.ks_2samp(ad_age, cn_age, alternative="two-sided", mode="auto")
            rows.extend(
                [
                    {
                        "cohort": cohort,
                        "test_name": "welch_t_age_AD_vs_CN",
                        "status": "ok",
                        "statistic": float(t_stat),
                        "p_value": float(t_p),
                        "effect": float(ad_age.mean() - cn_age.mean()),
                        "details": "Age difference, AD minus CN mean.",
                    },
                    {
                        "cohort": cohort,
                        "test_name": "mann_whitney_age_AD_vs_CN",
                        "status": "ok",
                        "statistic": float(u_stat),
                        "p_value": float(u_p),
                        "effect": float(ad_age.median() - cn_age.median()),
                        "details": "Age rank shift, AD minus CN median.",
                    },
                    {
                        "cohort": cohort,
                        "test_name": "ks_age_AD_vs_CN",
                        "status": "ok",
                        "statistic": float(ks_stat),
                        "p_value": float(ks_p),
                        "effect": float(ks_stat),
                        "details": "Two-sample KS test for age distributions.",
                    },
                ]
            )
        if stats is not None:
            table = pd.crosstab(sub["diagnosis"], sub["sex_group"])
            if table.shape[0] == 2 and table.shape[1] >= 2:
                chi2, p_val, _, _ = stats.chi2_contingency(table)
                rows.append(
                    {
                        "cohort": cohort,
                        "test_name": "chi_square_sex_by_diagnosis",
                        "status": "ok",
                        "statistic": float(chi2),
                        "p_value": float(p_val),
                        "effect": np.nan,
                        "details": table.to_dict(),
                    }
                )
                if table.shape == (2, 2):
                    odds, fisher_p = stats.fisher_exact(table)
                    rows.append(
                        {
                            "cohort": cohort,
                            "test_name": "fisher_exact_sex_by_diagnosis",
                            "status": "ok",
                            "statistic": float(odds),
                            "p_value": float(fisher_p),
                            "effect": float(odds),
                            "details": table.to_dict(),
                        }
                    )

        model_df = sub[sub["diagnosis"].isin(["CN", ad_label])].dropna(subset=["age", "sex_group"]).copy()
        if len(model_df) >= 20 and model_df["diagnosis"].nunique() == 2:
            try:
                import statsmodels.api as sm

                y = model_df["diagnosis"].eq(ad_label).astype(int)
                X = pd.get_dummies(model_df[["age", "sex_group"]], columns=["sex_group"], drop_first=True, dtype=float)
                X = sm.add_constant(X, has_constant="add")
                fit = sm.Logit(y, X).fit(disp=False, maxiter=200)
                coef = float(fit.params.get("age", np.nan))
                rows.append(
                    {
                        "cohort": cohort,
                        "test_name": "logit_diagnosis_age_plus_sex",
                        "status": "ok",
                        "statistic": coef,
                        "p_value": float(fit.pvalues.get("age", np.nan)),
                        "effect": float(math.exp(coef)) if np.isfinite(coef) else np.nan,
                        "details": "Diagnosis ~ age + sex_group; effect is age odds ratio per year.",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "cohort": cohort,
                        "test_name": "logit_diagnosis_age_plus_sex",
                        "status": "failed",
                        "statistic": np.nan,
                        "p_value": np.nan,
                        "effect": np.nan,
                        "details": str(exc),
                    }
                )
    return pd.DataFrame(rows)


def age_mfr_site_tests(adni: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    try:
        from scipy import stats
    except Exception as exc:  # pragma: no cover
        return pd.DataFrame(
            [
                {
                    "scope": "ADNI",
                    "grouping": "environment",
                    "test_name": "scipy_import",
                    "status": "failed",
                    "statistic": np.nan,
                    "p_value": np.nan,
                    "details": str(exc),
                }
            ]
        )

    for grouping, min_n in [("Manufacturer", 2), ("site_code", 5), ("Manufacturer_site_code", 5)]:
        data = adni.copy()
        if grouping == "Manufacturer_site_code":
            data[grouping] = data["Manufacturer"].astype(str) + ":" + data["site_code"].astype(str)
        groups = []
        labels = []
        summaries = []
        for label, sub in data.groupby(grouping, dropna=False):
            ages = pd.to_numeric(sub["age"], errors="coerce").dropna()
            if len(ages) >= min_n:
                groups.append(ages)
                labels.append(str(label))
                summaries.append(f"{label}:n={len(ages)},mean={ages.mean():.2f}")
        if len(groups) >= 2:
            stat, p_val = stats.kruskal(*groups)
            rows.append(
                {
                    "scope": "ADNI_CN_AD",
                    "grouping": grouping,
                    "test_name": "kruskal_age_by_group",
                    "status": "ok",
                    "statistic": float(stat),
                    "p_value": float(p_val),
                    "n_groups_tested": len(groups),
                    "details": "; ".join(summaries[:25]),
                }
            )
    return pd.DataFrame(rows)


def write_risk_docs(
    summary: pd.DataFrame,
    tests: pd.DataFrame,
    age_mfr: pd.DataFrame,
) -> None:
    adni_age_test = tests[
        (tests["cohort"].eq("ADNI_locked_v5_1b")) & (tests["test_name"].eq("logit_diagnosis_age_plus_sex"))
    ]
    oasis_next_age_test = tests[
        (tests["cohort"].eq("OASIS_next_ideal_60CN_60AD"))
        & (tests["test_name"].eq("logit_diagnosis_age_plus_sex"))
    ]
    adni_age_p = adni_age_test["p_value"].iloc[0] if len(adni_age_test) else np.nan
    next_age_p = oasis_next_age_test["p_value"].iloc[0] if len(oasis_next_age_test) else np.nan

    risk = [
        "# Conditioning Risk Assessment",
        "",
        "## Age/Sex Distribution",
        "",
        "The OASIS next-batch selection and calibration/test split are tightly age/sex matched by design. "
        "The OASIS pilot and ADNI locked cohort are less controlled, and ADNI also spans multiple "
        "manufacturers/sites.",
        "",
        "## Conditional VAE Risk",
        "",
        "- Age is biologically entangled with AD risk and brain connectivity changes. Treating Age as a decoder "
        "condition can encourage the VAE to represent age-explainable structure outside `z`.",
        "- If AD-related variance is age-correlated, decoder conditioning can push clinically useful signal out "
        "of the latent code. This is especially risky for `z_only` readouts.",
        "- A `z + Age/Sex` readout may recover part of the removed signal, but the interpretation changes: the "
        "classifier then uses explicit demographic covariates to compensate for a demographically conditioned "
        "representation.",
        "- Sex conditioning has similar but usually smaller risk; it can reduce nuisance variance, but sex-related "
        "disease heterogeneity could also be attenuated.",
        "- Prior conditional Age/Sex FAST 3x3 results in this revision were negative/non-promoted, so there is "
        "currently no empirical support for a FULL conditional Age/Sex run.",
        "",
        "## Recommended Role of Age/Sex",
        "",
        "- **Classifier covariate only:** keep Age/Sex in Stage B readout as currently done.",
        "- **Calibration covariate:** do not use Age/Sex for primary OASIS threshold calibration in the planned "
        "60-subject calibration subset; sample size is too small and it would change the externally reported "
        "decision rule. Consider only as exploratory future work if a larger external calibration cohort is available.",
        "- **Decoder condition:** not recommended for the current manuscript model because it risks removing "
        "age-linked disease variance from `z` and prior FAST conditional audits did not improve AD/CN ranking.",
        "",
        "## Evidence Snapshot",
        "",
        f"- ADNI logit age effect p-value, diagnosis ~ age + sex: `{adni_age_p}`.",
        f"- OASIS next-batch logit age effect p-value, diagnosis ~ age + sex: `{next_age_p}`.",
        "- ADNI age differs by manufacturer/site; see `age_manufacturer_site_association.csv`.",
    ]
    (OUT_DIR / "conditioning_risk_assessment.md").write_text("\n".join(risk) + "\n", encoding="utf-8")

    final = [
        "# Final Recommendation",
        "",
        "**Recommendation: keep Age/Sex as classifier covariates only.**",
        "",
        "Do not promote Age/Sex decoder conditioning for the manuscript model. Age is not a pure nuisance in "
        "AD-vs-CN classification; it is correlated with disease risk and may carry clinically meaningful "
        "connectivity variance. Conditioning the decoder on Age could improve reconstruction semantics while "
        "weakening the latent disease signal used by Stage B.",
        "",
        "For OASIS external validation, use the pre-registered calibration/test protocol: threshold-only "
        "calibration on the calibration subset and locked evaluation on the untouched test subset. Do not add "
        "age-conditioned thresholding or age-conditioned model selection in the primary report.",
        "",
        "Sex should remain a reported subgroup and classifier covariate. Numeric OASIS sex codes were preserved "
        "as raw codes in this audit because no local authoritative coding dictionary was found in the consumed "
        "manifest files.",
    ]
    (OUT_DIR / "final_recommendation.md").write_text("\n".join(final) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now().isoformat(timespec="seconds")

    adni = normalize_adni(pd.read_csv(ADNI_METADATA))
    pilot = normalize_oasis(pd.read_csv(OASIS_PILOT), "OASIS_pilot_Tanda_2026_05_25")
    next_batch = normalize_oasis(pd.read_csv(OASIS_NEXT), "OASIS_next_ideal_60CN_60AD")
    split = normalize_oasis(pd.read_csv(OASIS_SPLIT), "OASIS_60CN_60AD_split", "protocol_subset")
    combined = pd.concat([adni, pilot, next_batch, split], ignore_index=True)

    summary = build_summary(combined)
    tests = diagnosis_tests(combined)
    age_mfr = age_mfr_site_tests(adni)

    write_table(summary, "age_sex_summary_by_cohort_diagnosis")
    write_table(tests, "age_diagnosis_association_tests")
    write_table(age_mfr, "age_manufacturer_site_association")
    write_risk_docs(summary, tests, age_mfr)

    command_log = {
        "script": str(Path(__file__).resolve()),
        "started_at": started,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "read_only_audit",
        "inputs": {
            "adni_metadata": str(ADNI_METADATA),
            "oasis_pilot": str(OASIS_PILOT),
            "oasis_next_batch": str(OASIS_NEXT),
            "oasis_calibration_test_split": str(OASIS_SPLIT),
        },
        "outputs": [
            "age_sex_summary_by_cohort_diagnosis.csv/.md",
            "age_diagnosis_association_tests.csv/.md",
            "age_manufacturer_site_association.csv/.md",
            "conditioning_risk_assessment.md",
            "final_recommendation.md",
            "command_log.json",
        ],
        "n_rows_combined": int(len(combined)),
        "no_training": True,
        "no_model_selection": True,
        "no_input_modification": True,
    }
    (OUT_DIR / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
    print(f"Wrote audit package to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
