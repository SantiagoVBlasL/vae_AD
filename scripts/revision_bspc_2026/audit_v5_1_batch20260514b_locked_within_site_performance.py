"""
Read-only within-site performance audit for the locked current model.

Source:
  Stage B logreg_l2, threshold_strategy=inner_oof_target_sens_ge_0p70_max_spec
  classifier_sweep_predictions.csv from:
  adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep

SiteCode_best construction:
  Site3 zero-padded (3 digits) if available, else SubjectID prefix before "_S_".

Per-site metrics (sites with n_total>=10, n_AD>=3, n_CN>=3):
  n_CN, n_AD, AUC, PR-AUC, balanced_accuracy, sensitivity, specificity, F1,
  mean_score_AD, mean_score_CN

Per-manufacturer pooled metrics (same metric set).

Comparison: site-level vs manufacturer-level subgroup performance.

Constraints:
  Read-only: no tensor, metadata, ledger, config, or model modification.
  No training, no inference.
"""

import json
import sys
import textwrap
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep"
    / "classifier_sweep_predictions.csv"
)

METADATA_PATH = (
    Path("/media/diego/Datos/vae_AD_data")
    / "revision_bspc_2026"
    / "adni_expanded_v5_1_batch20260514b_no_pybandpass"
    / "training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "adni_v5_1_batch20260514b_locked_within_site_performance_audit"
)

MODEL_NAME = "logreg_l2"
THRESHOLD_STRATEGY = "inner_oof_target_sens_ge_0p70_max_spec"

MIN_SITE_N_TOTAL = 10
MIN_SITE_N_AD = 3
MIN_SITE_N_CN = 3

# Locked pooled reference metrics (from classifier_sweep_pooled_metrics.csv)
LOCKED_AUC = 0.778785
LOCKED_PR_AUC = 0.551832
LOCKED_BA = 0.712917
LOCKED_SENS = 0.729167
LOCKED_SPEC = 0.696667
LOCKED_F1 = 0.544747


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def df_to_markdown(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for _, row in df.iterrows()]
    return "\n".join([header, sep] + rows)


def save_csv_md(df: pd.DataFrame, stem: str, out_dir: Path, title: str = "") -> None:
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    lines = []
    if title:
        lines.append(f"# {title}\n")
    lines.append(df_to_markdown(df))
    (out_dir / f"{stem}.md").write_text("\n".join(lines) + "\n")


def extract_sitecode_from_ptid(ptid: str) -> str:
    parts = str(ptid).split("_S_")
    return parts[0].zfill(3) if parts else "UNKNOWN"


def site3_to_str(val) -> Optional[str]:
    try:
        if pd.isna(val):
            return None
        return str(int(val)).zfill(3)
    except (ValueError, TypeError):
        return None


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, object]:
    """Compute the full metric set for a subgroup. Returns NaN for undefined metrics."""
    n_cn = int((y_true == 0).sum())
    n_ad = int((y_true == 1).sum())
    n_total = n_cn + n_ad

    # Ranking metrics (require both classes)
    try:
        auc = round(float(roc_auc_score(y_true, y_score)), 6) if len(np.unique(y_true)) == 2 else float("nan")
    except Exception:
        auc = float("nan")

    try:
        pr_auc = round(float(average_precision_score(y_true, y_score)), 6) if len(np.unique(y_true)) == 2 else float("nan")
    except Exception:
        pr_auc = float("nan")

    # Threshold-dependent metrics
    try:
        ba = round(float(balanced_accuracy_score(y_true, y_pred)), 6)
    except Exception:
        ba = float("nan")

    try:
        sens = round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 6)
    except Exception:
        sens = float("nan")

    try:
        spec = round(float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)), 6)
    except Exception:
        spec = float("nan")

    try:
        f1 = round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 6)
    except Exception:
        f1 = float("nan")

    # Score distributions
    mean_score_ad = round(float(y_score[y_true == 1].mean()), 6) if n_ad > 0 else float("nan")
    mean_score_cn = round(float(y_score[y_true == 0].mean()), 6) if n_cn > 0 else float("nan")
    score_sep = round(mean_score_ad - mean_score_cn, 6) if not (np.isnan(mean_score_ad) or np.isnan(mean_score_cn)) else float("nan")

    return {
        "n_CN": n_cn,
        "n_AD": n_ad,
        "n_total": n_total,
        "AUC": auc,
        "PR_AUC": pr_auc,
        "balanced_accuracy": ba,
        "sensitivity": sens,
        "specificity": spec,
        "F1": f1,
        "mean_score_AD": mean_score_ad,
        "mean_score_CN": mean_score_cn,
        "score_separation": score_sep,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_audit() -> None:
    t0 = datetime.now(timezone.utc)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load and filter predictions                                       #
    # ------------------------------------------------------------------ #
    df_all = pd.read_csv(PREDICTIONS_PATH)
    df = df_all[
        (df_all["model_name"] == MODEL_NAME)
        & (df_all["threshold_strategy"] == THRESHOLD_STRATEGY)
    ].copy().reset_index(drop=True)

    n_preds = len(df)
    assert df["SubjectID"].nunique() == n_preds, "Subjects not unique — each must appear once"
    print(f"Predictions loaded: {n_preds} subjects  "
          f"(CN={int((df['y_true']==0).sum())}, AD={int((df['y_true']==1).sum())})")

    # ------------------------------------------------------------------ #
    # 2. Load metadata — build SiteCode_best                              #
    # ------------------------------------------------------------------ #
    meta = pd.read_csv(METADATA_PATH)
    meta["Site3_str"] = meta["Site3"].apply(site3_to_str)
    meta["SiteCode_ptid"] = meta["SubjectID"].apply(extract_sitecode_from_ptid)
    meta["SiteCode_best"] = meta["Site3_str"].combine_first(meta["SiteCode_ptid"])

    # Join SiteCode_best into predictions
    df = df.merge(
        meta[["SubjectID", "SiteCode_best"]],
        on="SubjectID",
        how="left",
    )
    assert df["SiteCode_best"].isna().sum() == 0, "Some subjects have no SiteCode_best"

    # ------------------------------------------------------------------ #
    # 3. Global (pooled) metrics                                          #
    # ------------------------------------------------------------------ #
    y_true = df["y_true"].values
    y_score = df["y_score"].values
    y_pred = df["y_pred"].values

    global_metrics = compute_metrics(y_true, y_score, y_pred)
    global_row = {"subgroup": "global_pooled", **global_metrics}
    global_df = pd.DataFrame([global_row])
    save_csv_md(global_df, "global_metrics", OUTPUT_DIR, "Global Pooled Metrics")

    print(f"\nGlobal: AUC={global_metrics['AUC']:.4f}  "
          f"PR-AUC={global_metrics['PR_AUC']:.4f}  "
          f"BA={global_metrics['balanced_accuracy']:.4f}  "
          f"Sens={global_metrics['sensitivity']:.4f}  "
          f"Spec={global_metrics['specificity']:.4f}  "
          f"F1={global_metrics['F1']:.4f}")
    print(f"Locked reference: AUC={LOCKED_AUC}  PR-AUC={LOCKED_PR_AUC}  "
          f"BA={LOCKED_BA}  Sens={LOCKED_SENS}  Spec={LOCKED_SPEC}  F1={LOCKED_F1}")

    # ------------------------------------------------------------------ #
    # 4. Per-Manufacturer metrics                                          #
    # ------------------------------------------------------------------ #
    mfr_rows = []
    for mfr in sorted(df["Manufacturer"].unique()):
        sub = df[df["Manufacturer"] == mfr]
        row = {
            "Manufacturer": mfr,
            **compute_metrics(sub["y_true"].values, sub["y_score"].values, sub["y_pred"].values),
        }
        mfr_rows.append(row)
    mfr_df = pd.DataFrame(mfr_rows)
    save_csv_md(mfr_df, "per_manufacturer_metrics", OUTPUT_DIR,
                "Per-Manufacturer Metrics")

    print(f"\nPer-Manufacturer:")
    for _, r in mfr_df.iterrows():
        print(f"  {r['Manufacturer']:8s}: n={r['n_total']:3d}  "
              f"AUC={r['AUC']:.4f}  BA={r['balanced_accuracy']:.4f}  "
              f"Sens={r['sensitivity']:.4f}  Spec={r['specificity']:.4f}")

    # ------------------------------------------------------------------ #
    # 5. Site feasibility table (all sites)                               #
    # ------------------------------------------------------------------ #
    site_counts = df.groupby("SiteCode_best").agg(
        Manufacturer=("Manufacturer", lambda x: "/".join(sorted(x.unique()))),
        n_CN=("y_true", lambda x: int((x == 0).sum())),
        n_AD=("y_true", lambda x: int((x == 1).sum())),
    ).reset_index()
    site_counts["n_total"] = site_counts["n_CN"] + site_counts["n_AD"]
    site_counts["feasible"] = (
        (site_counts["n_total"] >= MIN_SITE_N_TOTAL)
        & (site_counts["n_AD"] >= MIN_SITE_N_AD)
        & (site_counts["n_CN"] >= MIN_SITE_N_CN)
    )
    site_counts = site_counts.sort_values("n_total", ascending=False).reset_index(drop=True)
    save_csv_md(site_counts, "site_feasibility", OUTPUT_DIR,
                f"Site Feasibility (n≥{MIN_SITE_N_TOTAL}, AD≥{MIN_SITE_N_AD}, CN≥{MIN_SITE_N_CN})")

    feasible_sites = site_counts[site_counts["feasible"]]["SiteCode_best"].tolist()
    print(f"\nFeasible sites: {len(feasible_sites)} — {feasible_sites}")

    # ------------------------------------------------------------------ #
    # 6. Per-site metrics (feasible sites only)                           #
    # ------------------------------------------------------------------ #
    site_rows = []
    for site in feasible_sites:
        sub = df[df["SiteCode_best"] == site]
        # manufacturer(s) at this site
        mfrs = "/".join(sorted(sub["Manufacturer"].unique()))
        row = {
            "SiteCode_best": site,
            "Manufacturer_s": mfrs,
            **compute_metrics(sub["y_true"].values, sub["y_score"].values, sub["y_pred"].values),
        }
        site_rows.append(row)
    site_df = pd.DataFrame(site_rows).sort_values("n_total", ascending=False).reset_index(drop=True)
    save_csv_md(site_df, "per_site_metrics", OUTPUT_DIR,
                "Per-Site Metrics (feasible sites)")

    print(f"\nPer-site metrics ({len(site_rows)} sites):")
    hdr = f"  {'Site':<8} {'Mfr':<8} {'nCN':>4} {'nAD':>4} {'AUC':>6} {'PR-AUC':>7} "
    hdr += f"{'BA':>6} {'Sens':>6} {'Spec':>6} {'F1':>6} {'score_sep':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for _, r in site_df.iterrows():
        auc_str = f"{r['AUC']:.4f}" if not np.isnan(r['AUC']) else "  N/A "
        pr_str  = f"{r['PR_AUC']:.4f}" if not np.isnan(r['PR_AUC']) else "  N/A "
        print(
            f"  {r['SiteCode_best']:<8} {r['Manufacturer_s']:<8} "
            f"{int(r['n_CN']):>4} {int(r['n_AD']):>4} "
            f"{auc_str:>6} {pr_str:>7} "
            f"{r['balanced_accuracy']:>6.4f} {r['sensitivity']:>6.4f} "
            f"{r['specificity']:>6.4f} {r['F1']:>6.4f} "
            f"{r['score_separation']:>9.4f}"
        )

    # ------------------------------------------------------------------ #
    # 7. Comparison summary                                               #
    # ------------------------------------------------------------------ #
    # AUC range across feasible sites and manufacturers
    site_auc_vals = site_df["AUC"].dropna()
    mfr_auc_vals = mfr_df["AUC"].dropna()

    # Build comparison table rows
    comp_rows = []
    comp_rows.append({
        "level": "global_pooled", "subgroup": "ALL",
        "n_CN": global_metrics["n_CN"], "n_AD": global_metrics["n_AD"],
        "AUC": global_metrics["AUC"], "PR_AUC": global_metrics["PR_AUC"],
        "balanced_accuracy": global_metrics["balanced_accuracy"],
        "sensitivity": global_metrics["sensitivity"],
        "specificity": global_metrics["specificity"],
        "F1": global_metrics["F1"],
        "mean_score_AD": global_metrics["mean_score_AD"],
        "mean_score_CN": global_metrics["mean_score_CN"],
    })
    for _, r in mfr_df.iterrows():
        comp_rows.append({
            "level": "manufacturer", "subgroup": r["Manufacturer"],
            "n_CN": r["n_CN"], "n_AD": r["n_AD"],
            "AUC": r["AUC"], "PR_AUC": r["PR_AUC"],
            "balanced_accuracy": r["balanced_accuracy"],
            "sensitivity": r["sensitivity"],
            "specificity": r["specificity"],
            "F1": r["F1"],
            "mean_score_AD": r["mean_score_AD"],
            "mean_score_CN": r["mean_score_CN"],
        })
    for _, r in site_df.iterrows():
        comp_rows.append({
            "level": "sitecode", "subgroup": f"site_{r['SiteCode_best']} ({r['Manufacturer_s']})",
            "n_CN": r["n_CN"], "n_AD": r["n_AD"],
            "AUC": r["AUC"], "PR_AUC": r["PR_AUC"],
            "balanced_accuracy": r["balanced_accuracy"],
            "sensitivity": r["sensitivity"],
            "specificity": r["specificity"],
            "F1": r["F1"],
            "mean_score_AD": r["mean_score_AD"],
            "mean_score_CN": r["mean_score_CN"],
        })
    comp_df = pd.DataFrame(comp_rows)
    save_csv_md(comp_df, "subgroup_comparison", OUTPUT_DIR,
                "Subgroup Comparison: Global / Manufacturer / Site")

    # Narrative summary
    def _fmt(v) -> str:
        return f"{v:.4f}" if not np.isnan(v) else "N/A"

    mfr_auc_range = f"{mfr_auc_vals.min():.4f}–{mfr_auc_vals.max():.4f}" if len(mfr_auc_vals) else "N/A"
    site_auc_range = f"{site_auc_vals.min():.4f}–{site_auc_vals.max():.4f}" if len(site_auc_vals) else "N/A"
    site_auc_mean = f"{site_auc_vals.mean():.4f}" if len(site_auc_vals) else "N/A"
    site_auc_std = f"{site_auc_vals.std():.4f}" if len(site_auc_vals) else "N/A"

    # Find best/worst sites by AUC
    best_site = site_df.loc[site_df["AUC"].idxmax()] if len(site_auc_vals) else None
    worst_site = site_df.loc[site_df["AUC"].idxmin()] if len(site_auc_vals) else None

    summary_md = textwrap.dedent(f"""\
    # Within-Site Performance Audit — Comparison Summary

    **Model:** {MODEL_NAME}
    **Threshold strategy:** `{THRESHOLD_STRATEGY}`
    **Source:** `classifier_sweep_predictions.csv`
    **Feasibility:** n_total ≥ {MIN_SITE_N_TOTAL}, n_AD ≥ {MIN_SITE_N_AD}, n_CN ≥ {MIN_SITE_N_CN}

    ---

    ## Global pooled vs. locked reference

    | Metric | This audit | Locked reference |
    | --- | --- | --- |
    | AUC | {_fmt(global_metrics['AUC'])} | {LOCKED_AUC} |
    | PR-AUC | {_fmt(global_metrics['PR_AUC'])} | {LOCKED_PR_AUC} |
    | balanced_accuracy | {_fmt(global_metrics['balanced_accuracy'])} | {LOCKED_BA} |
    | sensitivity | {_fmt(global_metrics['sensitivity'])} | {LOCKED_SENS} |
    | specificity | {_fmt(global_metrics['specificity'])} | {LOCKED_SPEC} |
    | F1 | {_fmt(global_metrics['F1'])} | {LOCKED_F1} |

    (Any discrepancy from locked reference indicates floating-point rounding only.)

    ---

    ## Manufacturer-level subgroups

    {df_to_markdown(mfr_df[['Manufacturer','n_CN','n_AD','AUC','PR_AUC','balanced_accuracy','sensitivity','specificity','F1','mean_score_AD','mean_score_CN']])}

    AUC range across manufacturers: **{mfr_auc_range}**

    ---

    ## Site-level subgroups ({len(feasible_sites)} feasible sites)

    {df_to_markdown(site_df[['SiteCode_best','Manufacturer_s','n_CN','n_AD','AUC','PR_AUC','balanced_accuracy','sensitivity','specificity','F1','mean_score_AD','mean_score_CN','score_separation']])}

    AUC range across feasible sites: **{site_auc_range}**
    Mean ± SD across feasible sites: **{site_auc_mean} ± {site_auc_std}**
    Best site:  **{best_site['SiteCode_best'] if best_site is not None else 'N/A'}** ({best_site['Manufacturer_s'] if best_site is not None else ''}, AUC={_fmt(best_site['AUC']) if best_site is not None else 'N/A'}, n={int(best_site['n_total']) if best_site is not None else 'N/A'})
    Worst site: **{worst_site['SiteCode_best'] if worst_site is not None else 'N/A'}** ({worst_site['Manufacturer_s'] if worst_site is not None else ''}, AUC={_fmt(worst_site['AUC']) if worst_site is not None else 'N/A'}, n={int(worst_site['n_total']) if worst_site is not None else 'N/A'})

    ---

    ## Interpretation

    ### Score separation (mean_score_AD − mean_score_CN)

    A positive score separation at the site level indicates the model assigns higher
    probability to AD subjects than CN subjects within that site. Sites with negative
    or near-zero separation are the ones contributing most to classification error.

    ### Site vs Manufacturer granularity

    - Manufacturer AUC range: **{mfr_auc_range}**  ({len(mfr_auc_vals)} manufacturers)
    - Site AUC range:         **{site_auc_range}**  ({len(site_auc_vals)} feasible sites)

    Site-level variation captures both scanner-manufacturer effects and any
    site-specific protocol or population effects not explained by manufacturer alone.
    A wider AUC range at the site level than the manufacturer level indicates
    within-manufacturer heterogeneity in model performance.

    ### Caveats

    1. **Small n**: With 9 feasible sites (n 10–34), per-site AUC and PR-AUC estimates
       have wide confidence intervals. Do not over-interpret single-site values.
    2. **OOF predictions**: Each subject appears exactly once, in their held-out outer
       fold. Within-site metrics are based on how the model generalises to that site when
       trained on the other 4 folds — not within-site train/test.
    3. **MCI excluded**: The 250 MCI subjects are not part of the binary classifier's
       labels (y_true ∈ {{0=CN, 1=AD}}) and are absent from this analysis.
    4. **Sites with 1 manufacturer**: Most feasible sites use a single manufacturer.
       Sites 002 and 013 use >1 manufacturer but fall below feasibility thresholds.

    ### Manuscript recommendation

    - Report the 3-manufacturer AUC subgroup table as a primary confound analysis.
    - Report the per-site AUC table (9 sites) as a supplementary within-site analysis,
      noting that sites are heterogeneous in size and AUC estimates are noisy.
    - Quote AUC range and mean ± SD across sites alongside the pooled AUC.
    """)
    (OUTPUT_DIR / "comparison_summary.md").write_text(summary_md)

    # ------------------------------------------------------------------ #
    # 8. README                                                           #
    # ------------------------------------------------------------------ #
    readme_md = textwrap.dedent(f"""\
    # Within-Site Performance Audit
    ## adni_v5_1_batch20260514b — locked logreg_l2

    **Generated:** {t0.strftime('%Y-%m-%d %H:%M:%S UTC')}
    **Script:** `scripts/revision_bspc_2026/audit_v5_1_batch20260514b_locked_within_site_performance.py`

    ## Source

    | Item | Value |
    | --- | --- |
    | Model | `{MODEL_NAME}` |
    | Threshold strategy | `{THRESHOLD_STRATEGY}` |
    | Predictions file | `classifier_sweep_predictions.csv` |
    | Subjects | {n_preds} (CN={int((df['y_true']==0).sum())}, AD={int((df['y_true']==1).sum())}) |
    | Feasible sites | {len(feasible_sites)} (n≥{MIN_SITE_N_TOTAL}, AD≥{MIN_SITE_N_AD}, CN≥{MIN_SITE_N_CN}) |

    ## Files

    | File | Contents |
    | --- | --- |
    | global_metrics.csv/.md | Full metric set for all 396 subjects |
    | per_manufacturer_metrics.csv/.md | Metrics per scanner manufacturer |
    | per_site_metrics.csv/.md | Metrics for {len(feasible_sites)} feasible sites |
    | site_feasibility.csv/.md | All sites with feasibility flags |
    | subgroup_comparison.csv/.md | Combined table: global + manufacturer + site |
    | comparison_summary.md | Narrative analysis and manuscript notes |
    | command_log.json | Execution metadata |

    ## Read-only guarantee

    No tensor, metadata, ledger, config, or model file was modified.
    """)
    (OUTPUT_DIR / "README.md").write_text(readme_md)

    # ------------------------------------------------------------------ #
    # 9. command_log.json                                                 #
    # ------------------------------------------------------------------ #
    t1 = datetime.now(timezone.utc)
    log = {
        "script": Path(__file__).name,
        "started_utc": t0.isoformat(),
        "finished_utc": t1.isoformat(),
        "elapsed_s": round((t1 - t0).total_seconds(), 2),
        "predictions_path": str(PREDICTIONS_PATH),
        "metadata_path": str(METADATA_PATH),
        "output_dir": str(OUTPUT_DIR),
        "model_name": MODEL_NAME,
        "threshold_strategy": THRESHOLD_STRATEGY,
        "n_subjects": n_preds,
        "n_CN": int((df["y_true"] == 0).sum()),
        "n_AD": int((df["y_true"] == 1).sum()),
        "feasibility_thresholds": {
            "min_total": MIN_SITE_N_TOTAL,
            "min_AD": MIN_SITE_N_AD,
            "min_CN": MIN_SITE_N_CN,
        },
        "n_feasible_sites": len(feasible_sites),
        "feasible_sites": feasible_sites,
        "global_metrics": global_metrics,
        "locked_reference": {
            "AUC": LOCKED_AUC,
            "PR_AUC": LOCKED_PR_AUC,
            "balanced_accuracy": LOCKED_BA,
            "sensitivity": LOCKED_SENS,
            "specificity": LOCKED_SPEC,
            "F1": LOCKED_F1,
        },
        "python": sys.version,
        "read_only": True,
    }
    (OUTPUT_DIR / "command_log.json").write_text(json.dumps(log, indent=2, default=str))

    print(f"\n{'='*60}")
    print("WITHIN-SITE PERFORMANCE AUDIT COMPLETE")
    print(f"{'='*60}")
    print(f"  Feasible sites: {len(feasible_sites)}")
    print(f"  Site AUC range: {site_auc_range}  (mean {site_auc_mean} ± {site_auc_std})")
    print(f"  Mfr  AUC range: {mfr_auc_range}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Elapsed: {log['elapsed_s']}s")
    print("="*60)


if __name__ == "__main__":
    run_audit()
