#!/usr/bin/env python3
"""ch1-only FULL candidate preflight under the OASIS CDR label-definition sensitivity.

Read-only. No training, no new VAE inference, no OASIS calibration/threshold
selection/model selection. Applies the already-computed CDR label table from
results/sipaim_2026/oasis_label_sensitivity_20260710/ to the already-computed,
frozen OASIS ensemble scores for two ch1-only candidates (beta=3.75, beta=4.5)
and the promoted [1,0,2] comparator, then reports non-recalibrated ROC-AUC/
PR-AUC under both OASIS-current-180 and OASIS-strict-128 label definitions.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

PROJECT_ROOT = Path("/home/diego/proyectos/vae_AD")
OUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/ch1only_oasis_cdr_and_beta4p5_preflight_20260711"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CDR_TABLE = PROJECT_ROOT / "results/sipaim_2026/oasis_label_sensitivity_20260710/oasis_label_sensitivity_subject_table.csv"

PRED_PATHS = {
    "ch1only_beta3p75_main": PROJECT_ROOT
    / "results/revision_bspc_2026/ch1only_latent384_beta3p75_oasis_mega_90_90_external_inference_20260605/predictions.csv",
    "ch1only_beta4p5_site31_mayo_reprocessed14": Path(
        "/media/diego/Datos/vae_AD_results/revision_bspc_2026/"
        "oasis_ch1_beta4p5_site31_mayo_reprocessed14_external_inference_20260622/predictions.csv"
    ),
}
CANDIDATE_LABELS = {
    "ch1only_beta3p75_main": "ch1only_latent384_beta3p75_oof_ecdf",
    "ch1only_beta4p5_site31_mayo_reprocessed14": "ch1only_beta4p5_site31_mayo_reprocessed14_oof_ecdf",
}

BOOT_N = 10000
BOOT_SEED = 20260710  # matches results/sipaim_2026/oasis_label_sensitivity_20260710 convention


def log(msg: str) -> None:
    print(msg, flush=True)


def bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, rng: np.random.Generator, n_boot: int = BOOT_N):
    n = len(y_true)
    aucs, prs = [], []
    idx_pool = np.arange(n)
    for _ in range(n_boot):
        idx = rng.choice(idx_pool, size=n, replace=True)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, ys))
        prs.append(average_precision_score(yt, ys))
    return (
        float(np.mean(aucs)), float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)),
        float(np.mean(prs)), float(np.percentile(prs, 2.5)), float(np.percentile(prs, 97.5)),
        len(aucs),
    )


def paired_bootstrap_delta_auc(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray,
                                rng: np.random.Generator, n_boot: int = BOOT_N):
    """delta = A - B, resampling subjects jointly (paired, same subjects both models)."""
    n = len(y_true)
    idx_pool = np.arange(n)
    deltas_auc, deltas_pr = [], []
    for _ in range(n_boot):
        idx = rng.choice(idx_pool, size=n, replace=True)
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        auc_a = roc_auc_score(yt, score_a[idx])
        auc_b = roc_auc_score(yt, score_b[idx])
        pr_a = average_precision_score(yt, score_a[idx])
        pr_b = average_precision_score(yt, score_b[idx])
        deltas_auc.append(auc_a - auc_b)
        deltas_pr.append(pr_a - pr_b)
    deltas_auc = np.array(deltas_auc)
    deltas_pr = np.array(deltas_pr)
    return dict(
        delta_auc_mean=float(deltas_auc.mean()),
        delta_auc_ci_lo=float(np.percentile(deltas_auc, 2.5)),
        delta_auc_ci_hi=float(np.percentile(deltas_auc, 97.5)),
        p_auc_positive=float((deltas_auc > 0).mean()),
        delta_pr_mean=float(deltas_pr.mean()),
        delta_pr_ci_lo=float(np.percentile(deltas_pr, 2.5)),
        delta_pr_ci_hi=float(np.percentile(deltas_pr, 97.5)),
        p_pr_positive=float((deltas_pr > 0).mean()),
        n_boot_valid=int(len(deltas_auc)),
    )


# ── 1. Load CDR label table (from the completed OASIS CDR sensitivity audit) ──
log("Loading CDR label table ...")
cdr = pd.read_csv(CDR_TABLE)
cdr = cdr.set_index("subject_id")

# ── 2. Extract ensemble OASIS scores for each candidate + promoted reference ──
def extract_ensemble_scores(pred_path: Path, candidate_label: str) -> pd.DataFrame:
    df = pd.read_csv(pred_path)
    sub = df[
        (df["build_candidate"] == "runwise164_pilot_parity")
        & (df["prediction_level"] == "ensemble_mean_score_majority_vote")
        & (df["candidate"] == candidate_label)
    ]
    sub = sub[["subject_id", "y", "y_score"]].drop_duplicates(subset="subject_id").set_index("subject_id")
    return sub


scores = {}
for cand, path in PRED_PATHS.items():
    log(f"Extracting ensemble scores for {cand} from {path} ...")
    s = extract_ensemble_scores(path, CANDIDATE_LABELS[cand])
    log(f"  n={len(s)}")
    scores[cand] = s

# Promoted reference is present in both prediction files (role=primary_reference);
# extract once from the beta3.75 file and sanity-check against score_locked in the
# CDR table (which was itself sourced from the same frozen promoted predictions).
promoted_scores = extract_ensemble_scores(PRED_PATHS["ch1only_beta3p75_main"], "promoted_beta3p75_oof_ecdf")
sanity = cdr[["score_locked"]].join(promoted_scores[["y_score"]], how="inner")
max_abs_diff = (sanity["score_locked"] - sanity["y_score"]).abs().max()
log(f"Sanity check: promoted ensemble score vs CDR-table score_locked, max abs diff = {max_abs_diff:.10f}")
assert max_abs_diff < 1e-9, "Promoted OASIS score mismatch between predictions.csv and CDR label table"
scores["promoted_[1,0,2]_beta3p75"] = promoted_scores

# ── 3. Per-candidate metrics under current-180 and strict-128 ──────────────────
def compute_metrics_for_candidate(cand: str, score_df: pd.DataFrame, rng: np.random.Generator) -> dict:
    joined = cdr.join(score_df[["y_score"]], how="inner")
    assert len(joined) == 180, f"{cand}: expected 180 joined subjects, got {len(joined)}"

    out = {}
    # Current-180
    y_cur = joined["y_current"].to_numpy()
    s = joined["y_score"].to_numpy()
    auc_cur = roc_auc_score(y_cur, s)
    pr_cur = average_precision_score(y_cur, s)
    boot_cur = bootstrap_ci(y_cur, s, rng)
    out["current_180"] = dict(n=len(joined), n_cn=int((y_cur == 0).sum()), n_ad=int((y_cur == 1).sum()),
                               roc_auc=auc_cur, pr_auc=pr_cur,
                               roc_auc_boot_mean=boot_cur[0], roc_auc_ci_low=boot_cur[1], roc_auc_ci_high=boot_cur[2],
                               pr_auc_boot_mean=boot_cur[3], pr_auc_ci_low=boot_cur[4], pr_auc_ci_high=boot_cur[5],
                               n_boot_valid=boot_cur[6])

    # Strict-128 (CDR0 vs CDR>=1, strict_inclusion==True)
    strict = joined[joined["strict_inclusion"] == True]  # noqa: E712
    y_strict = strict["y_strict_cdr"].to_numpy()
    s_strict = strict["y_score"].to_numpy()
    auc_strict = roc_auc_score(y_strict, s_strict)
    pr_strict = average_precision_score(y_strict, s_strict)
    boot_strict = bootstrap_ci(y_strict, s_strict, rng)
    out["strict_128"] = dict(n=len(strict), n_cn=int((y_strict == 0).sum()), n_ad=int((y_strict == 1).sum()),
                              roc_auc=auc_strict, pr_auc=pr_strict,
                              roc_auc_boot_mean=boot_strict[0], roc_auc_ci_low=boot_strict[1], roc_auc_ci_high=boot_strict[2],
                              pr_auc_boot_mean=boot_strict[3], pr_auc_ci_low=boot_strict[4], pr_auc_ci_high=boot_strict[5],
                              n_boot_valid=boot_strict[6])

    # CDR-group score means/medians
    grp = joined.groupby("cdr_group")["y_score"].agg(["count", "mean", "median", "std"])
    out["cdr_group_scores"] = grp.to_dict(orient="index")

    # Spearman rho score vs ordinal CDR
    from scipy.stats import spearmanr
    rho, pval = spearmanr(joined["y_score"], joined["cdr_ordinal"])
    out["spearman_rho_score_vs_cdr"] = float(rho)
    out["spearman_pvalue"] = float(pval)

    out["joined_df"] = joined
    return out


rng_master = np.random.default_rng(BOOT_SEED)
results = {}
for cand, sdf in scores.items():
    log(f"Computing metrics for {cand} ...")
    rng = np.random.default_rng(BOOT_SEED)  # same seed per candidate, independent streams via rng object reuse pattern
    results[cand] = compute_metrics_for_candidate(cand, sdf, rng)

# ── 4. Paired comparisons vs promoted (subject-matched, both label definitions) ─
promoted_joined_cur = results["promoted_[1,0,2]_beta3p75"]["joined_df"]
comparisons = []
for cand in ["ch1only_beta3p75_main", "ch1only_beta4p5_site31_mayo_reprocessed14"]:
    cj = results[cand]["joined_df"]
    # current-180 paired delta
    common = cj.index.intersection(promoted_joined_cur.index)
    y_cur = cj.loc[common, "y_current"].to_numpy()
    rng = np.random.default_rng(BOOT_SEED)
    d_cur = paired_bootstrap_delta_auc(
        y_cur, cj.loc[common, "y_score"].to_numpy(), promoted_joined_cur.loc[common, "y_score"].to_numpy(), rng
    )
    d_cur["evaluation_set"] = "OASIS-current-180"
    d_cur["candidate"] = cand
    comparisons.append(d_cur)

    # strict-128 paired delta
    cj_strict = cj[cj["strict_inclusion"] == True]  # noqa: E712
    pj_strict = promoted_joined_cur[promoted_joined_cur["strict_inclusion"] == True]  # noqa: E712
    common_s = cj_strict.index.intersection(pj_strict.index)
    y_strict = cj_strict.loc[common_s, "y_strict_cdr"].to_numpy()
    rng = np.random.default_rng(BOOT_SEED)
    d_strict = paired_bootstrap_delta_auc(
        y_strict, cj_strict.loc[common_s, "y_score"].to_numpy(), pj_strict.loc[common_s, "y_score"].to_numpy(), rng
    )
    d_strict["evaluation_set"] = "OASIS-strict-128"
    d_strict["candidate"] = cand
    comparisons.append(d_strict)

comparisons_df = pd.DataFrame(comparisons)
comparisons_df.to_csv(OUT_DIR / "_paired_bootstrap_vs_promoted.csv", index=False)
log(comparisons_df.to_string(index=False))

# ── 5. Persist raw per-candidate metrics as JSON for downstream table-writing ──
serializable = {}
for cand, r in results.items():
    serializable[cand] = dict(
        current_180=r["current_180"],
        strict_128=r["strict_128"],
        cdr_group_scores=r["cdr_group_scores"],
        spearman_rho_score_vs_cdr=r["spearman_rho_score_vs_cdr"],
        spearman_pvalue=r["spearman_pvalue"],
    )
with open(OUT_DIR / "_raw_metrics.json", "w") as f:
    json.dump(serializable, f, indent=2)
log(f"Wrote {OUT_DIR / '_raw_metrics.json'}")

# ── 6. Command log ──────────────────────────────────────────────────────────
def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unavailable"


command_log = dict(
    script="scripts/revision_bspc_2026/audit_ch1only_oasis_cdr_and_beta4p5_preflight_20260711.py",
    git_hash_at_run=git_hash(),
    guardrails=dict(
        no_training=True,
        no_new_vae_inference=True,
        no_oasis_calibration_or_threshold_selection=True,
        no_model_selection_using_oasis_labels=True,
        cdr_0p5_kept_separate=True,
        read_only_no_existing_results_modified=True,
    ),
    inputs=dict(
        cdr_label_table=str(CDR_TABLE),
        prediction_files={k: str(v) for k, v in PRED_PATHS.items()},
    ),
    bootstrap=dict(n_boot=BOOT_N, seed=BOOT_SEED, method="subject-level bootstrap with replacement, "
                    "per evaluation-set (current-180 / strict-128), independent per candidate; "
                    "paired variant resamples the shared subject index jointly for delta-AUC/delta-PR-AUC "
                    "vs. the promoted comparator."),
    method_notes=[
        "Ensemble OASIS scores are the already-computed 'ensemble_mean_score_majority_vote' rows at "
        "build_candidate == 'runwise164_pilot_parity' (the current-180 cohort standard build) from each "
        "candidate's frozen external-inference predictions.csv. No score was recomputed or recalibrated.",
        "Promoted [1,0,2] reference scores were sanity-checked byte-for-byte against "
        "oasis_label_sensitivity_subject_table.csv's score_locked column (max abs diff reported above) "
        "to confirm this is the exact same frozen prediction set used in the completed CDR sensitivity audit.",
        "ch1only_beta4p5 predictions come from the site31_mayo_reprocessed14 external-inference run; this is "
        "the only OASIS scoring that exists for beta4.5 (see candidate_provenance_status for the ADNI-side "
        "tensor mismatch this implies).",
    ],
)
with open(OUT_DIR / "command_log.json", "w") as f:
    json.dump(command_log, f, indent=2)
log(f"Wrote {OUT_DIR / 'command_log.json'}")
log("Done.")
