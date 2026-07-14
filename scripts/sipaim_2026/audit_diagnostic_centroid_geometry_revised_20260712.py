#!/usr/bin/env python3
"""Diagnostic-centroid geometry, reformulated per the 2026-07-12 blocker-resolution spec.

Read-only. No training, no calibration, no OASIS-based fitting.

The axis is a **fold-local ADNI centroid-derived discriminant axis** (AD
train/dev centroid minus CN train/dev centroid, per fold, in the frozen
locked latent_mu space) -- explicitly NOT the classifier's decision
boundary/weight vector. Coordinates are never pooled across the five
independently-trained fold encoders: every centroid, axis, and projection is
computed fold-locally, and only the resulting *scalar* projections are
averaged across folds (an ensemble of scalars, matching the project's
existing score-ensembling convention), never raw 384-dim vectors.

The overall ADNI reference point used to center each fold's axis is a
class-prevalence-weighted centroid (0.5*CN + 0.5*AD), matching OASIS
current-180's exact 50/50 CN/AD composition -- not ADNI's own (imbalanced)
raw sample composition.

Four arms: locked_frozen_transfer (primary), previous_adni_fitted_siemens_combat
(primary comparator), external_dataset_combat_adni_reference (primary
comparator), external_dataset_combat_no_reference (secondary only, per
instruction).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path("/home/diego/proyectos/vae_AD")
OUT_DIR = PROJECT_ROOT / "results/sipaim_2026/final_blocker_resolution_20260712"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RUN_DIR = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026/recover035_latent384_beta3p75_T80_h10000_p560_full5x5")

ARM_SOURCES = {
    "locked_frozen_transfer": dict(
        path=PROJECT_ROOT / "results/revision_bspc_2026/promoted_latent384_oasis_vs_adni_latent_distance_audit_20260604/oasis_fold_latent_mu_runwise164.csv",
        subject_col="SubjectID", y_col="y", arm_filter=None, role="primary",
    ),
    "previous_adni_fitted_siemens_combat": dict(
        path=PROJECT_ROOT / "results/revision_bspc_2026/post_revision_exploratory_20260630/foldcombat_cleanrepro_final_audit_20260706/cleanrepro_oasis_fold_latent_mu.csv",
        subject_col="SubjectID", y_col="y", arm_filter=None, role="primary",
    ),
    "external_dataset_combat_adni_reference": dict(
        path=PROJECT_ROOT / "results/revision_bspc_2026/post_revision_exploratory_20260630/oasis_external_batch_combat_20260710/oasis_external_batch_combat_latents.csv",
        subject_col="SubjectID", y_col="y", arm_filter="external_dataset_combat_adni_reference", role="primary",
    ),
    "external_dataset_combat_no_reference": dict(
        path=PROJECT_ROOT / "results/revision_bspc_2026/post_revision_exploratory_20260630/oasis_external_batch_combat_20260710/oasis_external_batch_combat_latents.csv",
        subject_col="SubjectID", y_col="y", arm_filter="external_dataset_combat_no_reference", role="secondary",
    ),
}

BOOT_N = 10000
BOOT_SEED = 20260710
OASIS_CN_PREVALENCE = 0.5  # current-180 is exactly 90 CN / 90 AD
OASIS_AD_PREVALENCE = 0.5


def log(msg: str) -> None:
    print(msg, flush=True)


def load_mu(path: Path, mu_prefix: str = "mu_") -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)
    mu_cols = sorted([c for c in df.columns if c.startswith(mu_prefix)], key=lambda c: int(c.split("_")[1]))
    return df, mu_cols


# ── 1. Fold-local ADNI CN/AD train/dev centroids (locked encoder; same for all arms) ──
adni_fold = {}
for fold in range(1, 6):
    path = RUN_DIR / "classifier_only_readout/latent_cache" / f"fold_{fold}_trainDev_latent_mu.csv"
    df, mu_cols = load_mu(path)
    df["ResearchGroup_Mapped"] = df["ResearchGroup_Mapped"].astype(str).str.upper().str.strip()
    cn = df[df["ResearchGroup_Mapped"] == "CN"][mu_cols].to_numpy()
    ad = df[df["ResearchGroup_Mapped"] == "AD"][mu_cols].to_numpy()
    cn_centroid = cn.mean(axis=0)
    ad_centroid = ad.mean(axis=0)
    axis = ad_centroid - cn_centroid
    axis_unit = axis / np.linalg.norm(axis)
    overall_centroid_prevalence_weighted = OASIS_CN_PREVALENCE * cn_centroid + OASIS_AD_PREVALENCE * ad_centroid

    def proj(x, oc=overall_centroid_prevalence_weighted, ax=axis_unit):
        return (x - oc) @ ax

    adni_cn_proj = proj(cn)
    adni_ad_proj = proj(ad)
    adni_fold[fold] = dict(
        mu_cols=mu_cols, cn_centroid=cn_centroid, ad_centroid=ad_centroid, axis_unit=axis_unit,
        overall_centroid=overall_centroid_prevalence_weighted, proj_fn=proj,
        n_cn=len(cn), n_ad=len(ad),
        adni_cn_proj_mean=float(adni_cn_proj.mean()), adni_ad_proj_mean=float(adni_ad_proj.mean()),
        adni_cn_proj_std=float(adni_cn_proj.std(ddof=1)), adni_ad_proj_std=float(adni_ad_proj.std(ddof=1)),
        adni_ad_cn_separation_at_source=float(adni_ad_proj.mean() - adni_cn_proj.mean()),
    )
    log(f"fold {fold}: ADNI n_cn={len(cn)} n_ad={len(ad)}, "
        f"at-source AD-CN separation={adni_fold[fold]['adni_ad_cn_separation_at_source']:.4f}")

# ── 2. Per-arm, per-fold OASIS projections + metrics ────────────────────────
def bootstrap_metric_ci(cn_vals, ad_vals, rng, n_boot=BOOT_N):
    """Bootstrap (resample CN and AD subjects separately, preserving class sizes) for
    marginal shift, CN shift are computed outside; here we bootstrap AD-CN separation,
    Cohen's d, and projected AUC, which need both groups jointly."""
    n_cn, n_ad = len(cn_vals), len(ad_vals)
    seps, ds, aucs, margins = [], [], [], []
    for _ in range(n_boot):
        bcn = rng.choice(cn_vals, size=n_cn, replace=True)
        bad = rng.choice(ad_vals, size=n_ad, replace=True)
        sep = bad.mean() - bcn.mean()
        pooled_std = np.sqrt(((n_cn - 1) * bcn.var(ddof=1) + (n_ad - 1) * bad.var(ddof=1)) / (n_cn + n_ad - 2))
        d = sep / pooled_std if pooled_std > 0 else np.nan
        y = np.concatenate([np.zeros(n_cn), np.ones(n_ad)])
        s = np.concatenate([bcn, bad])
        auc = roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan
        margin = min(bad.mean(), -bcn.mean())
        seps.append(sep); ds.append(d); aucs.append(auc); margins.append(margin)
    return dict(
        sep_mean=np.mean(seps), sep_ci=(np.percentile(seps, 2.5), np.percentile(seps, 97.5)),
        d_mean=np.nanmean(ds), d_ci=(np.nanpercentile(ds, 2.5), np.nanpercentile(ds, 97.5)),
        auc_mean=np.mean(aucs), auc_ci=(np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)),
        margin_mean=np.mean(margins), margin_ci=(np.percentile(margins, 2.5), np.percentile(margins, 97.5)),
    )


fold_rows = []
ensemble_subject_proj = {}  # arm -> {SubjectID: [proj per fold]}
for arm, spec in ARM_SOURCES.items():
    df, mu_cols_arm = load_mu(spec["path"])
    if spec["arm_filter"] is not None:
        df = df[df["arm"] == spec["arm_filter"]]
    df["y"] = pd.to_numeric(df[spec["y_col"]], errors="coerce")
    ensemble_subject_proj.setdefault(arm, {})

    for fold in range(1, 6):
        dff = df[df["fold"] == fold]
        assert len(dff) == 180, f"{arm} fold {fold}: expected 180 OASIS subjects, got {len(dff)}"
        X = dff[mu_cols_arm].to_numpy()
        proj_fn = adni_fold[fold]["proj_fn"]
        proj_vals = proj_fn(X)
        subj_ids = dff[spec["subject_col"]].to_numpy()
        for sid, p in zip(subj_ids, proj_vals):
            ensemble_subject_proj[arm].setdefault(sid, []).append(p)

        y = dff["y"].to_numpy()
        cn_vals = proj_vals[y == 0]
        ad_vals = proj_vals[y == 1]
        oasis_cn_proj_mean = float(cn_vals.mean())
        oasis_ad_proj_mean = float(ad_vals.mean())
        marginal_shift = OASIS_CN_PREVALENCE * oasis_cn_proj_mean + OASIS_AD_PREVALENCE * oasis_ad_proj_mean
        cn_shift = oasis_cn_proj_mean - adni_fold[fold]["adni_cn_proj_mean"]
        ad_shift = oasis_ad_proj_mean - adni_fold[fold]["adni_ad_proj_mean"]
        ad_cn_sep = oasis_ad_proj_mean - oasis_cn_proj_mean
        pooled_std = np.sqrt(((len(cn_vals) - 1) * cn_vals.var(ddof=1) + (len(ad_vals) - 1) * ad_vals.var(ddof=1))
                              / (len(cn_vals) + len(ad_vals) - 2))
        cohens_d = ad_cn_sep / pooled_std if pooled_std > 0 else np.nan
        proj_auc = roc_auc_score(y, proj_vals)
        clinical_alignment_margin = min(oasis_ad_proj_mean, -oasis_cn_proj_mean)

        fold_rows.append(dict(
            arm=arm, role=spec["role"], fold=fold, n_cn=len(cn_vals), n_ad=len(ad_vals),
            adni_cn_proj_mean=adni_fold[fold]["adni_cn_proj_mean"], adni_ad_proj_mean=adni_fold[fold]["adni_ad_proj_mean"],
            adni_ad_cn_separation_at_source=adni_fold[fold]["adni_ad_cn_separation_at_source"],
            oasis_cn_proj_mean=oasis_cn_proj_mean, oasis_ad_proj_mean=oasis_ad_proj_mean,
            marginal_shift=marginal_shift, cn_shift=cn_shift, ad_shift=ad_shift,
            ad_cn_projected_separation=ad_cn_sep, cohens_d=cohens_d, projected_auc=proj_auc,
            clinical_alignment_margin=clinical_alignment_margin,
        ))
    log(f"arm {arm}: fold-level metrics computed for 5 folds")

fold_df = pd.DataFrame(fold_rows)
fold_df.to_csv(OUT_DIR / "_diagnostic_centroid_geometry_by_fold.csv", index=False)

# ── 3. Paired fold deltas (arm minus locked, per fold) ──────────────────────
locked_by_fold = fold_df[fold_df.arm == "locked_frozen_transfer"].set_index("fold")
delta_rows = []
metric_cols = ["marginal_shift", "cn_shift", "ad_shift", "ad_cn_projected_separation", "cohens_d",
               "projected_auc", "clinical_alignment_margin"]
for arm in ARM_SOURCES:
    if arm == "locked_frozen_transfer":
        continue
    arm_by_fold = fold_df[fold_df.arm == arm].set_index("fold")
    for metric in metric_cols:
        deltas = (arm_by_fold[metric] - locked_by_fold[metric]).to_numpy()
        delta_rows.append(dict(
            arm=arm, metric=metric, n_folds=len(deltas), mean_delta=float(np.mean(deltas)),
            sd_delta=float(np.std(deltas, ddof=1)), min_delta=float(np.min(deltas)), max_delta=float(np.max(deltas)),
            n_folds_arm_lower_than_locked=int((deltas < 0).sum()), n_folds_arm_higher_than_locked=int((deltas > 0).sum()),
            per_fold_deltas=json.dumps([float(d) for d in deltas]),
        ))
delta_df = pd.DataFrame(delta_rows)
delta_df.to_csv(OUT_DIR / "_paired_fold_deltas_vs_locked.csv", index=False)
log("\nPaired fold deltas (arm minus locked):")
log(delta_df[["arm", "metric", "mean_delta", "n_folds_arm_lower_than_locked", "n_folds_arm_higher_than_locked"]].to_string(index=False))

# ── 4. Ensemble (mean scalar projection across folds per subject) + subject-bootstrap CIs ──
ensemble_rows = []
for arm, spec in ARM_SOURCES.items():
    subj_proj = ensemble_subject_proj[arm]
    subj_ids = list(subj_proj.keys())
    ens_proj = np.array([np.mean(subj_proj[s]) for s in subj_ids])
    # y label: reload once for mapping (use fold-1 slice of df, should be same y for all folds per subject)
    df, mu_cols_arm = load_mu(spec["path"])
    if spec["arm_filter"] is not None:
        df = df[df["arm"] == spec["arm_filter"]]
    y_map = df.drop_duplicates(subset=spec["subject_col"]).set_index(spec["subject_col"])["y"]
    y = np.array([int(y_map.loc[s]) for s in subj_ids])

    cn_vals = ens_proj[y == 0]
    ad_vals = ens_proj[y == 1]
    oasis_cn_proj_mean = float(cn_vals.mean())
    oasis_ad_proj_mean = float(ad_vals.mean())
    adni_cn_mean_ens = float(np.mean([adni_fold[f]["adni_cn_proj_mean"] for f in range(1, 6)]))
    adni_ad_mean_ens = float(np.mean([adni_fold[f]["adni_ad_proj_mean"] for f in range(1, 6)]))
    marginal_shift = OASIS_CN_PREVALENCE * oasis_cn_proj_mean + OASIS_AD_PREVALENCE * oasis_ad_proj_mean
    cn_shift = oasis_cn_proj_mean - adni_cn_mean_ens
    ad_shift = oasis_ad_proj_mean - adni_ad_mean_ens
    ad_cn_sep = oasis_ad_proj_mean - oasis_cn_proj_mean
    pooled_std = np.sqrt(((len(cn_vals) - 1) * cn_vals.var(ddof=1) + (len(ad_vals) - 1) * ad_vals.var(ddof=1))
                          / (len(cn_vals) + len(ad_vals) - 2))
    cohens_d = ad_cn_sep / pooled_std if pooled_std > 0 else np.nan
    proj_auc = roc_auc_score(y, ens_proj)
    clinical_alignment_margin = min(oasis_ad_proj_mean, -oasis_cn_proj_mean)

    rng = np.random.default_rng(BOOT_SEED)
    boot = bootstrap_metric_ci(cn_vals, ad_vals, rng)
    # marginal/CN/AD shift bootstrap (simpler, single-group resampling)
    rng2 = np.random.default_rng(BOOT_SEED)
    marg_boot, cnshift_boot, adshift_boot = [], [], []
    for _ in range(BOOT_N):
        bcn = rng2.choice(cn_vals, size=len(cn_vals), replace=True)
        bad = rng2.choice(ad_vals, size=len(ad_vals), replace=True)
        marg_boot.append(OASIS_CN_PREVALENCE * bcn.mean() + OASIS_AD_PREVALENCE * bad.mean())
        cnshift_boot.append(bcn.mean() - adni_cn_mean_ens)
        adshift_boot.append(bad.mean() - adni_ad_mean_ens)

    ensemble_rows.append(dict(
        arm=arm, role=spec["role"], n_cn=len(cn_vals), n_ad=len(ad_vals),
        adni_cn_proj_mean_ensembled=adni_cn_mean_ens, adni_ad_proj_mean_ensembled=adni_ad_mean_ens,
        oasis_cn_proj_mean=oasis_cn_proj_mean, oasis_ad_proj_mean=oasis_ad_proj_mean,
        marginal_shift=marginal_shift, marginal_shift_ci_low=np.percentile(marg_boot, 2.5), marginal_shift_ci_high=np.percentile(marg_boot, 97.5),
        cn_shift=cn_shift, cn_shift_ci_low=np.percentile(cnshift_boot, 2.5), cn_shift_ci_high=np.percentile(cnshift_boot, 97.5),
        ad_shift=ad_shift, ad_shift_ci_low=np.percentile(adshift_boot, 2.5), ad_shift_ci_high=np.percentile(adshift_boot, 97.5),
        ad_cn_projected_separation=ad_cn_sep, ad_cn_sep_ci_low=boot["sep_ci"][0], ad_cn_sep_ci_high=boot["sep_ci"][1],
        cohens_d=cohens_d, cohens_d_ci_low=boot["d_ci"][0], cohens_d_ci_high=boot["d_ci"][1],
        projected_auc=proj_auc, projected_auc_ci_low=boot["auc_ci"][0], projected_auc_ci_high=boot["auc_ci"][1],
        clinical_alignment_margin=clinical_alignment_margin,
        clinical_alignment_margin_ci_low=boot["margin_ci"][0], clinical_alignment_margin_ci_high=boot["margin_ci"][1],
    ))
    log(f"arm {arm}: ensemble AUC={proj_auc:.4f} [{boot['auc_ci'][0]:.4f},{boot['auc_ci'][1]:.4f}], "
        f"Cohen's d={cohens_d:.4f}, margin={clinical_alignment_margin:.4f}")

ensemble_df = pd.DataFrame(ensemble_rows)
ensemble_df.to_csv(OUT_DIR / "diagnostic_centroid_geometry_revised.csv", index=False)
log(f"\nWrote {OUT_DIR / 'diagnostic_centroid_geometry_revised.csv'}")

with open(OUT_DIR / "_adni_fold_axis_reference.json", "w") as f:
    json.dump({str(f): {k: v for k, v in d.items() if k not in ("proj_fn", "cn_centroid", "ad_centroid", "axis_unit", "overall_centroid")}
               for f, d in adni_fold.items()}, f, indent=2)

log("Task 2 computation done.")
