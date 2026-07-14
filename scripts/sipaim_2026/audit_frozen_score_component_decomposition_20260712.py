#!/usr/bin/env python3
"""Frozen-score component audit: decompose the promoted [1,0,2] OASIS logit
into latent, Age, and Sex components, and test CDR association for the
latent-only component.

Read-only. No OASIS-based fitting, no OASIS-based model/threshold selection,
no calibration on OASIS. The Stage-B classifier itself is reconstructed
ADNI-only, deterministically, using the exact fixed recipe (hyperparameter
grid, fixed random seeds, fixed inner-CV splits) already established as
canonical in scripts/revision_bspc_2026/score_oasis_mega_90_90_external_inference_model_panel_20260604.py,
because the fitted Stage-B pipeline/OOF-ECDF transformer objects were never
persisted to disk (see command_log.json's "missing_artifact" note -- this
audit's key finding for Task 3). This reconstruction touches ADNI train/dev
data only, uses zero OASIS data, and makes zero new hyperparameter choices
beyond what is already fixed in the original pipeline -- it is a
reconstruction, not a retraining or a new model-selection step.

Separately (and NOT used for the primary CDR-association claim), this script
also verifies that the actually-persisted Stage-A joblib pipelines
(classifier_logreg_final_pipeline_fold_N.joblib) do NOT reproduce the
canonical OASIS y_score_raw values -- confirming that a zero-fit decomposition
using only currently-persisted artifacts is not possible.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, pearsonr
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path("/home/diego/proyectos/vae_AD")
OUT_DIR = PROJECT_ROOT / "results/sipaim_2026/final_blocker_resolution_20260712"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SCRIPT_DIR = PROJECT_ROOT / "scripts/revision_bspc_2026"
sys.path.insert(0, str(SCRIPT_DIR))
from run_v5_1_batch20260514b_mfrsplit_3840_classifier_only_sweep import make_ohe, score_1d  # noqa: E402

RUN_DIR = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026/recover035_latent384_beta3p75_T80_h10000_p560_full5x5")
OASIS_LOCKED_LATENTS = PROJECT_ROOT / "results/revision_bspc_2026/promoted_latent384_oasis_vs_adni_latent_distance_audit_20260604/oasis_fold_latent_mu_runwise164.csv"
CANON_PRED = PROJECT_ROOT / "results/revision_bspc_2026/ch1only_latent384_beta3p75_oasis_mega_90_90_external_inference_20260605/predictions.csv"
CDR_ALIGNMENT = OUT_DIR / "oasis_cdr_scan_alignment.csv"

SEED = 42
ORIGINAL_C_GRID = [0.001, 0.01, 0.1, 1.0]
INNER_FOLDS = 5


def log(msg: str) -> None:
    print(msg, flush=True)


# ── Part A: confirm persisted Stage-A joblib does NOT reproduce canonical score (missing-artifact evidence) ──
persisted_check_rows = []
lat_all = pd.read_csv(OASIS_LOCKED_LATENTS)
mu_cols = sorted([c for c in lat_all.columns if c.startswith("mu_")], key=lambda c: int(c.split("_")[1]))
canon_all = pd.read_csv(CANON_PRED)
canon_all = canon_all[
    (canon_all.candidate == "promoted_beta3p75_oof_ecdf")
    & (canon_all.prediction_level == "fold_model")
    & (canon_all.build_candidate == "runwise164_pilot_parity")
]

for fold in range(1, 6):
    pipe = joblib.load(RUN_DIR / f"fold_{fold}" / f"classifier_logreg_final_pipeline_fold_{fold}.joblib")
    latf = lat_all[lat_all["fold"] == fold].rename(columns={c: c.replace("mu_", "latent_") for c in mu_cols})
    X = latf[[c.replace("mu_", "latent_") for c in mu_cols] + ["Age", "Sex"]]
    dfs = [c.estimator.decision_function(X) for c in pipe.calibrated_classifiers_]
    raw_recon_persisted = np.mean(dfs, axis=0)
    canon_f = canon_all[canon_all.fold == str(fold)].set_index("subject_id")
    joined = canon_f[["y_score_raw"]].join(
        latf.set_index("SubjectID").assign(raw_recon_persisted=raw_recon_persisted)[["raw_recon_persisted"]],
        how="inner",
    )
    corr = joined["y_score_raw"].corr(joined["raw_recon_persisted"])
    persisted_check_rows.append(dict(fold=fold, n=len(joined),
                                      corr_persisted_stageA_pipeline_vs_canonical_y_score_raw=corr))
persisted_check_df = pd.DataFrame(persisted_check_rows)
log("Persisted Stage-A joblib vs canonical y_score_raw (should NOT be ~1.0):")
log(persisted_check_df.to_string(index=False))
persisted_check_df.to_csv(OUT_DIR / "_persisted_pipeline_mismatch_check.csv", index=False)

# ── Part B: ADNI-only deterministic Stage-B reconstruction, per fold, keeping fitted `best` ──
decomp_rows = []
verify_rows = []
for fold in range(1, 6):
    adni_path = RUN_DIR / "classifier_only_readout/latent_cache" / f"fold_{fold}_trainDev_latent_mu.csv"
    train = pd.read_csv(adni_path)
    mu_cols_adni = sorted([c for c in train.columns if c.startswith("mu_")], key=lambda c: int(c.split("_")[1]))
    train = train.rename(columns={c: c.replace("mu_", "latent_") for c in mu_cols_adni})
    latent_cols = [c.replace("mu_", "latent_") for c in mu_cols_adni]
    assert len(latent_cols) == 384

    train["ResearchGroup_Mapped"] = train["ResearchGroup_Mapped"].astype(str).str.upper().str.strip()
    train["y"] = train["ResearchGroup_Mapped"].map({"CN": 0, "AD": 1})
    train = train.dropna(subset=["y"]).copy()
    train["y"] = train["y"].astype(int)
    train["Sex"] = train["Sex"].astype(str).str.upper().str.strip().map({"M": "M", "MALE": "M", "F": "F", "FEMALE": "F"})
    train["Age"] = pd.to_numeric(train["Age"], errors="coerce")

    x_train = train[latent_cols + ["Age", "Sex"]].copy()
    y_train = train["y"].to_numpy(dtype=int)

    pre = __import__("sklearn.compose", fromlist=["ColumnTransformer"]).ColumnTransformer(
        [
            ("latent", Pipeline([("scaler", StandardScaler())]), latent_cols),
            ("age", Pipeline([("imp", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), ["Age"]),
            ("sex", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", make_ohe())]), ["Sex"]),
        ],
        remainder="drop", sparse_threshold=0.0,
    )
    pipe = Pipeline([
        ("pre", pre),
        ("model", LogisticRegression(penalty="l2", solver="lbfgs", class_weight="balanced",
                                       max_iter=5000, random_state=SEED + fold)),
    ])
    key_df = pd.DataFrame({
        "dx": train["ResearchGroup_Mapped"].fillna("DX_UNKNOWN").astype(str),
        "mfr": train.get("Manufacturer", pd.Series(["MFR_UNKNOWN"] * len(train))).fillna("MFR_UNKNOWN").astype(str),
    })
    key = key_df.apply(lambda r: "_".join(r.values.astype(str)), axis=1)
    min_count = int(key.value_counts().min())
    strat_key = key if min_count >= INNER_FOLDS else train["y"]
    cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=SEED + fold + 30)
    splits = list(cv.split(x_train, strat_key))
    search = GridSearchCV(pipe, {"model__C": ORIGINAL_C_GRID}, scoring="roc_auc", cv=splits, n_jobs=8, refit=True)
    search.fit(x_train, y_train)
    best = search.best_estimator_

    # OOF scores on ADNI trainDev (for ECDF reference) -- reused exactly as in the canonical script.
    oof_scores = cross_val_predict(clone(best), x_train, y_train, cv=splits, method="predict_proba", n_jobs=8)[:, 1]

    # Score OASIS locked-arm latents for this fold.
    latf = lat_all[lat_all["fold"] == fold].rename(columns={c: c.replace("mu_", "latent_") for c in mu_cols})
    x_ext = latf[latent_cols + ["Age", "Sex"]].copy()
    raw_ext = score_1d(best, x_ext)  # predict_proba, matches canonical y_score_raw definition

    def oof_ecdf_interp(oof, values):
        s = np.sort(oof)
        n = len(s)
        pct = (np.arange(1, n + 1) - 0.5) / n
        return np.interp(values, s, pct, left=0.0, right=1.0)

    ecdf_ext = oof_ecdf_interp(oof_scores, raw_ext)

    canon_f = canon_all[canon_all.fold == str(fold)].set_index("subject_id")
    verify = canon_f[["y_score_raw", "y_score"]].join(
        latf.set_index("SubjectID").assign(raw_recon=raw_ext, ecdf_recon=ecdf_ext)[["raw_recon", "ecdf_recon"]],
        how="inner",
    )
    verify_rows.append(dict(
        fold=fold, n=len(verify),
        max_abs_diff_raw_score=(verify["y_score_raw"] - verify["raw_recon"]).abs().max(),
        max_abs_diff_ecdf_score=(verify["y_score"] - verify["ecdf_recon"]).abs().max(),
        corr_raw=verify["y_score_raw"].corr(verify["raw_recon"]),
    ))

    # ── Decompose the linear logit (pre-sigmoid decision_function) ──
    model: LogisticRegression = best.named_steps["model"]
    ct = best.named_steps["pre"]
    scaler_latent: StandardScaler = ct.named_transformers_["latent"].named_steps["scaler"]
    scaler_age: StandardScaler = ct.named_transformers_["age"].named_steps["scaler"]
    ohe = ct.named_transformers_["sex"].named_steps["ohe"]
    sex_categories = list(ohe.categories_[0])

    coef = model.coef_.ravel()
    intercept = float(model.intercept_[0])
    n_latent = len(latent_cols)
    coef_latent = coef[:n_latent]
    coef_age = coef[n_latent]
    coef_sex = coef[n_latent + 1: n_latent + 1 + len(sex_categories)]

    z_latent = (x_ext[latent_cols].to_numpy() - scaler_latent.mean_) / scaler_latent.scale_
    latent_component = z_latent @ coef_latent

    age_imputed = x_ext["Age"].fillna(x_ext["Age"].median()).to_numpy()
    z_age = (age_imputed - scaler_age.mean_[0]) / scaler_age.scale_[0]
    age_component = z_age * coef_age

    sex_vals = x_ext["Sex"].astype(str)
    sex_onehot = np.zeros((len(x_ext), len(sex_categories)))
    for i, cat in enumerate(sex_categories):
        sex_onehot[:, i] = (sex_vals == cat).astype(float)
    sex_component = sex_onehot @ coef_sex

    reconstructed_logit = latent_component + age_component + sex_component + intercept

    dfold = pd.DataFrame({
        "SubjectID": latf["SubjectID"].to_numpy(),
        "fold": fold,
        "latent_component": latent_component,
        "age_component": age_component,
        "sex_component": sex_component,
        "intercept": intercept,
        "reconstructed_logit": reconstructed_logit,
        "reconstructed_proba": 1.0 / (1.0 + np.exp(-reconstructed_logit)),
        "raw_ext_score_1d": raw_ext,
    })
    decomp_rows.append(dfold)
    log(f"fold {fold}: decomposition done, max|logit_recon_proba - raw_ext| = "
        f"{np.max(np.abs(dfold['reconstructed_proba'] - dfold['raw_ext_score_1d'])):.2e}")

decomp = pd.concat(decomp_rows, ignore_index=True)
verify_df = pd.DataFrame(verify_rows)
log("\nVerification: reconstructed Stage-B raw/ECDF score vs canonical predictions.csv:")
log(verify_df.to_string(index=False))
verify_df.to_csv(OUT_DIR / "_stageB_reconstruction_verification.csv", index=False)

decomp.to_csv(OUT_DIR / "_score_component_decomposition_by_fold.csv", index=False)

# ── Ensemble across folds (mean component per subject, matching the project's ensemble convention) ──
ens = decomp.groupby("SubjectID").agg(
    latent_component=("latent_component", "mean"),
    age_component=("age_component", "mean"),
    sex_component=("sex_component", "mean"),
    reconstructed_logit=("reconstructed_logit", "mean"),
).reset_index()

# ── CDR association for the latent-only component (Task-1 mapping B: nearest within 365d) ──
align = pd.read_csv(CDR_ALIGNMENT)
merged = ens.merge(
    align[["subject_id", "B_nearest_365d_CDRTOT", "Age", "Sex", "sex_male"]],
    left_on="SubjectID", right_on="subject_id", how="inner",
)
merged = merged.dropna(subset=["B_nearest_365d_CDRTOT"])
log(f"\nn subjects with latent-component + CDR (mapping B): {len(merged)}")

rho_raw, p_raw = spearmanr(merged["latent_component"], merged["B_nearest_365d_CDRTOT"])


def partial_spearman(score, cdr_ordinal, age, sex_male):
    r_score = rankdata(score)
    r_cdr = rankdata(cdr_ordinal)
    X = np.column_stack([age, sex_male]).astype(float)
    resid_score = r_score - LinearRegression().fit(X, r_score).predict(X)
    resid_cdr = r_cdr - LinearRegression().fit(X, r_cdr).predict(X)
    rho, pval = pearsonr(resid_score, resid_cdr)
    return float(rho), float(pval)


rho_partial, p_partial = partial_spearman(merged["latent_component"], merged["B_nearest_365d_CDRTOT"],
                                            merged["Age"], merged["sex_male"])

# Also test the full reconstructed logit (latent+age+sex+intercept) and the age/sex components alone, for context.
rho_full, p_full = spearmanr(merged["reconstructed_logit"], merged["B_nearest_365d_CDRTOT"])
rho_age, p_age = spearmanr(merged["age_component"], merged["B_nearest_365d_CDRTOT"])

latent_cdr_result = dict(
    n=len(merged),
    spearman_rho_latent_component_vs_cdr=rho_raw, spearman_pvalue_latent_component_vs_cdr=p_raw,
    partial_spearman_rho_latent_component_vs_cdr_age_sex_adj=rho_partial,
    partial_spearman_pvalue_latent_component_vs_cdr_age_sex_adj=p_partial,
    spearman_rho_full_reconstructed_logit_vs_cdr=rho_full, spearman_pvalue_full_reconstructed_logit_vs_cdr=p_full,
    spearman_rho_age_component_vs_cdr=rho_age, spearman_pvalue_age_component_vs_cdr=p_age,
)
log(json.dumps(latent_cdr_result, indent=2))

with open(OUT_DIR / "_latent_component_cdr_association.json", "w") as f:
    json.dump(latent_cdr_result, f, indent=2)
ens.merge(align[["subject_id"]], left_on="SubjectID", right_on="subject_id", how="left").drop(columns=["subject_id"]).to_csv(
    OUT_DIR / "_score_component_ensemble_per_subject.csv", index=False
)

log("Task 3 computation done.")
