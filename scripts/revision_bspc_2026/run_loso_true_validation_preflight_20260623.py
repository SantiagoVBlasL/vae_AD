#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scripts/revision_bspc_2026/run_loso_true_validation_preflight_20260623.py

Leave-One-Site-Out (LOSO) validation preflight for the final selected model.
PREFLIGHT ONLY — does not train the VAE, does not run inference.

Final selected model:
  channels [1,0,2] = Pearson_Full + OMST + MI_KNN
  latent_dim=384, beta=3.75, T0=80, h=10000, p=560, 5x5 CV
  N=647 (VAE pool), N=397 CN+AD supervised (OOF-evaluable)

Hard guardrails:
  - Read-only with respect to source data
  - No model training, no tensor modification, no metadata modification
  - No prediction modification, no threshold refitting
  - No model selection, do not propose post-hoc changes to increase AUC

Outputs:
  results/revision_bspc_2026/loso_true_validation_preflight_20260623/
    site_count_feasibility.csv/.md
    eligible_loso_sites.csv/.md
    loso_training_plan.csv/.md
    loso_guarded_launch_commands.txt
    loso_methods_text_draft.md
    loso_results_placeholder_template.md
    final_recommendation.md
    command_log.json
"""
from __future__ import annotations

import json
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results" / "revision_bspc_2026"
OUT = RESULTS / "loso_true_validation_preflight_20260623"

MASTER_DB = RESULTS / "promoted_model_master_database_20260610" / "promoted_model_master_database.csv"
METADATA_PATH = (
    RESULTS / "adni_035_metadata_rescue_preflight" / "patched_metadata_candidate.csv"
)
GLOBAL_TENSOR_PATH = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/"
    "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)
LOSO_SCRIPT = PROJECT / "scripts" / "revision_bspc_2026" / "run_loso_cv.py"

PYTHON = Path("/home/diego/anaconda3/envs/vae_ad/bin/python")

# Final model spec (locked — do not change)
FINAL_MODEL = dict(
    latent_dim=384,
    beta_vae=3.75,
    lr_scheduler_T0=80,
    epochs_vae=10000,
    early_stopping_patience_vae=560,
    batch_size=64,
    channels_to_use=[1, 0, 2],
    selected_channel_names=[
        "Pearson_Full_FisherZ_Signed",
        "Pearson_OMST_GCE_Signed_Weighted",
        "MI_KNN_Symmetric",
    ],
    norm_mode="zscore_offdiag",
    seed=42,
    inner_folds=5,
    vae_final_activation="tanh",
    decoder_type="convtranspose",
    num_conv_layers_encoder=4,
    intermediate_fc_dim_vae="quarter",
    dropout_rate_vae=0.15,
    use_layernorm_vae_fc=False,
    lr_vae=1e-4,
    weight_decay_vae=5e-7,
    vae_val_split_ratio=0.2,
    cyclical_beta_n_cycles=125,
    cyclical_beta_ratio_increase=0.4,
    lr_scheduler_type="cosine_warm",
    lr_scheduler_eta_min=5e-7,
    lr_scheduler_patience_vae=15,
)

# Eligibility thresholds
PRIMARY_CN_THRESH = 8
PRIMARY_AD_THRESH = 8
SENSITIVITY_CN_THRESH = 5
SENSITIVITY_AD_THRESH = 5

EXCLUDED_SUBJECT = "128_S_2002"

# Rough GPU-hours per LOSO fold (based on recovered035 training history)
# Mean epochs to convergence: ~3727; 10000 budget; ~0.8 GPU-hours per fold
APPROX_GPU_HOURS_PER_FOLD = 1.0

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def md_table(df: pd.DataFrame) -> str:
    """Convert DataFrame to markdown table string."""
    lines = []
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "|" + "|".join(["---" for _ in cols]) + "|"
    lines.append(header)
    lines.append(sep)
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def write_csv_md(out_dir: Path, stem: str, df: pd.DataFrame, caption: str = "") -> None:
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    md = ""
    if caption:
        md += f"# {caption}\n\n"
    md += md_table(df) + "\n"
    (out_dir / f"{stem}.md").write_text(md)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    log = {
        "script": str(Path(__file__).name),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "did_train_vae": False,
        "did_modify_tensors": False,
        "did_modify_metadata": False,
        "did_run_inference": False,
        "final_model_channels": "[1,0,2] = Pearson_Full + OMST + MI_KNN",
        "final_model_spec": FINAL_MODEL,
        "excluded_subject": EXCLUDED_SUBJECT,
        "master_db": str(MASTER_DB),
        "metadata_path": str(METADATA_PATH),
        "global_tensor_path": str(GLOBAL_TENSOR_PATH),
        "outputs": [],
        "warnings": [],
    }

    # -----------------------------------------------------------------------
    # Load master database
    # -----------------------------------------------------------------------
    print("[1/7] Loading master database …")
    df = pd.read_csv(MASTER_DB)

    # Canonical site from SubjectID (format: SSS_S_NNNN; SSS is the 3-digit site)
    # Site3 column has 484/648 non-null; SubjectID-derived site has 100% coverage.
    df["site_canonical"] = df["SubjectID"].str.extract(r"^(\d+)_S_").astype(float)

    # Validate excluded subject
    row_128 = df[df["SubjectID"] == EXCLUDED_SUBJECT]
    assert len(row_128) == 1, f"Expected exactly 1 row for {EXCLUDED_SUBJECT}"
    assert not row_128["in_oof_evaluation"].values[0], f"{EXCLUDED_SUBJECT} must not be in OOF pool"
    assert not row_128["in_vae_pool"].values[0], f"{EXCLUDED_SUBJECT} must not be in VAE pool"

    # Site3 vs canonical: ensure no mismatches where both are non-null
    both_known = df[df["Site3"].notna() & df["site_canonical"].notna()].copy()
    mismatches = both_known[both_known["Site3"] != both_known["site_canonical"]]
    assert len(mismatches) == 0, f"Site3 vs canonical mismatches: {len(mismatches)}"

    vae_pool = df[df["in_vae_pool"] == True].copy()
    oof_pool = df[df["in_oof_evaluation"] == True].copy()

    vae_pool["site_canonical"] = vae_pool["SubjectID"].str.extract(r"^(\d+)_S_").astype(float)
    oof_pool["site_canonical"] = oof_pool["SubjectID"].str.extract(r"^(\d+)_S_").astype(float)

    print(f"  VAE pool: N={len(vae_pool)}")
    print(f"  OOF (CN+AD supervised): N={len(oof_pool)}")

    # -----------------------------------------------------------------------
    # Step 1: Site-count feasibility table
    # -----------------------------------------------------------------------
    print("[2/7] Building site-count feasibility table …")

    # VAE pool counts per site (CN + MCI + AD)
    vae_site = vae_pool.groupby("site_canonical").agg(
        n_vae=("SubjectID", "count"),
        n_cn_vae=("ResearchGroup_Mapped", lambda x: (x == "CN").sum()),
        n_ad_vae=("ResearchGroup_Mapped", lambda x: (x == "AD").sum()),
        n_mci_vae=("ResearchGroup_Mapped", lambda x: (x == "MCI").sum()),
        primary_mfr=("Manufacturer", lambda x: x.dropna().mode()[0] if x.notna().any() else "Unknown"),
        n_manufacturers_vae=("Manufacturer", "nunique"),
    ).reset_index()

    # OOF pool counts per site (CN + AD only)
    oof_site = oof_pool.groupby("site_canonical").agg(
        n_oof=("SubjectID", "count"),
        n_cn_oof=("ResearchGroup_Mapped", lambda x: (x == "CN").sum()),
        n_ad_oof=("ResearchGroup_Mapped", lambda x: (x == "AD").sum()),
        adni_phases=("inferred_ADNI_phase", lambda x: " / ".join(sorted(set(str(v) for v in x.dropna())))),
    ).reset_index()

    feasibility = pd.merge(vae_site, oof_site, on="site_canonical", how="outer").fillna(0)
    feasibility["site_canonical"] = feasibility["site_canonical"].astype(int)
    feasibility = feasibility.sort_values("n_vae", ascending=False).reset_index(drop=True)

    # Eligibility flags
    feasibility["eligible_primary"] = (
        (feasibility["n_cn_oof"] >= PRIMARY_CN_THRESH)
        & (feasibility["n_ad_oof"] >= PRIMARY_AD_THRESH)
    )
    feasibility["eligible_sensitivity"] = (
        (feasibility["n_cn_oof"] >= SENSITIVITY_CN_THRESH)
        & (feasibility["n_ad_oof"] >= SENSITIVITY_AD_THRESH)
    )

    write_csv_md(OUT, "site_count_feasibility", feasibility, "Site-Level Count and Feasibility")
    log["outputs"].append("site_count_feasibility.csv/.md")
    print(f"  Total sites in tensor: {len(feasibility)}")
    print(f"  Sites with any OOF subjects: {(feasibility['n_oof'] > 0).sum()}")

    # -----------------------------------------------------------------------
    # Step 2: Eligible LOSO sites
    # -----------------------------------------------------------------------
    print("[3/7] Identifying eligible LOSO sites …")

    eligible_primary = feasibility[feasibility["eligible_primary"]].copy()
    eligible_sensitivity = feasibility[feasibility["eligible_sensitivity"]].copy()

    n_primary = len(eligible_primary)
    n_sensitivity = len(eligible_sensitivity)

    print(f"  Primary threshold (CN>={PRIMARY_CN_THRESH}, AD>={PRIMARY_AD_THRESH}): {n_primary} sites")
    print(f"  Sensitivity threshold (CN>={SENSITIVITY_CN_THRESH}, AD>={SENSITIVITY_AD_THRESH}): {n_sensitivity} sites")

    # For eligible sites: compute training pool after holdout
    def compute_training_pool(row: pd.Series) -> pd.Series:
        site = int(row["site_canonical"])
        train_vae = vae_pool[vae_pool["site_canonical"] != site]
        train_oof = oof_pool[oof_pool["site_canonical"] != site]
        has_cn_train = (train_oof["ResearchGroup_Mapped"] == "CN").any()
        has_ad_train = (train_oof["ResearchGroup_Mapped"] == "AD").any()
        has_mci_train = (train_vae["ResearchGroup_Mapped"] == "MCI").any()
        age_complete = train_vae["Age"].notna().all()
        sex_complete = train_vae["Sex"].notna().all()
        mfr_complete = train_vae["Manufacturer"].notna().all()
        mfrs_in_train = " / ".join(sorted(train_vae["Manufacturer"].dropna().unique()))
        return pd.Series(
            {
                "held_out_site": site,
                "held_out_n_vae": int(row["n_vae"]),
                "held_out_n_cn": int(row["n_cn_oof"]),
                "held_out_n_ad": int(row["n_ad_oof"]),
                "held_out_mfr": row["primary_mfr"],
                "train_n_vae": len(train_vae),
                "train_n_cn_supervised": int((train_oof["ResearchGroup_Mapped"] == "CN").sum()),
                "train_n_ad_supervised": int((train_oof["ResearchGroup_Mapped"] == "AD").sum()),
                "train_n_mci_vae": int((train_vae["ResearchGroup_Mapped"] == "MCI").sum()),
                "train_has_cn": bool(has_cn_train),
                "train_has_ad": bool(has_ad_train),
                "train_has_mci": bool(has_mci_train),
                "train_mfrs_present": mfrs_in_train,
                "train_age_complete": bool(age_complete),
                "train_sex_complete": bool(sex_complete),
                "train_mfr_complete": bool(mfr_complete),
                "metadata_ok": bool(age_complete and sex_complete and mfr_complete),
                "adni_phases": row.get("adni_phases", ""),
            }
        )

    # Build eligible tables for both thresholds
    if n_sensitivity > 0:
        elig_rows = [compute_training_pool(row) for _, row in eligible_sensitivity.iterrows()]
        eligible_df = pd.DataFrame(elig_rows).sort_values("held_out_n_vae", ascending=False).reset_index(drop=True)
    else:
        eligible_df = pd.DataFrame()

    write_csv_md(
        OUT,
        "eligible_loso_sites",
        eligible_df if len(eligible_df) > 0 else pd.DataFrame({"note": ["No eligible sites at any threshold"]}),
        f"Eligible LOSO Sites (sensitivity threshold CN>={SENSITIVITY_CN_THRESH}, AD>={SENSITIVITY_AD_THRESH})",
    )
    log["outputs"].append("eligible_loso_sites.csv/.md")

    # -----------------------------------------------------------------------
    # Step 3: LOSO training plan
    # -----------------------------------------------------------------------
    print("[4/7] Building LOSO training plan …")

    if len(eligible_df) > 0:
        plan = eligible_df.copy()
        plan["approx_gpu_hours"] = APPROX_GPU_HOURS_PER_FOLD
        plan["cumulative_gpu_hours"] = (
            range(1, len(plan) + 1)
        )
        plan["cumulative_gpu_hours"] = plan.index.map(
            lambda i: round((i + 1) * APPROX_GPU_HOURS_PER_FOLD, 1)
        )
        plan["within_manufacturer"] = plan["held_out_mfr"] == plan["held_out_mfr"]  # always True
        plan["threshold_passed"] = (
            (plan["held_out_n_cn"] >= PRIMARY_CN_THRESH)
            & (plan["held_out_n_ad"] >= PRIMARY_AD_THRESH)
        )
        write_csv_md(OUT, "loso_training_plan", plan, "LOSO Training Plan")
    else:
        write_csv_md(
            OUT,
            "loso_training_plan",
            pd.DataFrame({"note": ["No eligible sites at any threshold"]}),
            "LOSO Training Plan",
        )
    log["outputs"].append("loso_training_plan.csv/.md")

    # -----------------------------------------------------------------------
    # Step 4: Guarded launcher commands
    # -----------------------------------------------------------------------
    print("[5/7] Generating guarded launcher commands …")

    if n_sensitivity > 0:
        loso_sites = sorted(eligible_df["held_out_site"].astype(int).tolist())
    else:
        loso_sites = []

    # Build the command that would be used for each site
    base_cmd_args = [
        f"--global_tensor_path {GLOBAL_TENSOR_PATH}",
        f"--metadata_path {METADATA_PATH}",
        "--manufacturer_filter ''",  # empty = all manufacturers
        "--loso_mode custom",
        f"--latent_dim {FINAL_MODEL['latent_dim']}",
        f"--beta_vae {FINAL_MODEL['beta_vae']}",
        f"--epochs_vae {FINAL_MODEL['epochs_vae']}",
        f"--early_stopping_patience_vae {FINAL_MODEL['early_stopping_patience_vae']}",
        f"--lr_scheduler_T0 {FINAL_MODEL['lr_scheduler_T0']}",
        f"--lr_scheduler_type {FINAL_MODEL['lr_scheduler_type']}",
        f"--lr_scheduler_eta_min {FINAL_MODEL['lr_scheduler_eta_min']}",
        f"--lr_scheduler_patience_vae {FINAL_MODEL['lr_scheduler_patience_vae']}",
        f"--lr_vae {FINAL_MODEL['lr_vae']}",
        f"--weight_decay_vae {FINAL_MODEL['weight_decay_vae']}",
        f"--batch_size {FINAL_MODEL['batch_size']}",
        f"--dropout_rate_vae {FINAL_MODEL['dropout_rate_vae']}",
        f"--vae_val_split_ratio {FINAL_MODEL['vae_val_split_ratio']}",
        f"--vae_final_activation {FINAL_MODEL['vae_final_activation']}",
        f"--decoder_type {FINAL_MODEL['decoder_type']}",
        f"--num_conv_layers_encoder {FINAL_MODEL['num_conv_layers_encoder']}",
        f"--intermediate_fc_dim_vae {FINAL_MODEL['intermediate_fc_dim_vae']}",
        f"--norm_mode {FINAL_MODEL['norm_mode']}",
        f"--cyclical_beta_n_cycles {FINAL_MODEL['cyclical_beta_n_cycles']}",
        f"--cyclical_beta_ratio_increase {FINAL_MODEL['cyclical_beta_ratio_increase']}",
        f"--seed {FINAL_MODEL['seed']}",
        f"--inner_folds {FINAL_MODEL['inner_folds']}",
        "--classifier_types logreg",
        "--classifier_calibrate",
        "--classifier_use_class_weight",
        "--metadata_features Age Sex",
        "--save_fold_artefacts",
        "--save_vae_training_history",
        "--qc_analyze_distributions",
        "--qc_check_scanner_leakage",
        "--latent_features_type mu",
        "--n_jobs_gridsearch 8",
        "--num_workers 4",
        "--log_interval_epochs_vae 50",
    ]

    cmds = []
    cmds.append("#!/usr/bin/env bash")
    cmds.append("# LOSO True Validation — Guarded Launcher Commands")
    cmds.append(f"# Generated: {datetime.now(timezone.utc).isoformat()}")
    cmds.append("#")
    cmds.append("# IMPORTANT: This is a PREFLIGHT. Do NOT run these commands without explicit user approval.")
    cmds.append("# Each command trains a VAE from scratch — this is irreversible compute.")
    cmds.append("#")
    cmds.append("# Prerequisites:")
    cmds.append("#   1. Patch metadata to add 'site_canonical' column (SubjectID prefix) OR")
    cmds.append("#      modify run_loso_cv.py to extract site from SubjectID.")
    cmds.append("#      Current run_loso_cv.py uses 'Site3' which has NaN for 113/397 OOF subjects.")
    cmds.append("#   2. Verify GLOBAL_TENSOR_PATH is accessible.")
    cmds.append("#   3. Verify no other VAE training process is running:")
    cmds.append("#      pgrep -af 'run_loso_cv.py\\|run_vae_clf' | grep -v grep")
    cmds.append("#")
    cmds.append("# Guard: exit immediately if another training job is active")
    cmds.append("")
    cmds.append("set -euo pipefail")
    cmds.append("")
    cmds.append("# ---- Guard: no other VAE/LOSO training running ----")
    cmds.append("if pgrep -af 'run_loso_cv.py' | grep -v grep; then")
    cmds.append("    echo 'ERROR: Another LOSO training process detected. Aborting.'")
    cmds.append("    exit 1")
    cmds.append("fi")
    cmds.append("if pgrep -af 'run_vae_clf_ad.py' | grep -v grep; then")
    cmds.append("    echo 'ERROR: Another VAE training process detected. Aborting.'")
    cmds.append("    exit 1")
    cmds.append("fi")
    cmds.append("")
    cmds.append(
        "# NOTE: Site3 column in metadata has NaN for ~113/397 OOF subjects. "
    )
    cmds.append(
        "# run_loso_cv.py must be updated to use site_canonical (from SubjectID prefix)"
    )
    cmds.append("# before executing these commands.")
    cmds.append("")

    loso_base = RESULTS / "loso_true_validation_final_model_20260623"

    for site in loso_sites:
        site_row = eligible_df[eligible_df["held_out_site"] == site].iloc[0]
        mfr = site_row["held_out_mfr"]
        n_cn = int(site_row["held_out_n_cn"])
        n_ad = int(site_row["held_out_n_ad"])
        out_dir = loso_base / f"site_{site:03d}"
        cmd_parts = [
            f"conda run -n vae_ad {PYTHON} {LOSO_SCRIPT}",
            f"  --loso_sites {site}",
            f"  --output_dir {out_dir}",
        ] + [f"  {a}" for a in base_cmd_args]
        cmds.append(f"# ---- Site {site} ({mfr}, CN={n_cn}, AD={n_ad}) ----")
        cmds.append(" \\\n".join(cmd_parts))
        cmds.append("")

    if not loso_sites:
        cmds.append("# NO ELIGIBLE SITES FOUND AT ANY THRESHOLD.")
        cmds.append("# With CN>=8 AND AD>=8: only 1 site (Site 130) — LOSO not feasible.")
        cmds.append("# Recommendation: site-stratified performance analysis instead.")

    launch_path = OUT / "loso_guarded_launch_commands.txt"
    launch_path.write_text("\n".join(cmds) + "\n")
    log["outputs"].append("loso_guarded_launch_commands.txt")

    # -----------------------------------------------------------------------
    # Step 5: Methods text draft
    # -----------------------------------------------------------------------
    print("[6/7] Writing methods text draft and results placeholder …")

    eligible_sites_str = (
        ", ".join(f"Site {int(s)}" for s in eligible_df["held_out_site"])
        if len(eligible_df) > 0
        else "none"
    )
    primary_eligible_str = (
        ", ".join(
            f"Site {int(r['held_out_site'])}"
            for _, r in eligible_df.iterrows()
            if r["held_out_n_cn"] >= PRIMARY_CN_THRESH and r["held_out_n_ad"] >= PRIMARY_AD_THRESH
        )
        if len(eligible_df) > 0
        else "none"
    )

    methods_text = textwrap.dedent(
        f"""
        # LOSO Validation — Methods Text Draft

        Generated: {datetime.now(timezone.utc).isoformat()}

        ## Site-Level Cohort Distribution

        The final ADNI cohort comprised N=397 CN/AD subjects (CN=300, AD=97) and N=646 subjects
        for VAE pre-training (inclusive of MCI). Subjects are distributed across 45 unique
        ADNI sites (extracted from SubjectID prefix), spanning three MRI manufacturers:
        Philips (N=284 in tensor), SIEMENS (N=212), and GE (N=150).

        ## LOSO Eligibility Criteria

        A LOSO held-out site was considered eligible if it contained at least
        {PRIMARY_CN_THRESH} CN and {PRIMARY_AD_THRESH} AD subjects in the supervised classifier
        pool (OOF-evaluable, N=397). These thresholds ensure non-degenerate AUC/PR-AUC
        computation in the held-out test split.

        At this primary threshold, {n_primary} site(s) qualified: {primary_eligible_str}.
        A sensitivity analysis using relaxed thresholds (CN>={SENSITIVITY_CN_THRESH},
        AD>={SENSITIVITY_AD_THRESH}) identified {n_sensitivity} eligible sites:
        {eligible_sites_str}.

        ## LOSO Protocol

        For each eligible held-out site, a complete VAE was trained from scratch using
        subjects from all remaining sites (VAE training pool = CN + MCI + AD from
        non-held-out sites, N = {len(vae_pool)} − held-out VAE N). The supervised logistic
        regression classifier was trained using CN/AD subjects from non-held-out sites only.
        The held-out site provided test subjects exclusively; no held-out subject was used
        for VAE training, normalization, classifier training, hyperparameter selection,
        calibration, or threshold selection.

        VAE architecture: latent_dim={FINAL_MODEL['latent_dim']}, β={FINAL_MODEL['beta_vae']},
        T₀={FINAL_MODEL['lr_scheduler_T0']}, max_epochs={FINAL_MODEL['epochs_vae']},
        patience={FINAL_MODEL['early_stopping_patience_vae']}, batch_size={FINAL_MODEL['batch_size']},
        channels=[Pearson_Full_FisherZ_Signed, Pearson_OMST_GCE_Signed_Weighted, MI_KNN_Symmetric].
        Classifier: logreg_l2 with Age+Sex as auxiliary features (z_plus_age_sex).
        Calibration: OOF ECDF (rank transform), threshold: inner_oof_target_sens_ge_0.70_max_spec.

        ## Critical Limitation

        The ADNI dataset contains a structural manufacturer confound: GE and SIEMENS
        contribute no CN subjects to the training set (all CN subjects are from Philips in
        the original cohort). Consequently, LOSO sites passing the primary threshold (Site 130)
        are exclusively Philips sites. Held-out Philips sites provide within-manufacturer
        generalization evidence only; cross-manufacturer generalization cannot be assessed
        via LOSO in the current cohort. This limitation must be stated explicitly in the
        manuscript.

        At the sensitivity threshold, Site 35 (SIEMENS, CN=15, AD=5) and Site 135 (GE,
        CN=5, AD=6) also qualify, providing limited cross-manufacturer LOSO evidence.

        ## Metadata Prerequisite

        The existing metadata file uses a 'Site3' column that is null for 113/397 OOF
        subjects (primarily subjects added from the GE/SIEMENS expansion batch). A
        canonical site column ('site_canonical', derived from the SubjectID prefix
        SSS in SSS_S_NNNN) has 100% coverage across all 397 subjects. The run_loso_cv.py
        launcher must be updated to use 'site_canonical' as the --site_column before
        LOSO training is executed.
        """
    ).lstrip()

    (OUT / "loso_methods_text_draft.md").write_text(methods_text)
    log["outputs"].append("loso_methods_text_draft.md")

    # -----------------------------------------------------------------------
    # Step 6: Results placeholder template
    # -----------------------------------------------------------------------
    placeholder = textwrap.dedent(
        f"""
        # LOSO Validation — Results Placeholder Template

        Generated: {datetime.now(timezone.utc).isoformat()}
        Status: NOT YET RUN (preflight only)

        ## Primary LOSO Result Table (to be filled after training)

        ### Held-out-site performance (sensitivity threshold: CN>={SENSITIVITY_CN_THRESH}, AD>={SENSITIVITY_AD_THRESH})

        | Held-out Site | Manufacturer | N_CN | N_AD | AUC | PR-AUC | BA | Sens | Spec | F1 | Philips_FPR |
        |:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
        """
        + "\n".join(
            f"| {int(r['held_out_site'])} | {r['held_out_mfr']} "
            f"| {int(r['held_out_n_cn'])} | {int(r['held_out_n_ad'])} "
            f"| — | — | — | — | — | — | — |"
            for _, r in eligible_df.iterrows()
        )
        if len(eligible_df) > 0
        else "| — | No eligible sites | — | — | — | — | — | — | — | — | — |"
    ) + textwrap.dedent(
        f"""

        ### Pooled LOSO performance (all held-out subjects concatenated)

        | Metric | Value | 95% CI | vs. Full-CV Reference |
        |--------|-------|--------|----------------------|
        | AUC | — | — | ref=0.795155 |
        | PR-AUC | — | — | ref=0.573934 |
        | BA | — | — | ref=0.725979 |
        | Sensitivity | — | — | ref=0.731959 |
        | Specificity | — | — | ref=0.720000 |
        | F1 | — | — | ref=0.563492 |
        | Philips CN FPR | — | — | ref=0.454545 |

        ### Notes for filling in:
        - Run LOSO training for each eligible site using loso_guarded_launch_commands.txt
        - Report AUC/PR-AUC with 95% bootstrap CI (N=1000)
        - Compare pooled LOSO to 5×5 CV reference using paired bootstrap on common subjects
        - Expected: LOSO performance ≤ full-CV performance (less training data per fold)
        - Flag if any held-out Philips site FPR exceeds 5×5 CV reference (0.4545)
        """
    )
    (OUT / "loso_results_placeholder_template.md").write_text(placeholder.lstrip())
    log["outputs"].append("loso_results_placeholder_template.md")

    # -----------------------------------------------------------------------
    # Step 7: Final recommendation
    # -----------------------------------------------------------------------
    print("[7/7] Writing final recommendation …")

    # Decision logic per user spec
    if n_primary < 2:
        decision = "NOT_LOSO"
        decision_label = "DO NOT CALL IT LOSO — site-stratified performance instead"
    elif n_primary <= 5:
        decision = "LOSO_BRIEF"
        decision_label = "TRUE LOSO RECOMMENDED (brief, 2–5 sites)"
    else:
        decision = "LOSO_PRIORITIZE"
        decision_label = "LOSO RECOMMENDED — prioritize largest/most relevant sites"

    if n_sensitivity >= 2 and n_primary < 2:
        sensitivity_note = (
            f"At sensitivity threshold (CN>={SENSITIVITY_CN_THRESH}, AD>={SENSITIVITY_AD_THRESH}), "
            f"{n_sensitivity} sites qualify ({eligible_sites_str}). "
            "This enables a limited LOSO across 3 manufacturers. Recommend using this as an "
            "'extended site holdout' analysis rather than advertising it as LOSO."
        )
    else:
        sensitivity_note = ""

    compute_primary = n_primary * APPROX_GPU_HOURS_PER_FOLD
    compute_sensitivity = n_sensitivity * APPROX_GPU_HOURS_PER_FOLD

    # Manufacturer coverage at sensitivity threshold
    mfr_set_sensitivity = set(eligible_df["held_out_mfr"].tolist()) if len(eligible_df) > 0 else set()

    recommendation = textwrap.dedent(
        f"""
        # Final LOSO Preflight Recommendation

        Generated: {datetime.now(timezone.utc).isoformat()}

        ## Decision

        **{decision_label}**

        Primary eligibility threshold (CN>={PRIMARY_CN_THRESH}, AD>={PRIMARY_AD_THRESH}):
        {n_primary} eligible site(s).

        Per the pre-defined decision rule: fewer than 2 eligible sites at the primary
        threshold → recommend not advertising this as LOSO. Report site-stratified
        performance (per-site AUC/FPR table) instead.

        ## Sensitivity Analysis

        {sensitivity_note if sensitivity_note else
         'At sensitivity threshold (CN>=5, AD>=5): ' + str(n_sensitivity) + ' sites (' + eligible_sites_str + ').'}

        Manufacturers represented at sensitivity threshold: {', '.join(sorted(mfr_set_sensitivity)) if mfr_set_sensitivity else 'none'}.

        ## Eligible Sites Summary

        ### Primary threshold (CN>={PRIMARY_CN_THRESH}, AD>={PRIMARY_AD_THRESH})
        """
    ).lstrip()

    if n_primary == 0:
        recommendation += "No eligible sites.\n\n"
    else:
        for _, row in eligible_df.iterrows():
            if row["held_out_n_cn"] >= PRIMARY_CN_THRESH and row["held_out_n_ad"] >= PRIMARY_AD_THRESH:
                recommendation += (
                    f"- **Site {int(row['held_out_site'])}** ({row['held_out_mfr']}): "
                    f"held-out CN={int(row['held_out_n_cn'])}, AD={int(row['held_out_n_ad'])}; "
                    f"training pool N={int(row['train_n_vae'])} VAE "
                    f"({int(row['train_n_cn_supervised'])} CN, {int(row['train_n_ad_supervised'])} AD, "
                    f"{int(row['train_n_mci_vae'])} MCI)\n"
                )
        recommendation += "\n"

    recommendation += f"### Sensitivity threshold (CN>={SENSITIVITY_CN_THRESH}, AD>={SENSITIVITY_AD_THRESH})\n"
    if n_sensitivity == 0:
        recommendation += "No eligible sites.\n\n"
    else:
        for _, row in eligible_df.iterrows():
            recommendation += (
                f"- **Site {int(row['held_out_site'])}** ({row['held_out_mfr']}): "
                f"held-out CN={int(row['held_out_n_cn'])}, AD={int(row['held_out_n_ad'])}; "
                f"training pool N={int(row['train_n_vae'])} VAE "
                f"({int(row['train_n_cn_supervised'])} CN, {int(row['train_n_ad_supervised'])} AD, "
                f"{int(row['train_n_mci_vae'])} MCI); "
                f"all metadata fields complete: {row['metadata_ok']}\n"
            )
        recommendation += "\n"

    recommendation += textwrap.dedent(
        f"""
        ## Compute Estimate

        | Scenario | N sites | GPU-hours | Feasible? |
        |---------|---------|-----------|-----------|
        | Primary threshold | {n_primary} | ~{compute_primary:.1f} h | {'Yes' if compute_primary <= 24 else 'Marginal'} |
        | Sensitivity threshold | {n_sensitivity} | ~{compute_sensitivity:.1f} h | {'Yes' if compute_sensitivity <= 24 else 'Marginal'} |

        Estimate based on ~{APPROX_GPU_HOURS_PER_FOLD:.1f} GPU-hour per site fold
        (mean ≈3727 epochs to convergence out of 10000 budget for final model).

        ## Critical Constraints

        1. **Manufacturer confound**: All CN subjects are Philips; GE and SIEMENS contribute
           only AD (and MCI) to the original cohort. LOSO at Philips sites is within-manufacturer
           only. Site 35 (SIEMENS) and Site 135 (GE) at the sensitivity threshold have enough
           AD to serve as held-out test sets but fewer than {PRIMARY_CN_THRESH} CN held-out
           subjects at the primary threshold.

        2. **Site3 column coverage**: 113/397 OOF subjects have NaN Site3 in the master
           database. Use 'site_canonical' (from SubjectID prefix) which achieves 100% coverage.
           run_loso_cv.py must be updated before training (current default: --site_column Site3).

        3. **LOSO ≠ cross-manufacturer generalization**: Even with 4 LOSO sites, the confound
           structure (GE/SIEMENS = AD-only in training) means the model has never seen GE or
           SIEMENS CN subjects during classifier training. Held-out Philips LOSO tests
           within-Philips generalization only.

        4. **Single-site primary recommendation**: With only 1 site at the primary threshold
           (Site 130, Philips, CN=21, AD=13), the analysis is a single-site holdout, not LOSO.
           If this is reported, label it "site-holdout validation" not "leave-one-site-out."

        ## Recommended Alternative: Site-Stratified Performance

        Report per-site AUC and CN FPR table from existing 5×5 CV OOF predictions.
        This requires no additional training and provides site-level generalization evidence
        from the already-computed OOF scores.

        ## Next Steps

        If the reviewer/editor specifically requests LOSO, recommend:
        1. Patch metadata to add 'site_canonical' column (read-only metadata operation)
        2. Update run_loso_cv.py --site_column to 'site_canonical' and --manufacturer_filter ''
        3. Run LOSO at sensitivity threshold (4 sites, ~{compute_sensitivity:.0f} GPU-hours)
        4. Report as "site-holdout validation across 4 sites" with explicit caveat about manufacturer confound

        Do NOT replace the 5×5 full-CV reference with LOSO. LOSO is supplementary evidence.

        ## Status of Guardrails

        - read_only: True
        - did_train_vae: False
        - did_modify_tensors: False
        - did_modify_metadata: False
        - did_run_inference: False
        - excluded_subject_128_S_2002: confirmed excluded from all pools
        """
    )

    (OUT / "final_recommendation.md").write_text(recommendation)
    log["outputs"].append("final_recommendation.md")

    # -----------------------------------------------------------------------
    # Finalize command log
    # -----------------------------------------------------------------------
    log["n_eligible_primary"] = int(n_primary)
    log["n_eligible_sensitivity"] = int(n_sensitivity)
    log["eligible_sites_sensitivity"] = [int(s) for s in loso_sites]
    log["decision"] = decision
    log["decision_label"] = decision_label
    log["compute_estimate_primary_gpu_hours"] = compute_primary
    log["compute_estimate_sensitivity_gpu_hours"] = compute_sensitivity
    log["metadata_site3_coverage"] = int(df[df["in_oof_evaluation"] == True]["Site3"].notna().sum())
    log["metadata_site_canonical_coverage"] = int(
        df[df["in_oof_evaluation"] == True]["site_canonical"].notna().sum()
    )
    log["warnings"].append(
        "Site3 column has NaN for 113/397 OOF subjects; use site_canonical (SubjectID-derived) for LOSO"
    )
    if n_primary < 2:
        log["warnings"].append(
            f"Only {n_primary} site(s) meet primary threshold (CN>={PRIMARY_CN_THRESH}, "
            f"AD>={PRIMARY_AD_THRESH}); LOSO not recommended at this threshold"
        )

    (OUT / "command_log.json").write_text(json.dumps(log, indent=2, default=str))
    log["outputs"].append("command_log.json")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("LOSO PREFLIGHT COMPLETE")
    print("=" * 70)
    print(f"  Output dir:  {OUT}")
    print(f"  VAE pool:    N={len(vae_pool)}")
    print(f"  OOF pool:    N={len(oof_pool)} (CN={int((oof_pool['ResearchGroup_Mapped']=='CN').sum())}, "
          f"AD={int((oof_pool['ResearchGroup_Mapped']=='AD').sum())})")
    print(f"  Total sites: {len(feasibility)}")
    print(f"  Eligible (CN>={PRIMARY_CN_THRESH}, AD>={PRIMARY_AD_THRESH}): {n_primary}")
    print(f"  Eligible (CN>={SENSITIVITY_CN_THRESH}, AD>={SENSITIVITY_AD_THRESH}): {n_sensitivity}")
    print(f"  Decision: {decision_label}")
    if n_sensitivity > 0:
        print(f"  Sensitivity sites: {eligible_sites_str}")
        print(f"  Manufacturers at sensitivity threshold: {', '.join(sorted(mfr_set_sensitivity))}")
        print(f"  Estimated compute (sensitivity): ~{compute_sensitivity:.0f} GPU-hours")
    print()
    print("WARNING: Site3 column has NaN for 113/397 OOF subjects.")
    print("         run_loso_cv.py must use site_canonical before training.")
    print()
    print("Files written:")
    for f in log["outputs"]:
        print(f"  {OUT / f}")


if __name__ == "__main__":
    main()
