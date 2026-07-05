"""
OASIS pilot vs new cohort clinical/demographic/motion difficulty audit.

Cohort A (pilot): 30 CN + 30 AD  (oasis_tanda_2026_05_25)
Cohort B (new):   60 CN + 60 AD  (oasis_next_60cn_60ad)

Outputs
-------
results/revision_bspc_2026/oasis_pilot_vs_new_clinical_motion_difficulty_audit_20260531/
  cohort_demographics.csv / .md
  cohort_motion.csv / .md
  cdr_distribution.csv / .md
  run_counts.csv / .md
  fn_vs_tp_ad_new.csv / .md
  summary.md
"""

import argparse, json, re, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

ROOT = Path(".")
OUT_DIR = Path(
    "results/revision_bspc_2026/"
    "oasis_pilot_vs_new_clinical_motion_difficulty_audit_20260531"
)

# ── data paths ────────────────────────────────────────────────────────────────

TANDA_RUN_MANIFEST = ROOT / "results/revision_bspc_2026/oasis_tanda_2026_05_25_audit/run_manifest.csv"
TANDA_SUBJECT_MANIFEST = ROOT / "results/revision_bspc_2026/oasis_tanda_2026_05_25_audit/subject_session_manifest.csv"
TANDA_SCORES = ROOT / "results/revision_bspc_2026/oasis_tanda_2026_05_25_external_scoring/predictions.csv"
TANDA_RP_DIR = ROOT / "data/Tanda_2026_05_25/RealignParameter"

NEW_SPLIT = ROOT / "results/revision_bspc_2026/oasis_60cn_60ad_calibration_test_protocol/split_calibration_test.csv"
NEW_SELECTED = ROOT / "data/OASIS_next_60CN_60AD_2026_05_26/selected_subjects_used.csv"
NEW_MOTION = ROOT / "results/revision_bspc_2026/oasis_next_60cn_60ad_processed_aal3_handoff_qc_20260530/motion_qc_by_file.csv"
NEW_SUBJECT_MANIFEST = ROOT / "results/revision_bspc_2026/oasis_next_60cn_60ad_tensor_build_pilot_parity_runwise_20260531/subject_manifest.csv"
NEW_SCORES = ROOT / (
    "results/revision_bspc_2026/"
    "oasis_next_60cn_60ad_external_scoring_pilot_parity_runwise_horizon4480_20260531/"
    "subject_scores.csv"
)

ADNI_FIXED_THRESHOLD = 0.483133  # mean of per-fold inner_oof_target_sens thresholds


# ── helpers ───────────────────────────────────────────────────────────────────

def md_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def fmt(v, decimals=2):
    if pd.isna(v):
        return "NaN"
    return f"{v:.{decimals}f}"


def mean_sd(series):
    s = series.dropna()
    return f"{s.mean():.2f} ± {s.std():.2f}"


def mannwhitney_p(a, b):
    a, b = a.dropna(), b.dropna()
    if len(a) < 3 or len(b) < 3:
        return np.nan
    _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return p


def session_day(experiment_id: str) -> int:
    """Extract day-offset from OAS3 id, handles _d0129 and ses-d0129 formats."""
    m = re.search(r"[-_]d(\d+)", str(experiment_id))
    return int(m.group(1)) if m else np.nan


def load_pilot_fd() -> pd.DataFrame:
    """Read per-run FD_Jenkinson files from DPABI RealignParameter tree."""
    records = []
    for run_dir in TANDA_RP_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        run_key = run_dir.name  # e.g. sub-OAS30001_ses-d0129_task-rest_run-02_bold
        fd_file = run_dir / f"FD_Jenkinson_{run_key}.txt"
        if not fd_file.exists():
            # fall back to Power
            fd_file = run_dir / f"FD_Power_{run_key}.txt"
        if not fd_file.exists():
            continue
        fd_series = np.loadtxt(fd_file)
        fd_series = fd_series[fd_series > 0] if fd_series.ndim == 1 else fd_series.ravel()
        # parse subject_id from run_key
        parts = run_key.split("_ses-")
        subject_id = parts[0]
        records.append(
            {
                "run_key": run_key,
                "subject_id": subject_id,
                "n_frames": len(fd_series),
                "mean_fd": float(np.mean(fd_series)),
                "max_fd": float(np.max(fd_series)),
                "pct_frames_fd_gt0p2": float(np.mean(fd_series > 0.2) * 100),
                "pct_frames_fd_gt0p5": float(np.mean(fd_series > 0.5) * 100),
            }
        )
    return pd.DataFrame(records)


# ── data loading ──────────────────────────────────────────────────────────────

def build_pilot_df() -> pd.DataFrame:
    """One row per subject for pilot cohort with all available attributes."""
    subj_df = pd.read_csv(TANDA_SUBJECT_MANIFEST)
    run_df = pd.read_csv(TANDA_RUN_MANIFEST)

    # CDR per subject (take from first run with non-null values; CDR is subject-level)
    cdr_per_subj = (
        run_df[["subject_id", "CDRTOT", "CDRSUM"]]
        .drop_duplicates("subject_id")
        .set_index("subject_id")
    )

    # QC-ok run count per subject
    run_counts = (
        run_df[run_df["roi_qc_ok"] == True]
        .groupby("subject_id")
        .size()
        .rename("n_runs_qc_ok")
    )
    total_runs = run_df.groupby("subject_id").size().rename("n_runs_total")

    # FD per subject — aggregate over runs
    fd_run = load_pilot_fd()
    fd_subj = (
        fd_run.groupby("subject_id")
        .agg(
            mean_fd=("mean_fd", "mean"),
            max_fd=("max_fd", "max"),
            pct_frames_fd_gt0p2=("pct_frames_fd_gt0p2", "mean"),
            pct_frames_fd_gt0p5=("pct_frames_fd_gt0p5", "mean"),
        )
        .reset_index()
    )

    # Scores — use ensemble (fold-averaged) probability from concatenated_timeseries candidate
    scores_raw = pd.read_csv(TANDA_SCORES)
    # Use ensemble model (prediction_level='ensemble') for concat candidate
    scores_ens = scores_raw[
        (scores_raw["prediction_level"] == "ensemble")
        & (scores_raw["build_candidate"] == "concatenated_timeseries")
    ][["SubjectID", "y_true", "y_score", "threshold_strategy", "adni_threshold"]].copy()
    # De-dup if multiple threshold strategies
    scores_ens = scores_ens.drop_duplicates("SubjectID")
    scores_ens.rename(columns={"SubjectID": "subject_id", "y_score": "prob_ensemble"}, inplace=True)

    # Session day from session_id
    subj_df["session_day"] = subj_df["session_id"].apply(session_day)

    df = subj_df.merge(cdr_per_subj, on="subject_id", how="left")
    df = df.merge(run_counts, on="subject_id", how="left")
    df = df.merge(total_runs, on="subject_id", how="left")
    df = df.merge(fd_subj, on="subject_id", how="left")
    df = df.merge(scores_ens[["subject_id", "prob_ensemble"]], on="subject_id", how="left")

    df["cohort"] = "pilot"
    df["y"] = (df["diagnosis"] == "AD_DEMENTIA").astype(int)
    # sex numeric: tanda has 1=male,2=female
    df["sex_label"] = df["sex"].map({1: "M", 2: "F"}).fillna(df["sex"])

    return df


def build_new_df() -> pd.DataFrame:
    """One row per subject for new 60+60 cohort."""
    split = pd.read_csv(NEW_SPLIT)
    selected = pd.read_csv(NEW_SELECTED)[
        ["subject_id", "CDRTOT", "CDRSUM", "diagnosis_confidence"]
    ].drop_duplicates("subject_id")

    subj_manifest = pd.read_csv(NEW_SUBJECT_MANIFEST)
    # n_runs per subject (selected_qc_runs is a count column)
    subj_manifest["n_runs_qc_ok"] = subj_manifest["selected_qc_runs"].astype(float)

    # Motion QC — aggregate per subject
    motion = pd.read_csv(NEW_MOTION)
    motion_subj = (
        motion.groupby("subject_id")
        .agg(
            mean_fd=("mean_fd_jenkinson", "mean"),
            max_fd=("max_fd_jenkinson", "max"),
            pct_frames_fd_gt0p2=("pct_frames_fd_gt0p2", "mean"),
            pct_frames_fd_gt0p5=("pct_frames_fd_gt0p5", "mean"),
            n_runs_motion_qc=("run_key", "count"),
        )
        .reset_index()
    )

    # Scores — use 140TR pilot_parity candidate (calibration + locked_test combined)
    scores_raw = pd.read_csv(NEW_SCORES)
    # Use 140TR candidate; one row per subject per candidate
    scores_140 = scores_raw[scores_raw["candidate"] == "runwise_140TR_pilot_parity"][
        ["SubjectID", "y", "prob_ensemble", "prob_fold1", "prob_fold2",
         "prob_fold3", "prob_fold4", "prob_fold5"]
    ].copy()
    scores_140.rename(columns={"SubjectID": "subject_id"}, inplace=True)

    # Session day
    split["session_day"] = split["session_id"].apply(session_day)

    df = split[
        ["subject_id", "session_id", "diagnosis", "age_at_MR", "sex",
         "protocol_subset", "session_day"]
    ].copy()
    df = df.merge(selected, on="subject_id", how="left")
    df = df.merge(
        subj_manifest[["subject_id", "n_runs_qc_ok", "selected_total_timepoints"]],
        on="subject_id", how="left",
    )
    df = df.merge(motion_subj, on="subject_id", how="left")
    df = df.merge(scores_140, on="subject_id", how="left")

    df["cohort"] = "new"
    df["y"] = (df["diagnosis"] == "AD_DEMENTIA").astype(int)
    df["sex_label"] = df["sex"].map({1: "M", 2: "F"}).fillna(df["sex"].astype(str))
    # n_runs_total not separately tracked for new cohort; use n_runs_qc_ok
    df["n_runs_total"] = df["n_runs_qc_ok"]

    return df


# ── analysis blocks ───────────────────────────────────────────────────────────

def section_demographics(pilot: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort_label, df in [("pilot (30CN+30AD)", pilot), ("new (60CN+60AD)", new)]:
        for dx in ["CN", "AD_DEMENTIA"]:
            sub = df[df["diagnosis"] == dx]
            n = len(sub)
            if n == 0:
                continue
            rows.append(
                {
                    "cohort": cohort_label,
                    "diagnosis": dx,
                    "N": n,
                    "age_mean_sd": mean_sd(sub["age_at_MR"]),
                    "age_min": fmt(sub["age_at_MR"].min()),
                    "age_max": fmt(sub["age_at_MR"].max()),
                    "pct_female": fmt(100 * (sub["sex_label"] == "F").mean()),
                    "session_day_mean_sd": mean_sd(sub["session_day"]),
                }
            )
    return pd.DataFrame(rows)


def section_cdr(pilot: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort_label, df in [("pilot", pilot), ("new", new)]:
        for dx in ["CN", "AD_DEMENTIA"]:
            sub = df[df["diagnosis"] == dx]
            n_cdr = sub["CDRTOT"].notna().sum()
            rows.append(
                {
                    "cohort": cohort_label,
                    "diagnosis": dx,
                    "N_with_CDR": int(n_cdr),
                    "CDRTOT_mean_sd": mean_sd(sub["CDRTOT"]),
                    "CDRSUM_mean_sd": mean_sd(sub["CDRSUM"]),
                    "CDRTOT_0": int((sub["CDRTOT"] == 0).sum()),
                    "CDRTOT_0p5": int((sub["CDRTOT"] == 0.5).sum()),
                    "CDRTOT_1": int((sub["CDRTOT"] == 1.0).sum()),
                    "CDRTOT_2plus": int((sub["CDRTOT"] >= 2.0).sum()),
                }
            )
    return pd.DataFrame(rows)


def section_runs(pilot: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort_label, df in [("pilot", pilot), ("new", new)]:
        for dx in ["CN", "AD_DEMENTIA"]:
            sub = df[df["diagnosis"] == dx]
            rows.append(
                {
                    "cohort": cohort_label,
                    "diagnosis": dx,
                    "n_runs_qc_ok_mean_sd": mean_sd(sub["n_runs_qc_ok"]),
                    "n_runs_1": int((sub["n_runs_qc_ok"] == 1).sum()),
                    "n_runs_2": int((sub["n_runs_qc_ok"] == 2).sum()),
                    "n_runs_3plus": int((sub["n_runs_qc_ok"] >= 3).sum()),
                }
            )
    return pd.DataFrame(rows)


def section_motion(pilot: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort_label, df in [("pilot", pilot), ("new", new)]:
        for dx in ["CN", "AD_DEMENTIA"]:
            sub = df[df["diagnosis"] == dx]
            rows.append(
                {
                    "cohort": cohort_label,
                    "diagnosis": dx,
                    "N": len(sub),
                    "mean_FD_mean_sd": mean_sd(sub["mean_fd"]),
                    "max_FD_mean_sd": mean_sd(sub["max_fd"]),
                    "pct_gt0p2_mean_sd": mean_sd(sub["pct_frames_fd_gt0p2"]),
                    "pct_gt0p5_mean_sd": mean_sd(sub["pct_frames_fd_gt0p5"]),
                    "pct_mean_FD_gt0p2": fmt(
                        100 * (sub["mean_fd"] > 0.2).mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def section_fn_vs_tp(new: pd.DataFrame) -> pd.DataFrame:
    ad = new[new["diagnosis"] == "AD_DEMENTIA"].copy()
    ad["predicted_positive"] = (ad["prob_ensemble"] >= ADNI_FIXED_THRESHOLD).astype(int)
    ad["outcome"] = ad["predicted_positive"].map({1: "TP", 0: "FN"})

    rows = []
    for outcome in ["TP", "FN"]:
        sub = ad[ad["outcome"] == outcome]
        n = len(sub)
        if n == 0:
            continue
        rows.append(
            {
                "outcome": outcome,
                "N": n,
                "age_mean_sd": mean_sd(sub["age_at_MR"]),
                "pct_female": fmt(100 * (sub["sex_label"] == "F").mean()),
                "CDRTOT_mean_sd": mean_sd(sub["CDRTOT"]),
                "CDRSUM_mean_sd": mean_sd(sub["CDRSUM"]),
                "mean_FD_mean_sd": mean_sd(sub["mean_fd"]),
                "pct_gt0p2_mean_sd": mean_sd(sub["pct_frames_fd_gt0p2"]),
                "prob_ensemble_mean_sd": mean_sd(sub["prob_ensemble"]),
                "CDRTOT_0": int((sub["CDRTOT"] == 0).sum()),
                "CDRTOT_0p5": int((sub["CDRTOT"] == 0.5).sum()),
                "CDRTOT_1": int((sub["CDRTOT"] == 1.0).sum()),
                "CDRTOT_2plus": int((sub["CDRTOT"] >= 2.0).sum()),
            }
        )

    # Statistical tests
    tp_sub = ad[ad["outcome"] == "TP"]
    fn_sub = ad[ad["outcome"] == "FN"]
    tests = {
        "metric": ["age", "CDRTOT", "CDRSUM", "mean_FD", "prob_ensemble"],
        "TP_mean_sd": [
            mean_sd(tp_sub["age_at_MR"]),
            mean_sd(tp_sub["CDRTOT"]),
            mean_sd(tp_sub["CDRSUM"]),
            mean_sd(tp_sub["mean_fd"]),
            mean_sd(tp_sub["prob_ensemble"]),
        ],
        "FN_mean_sd": [
            mean_sd(fn_sub["age_at_MR"]),
            mean_sd(fn_sub["CDRTOT"]),
            mean_sd(fn_sub["CDRSUM"]),
            mean_sd(fn_sub["mean_fd"]),
            mean_sd(fn_sub["prob_ensemble"]),
        ],
        "MW_p": [
            fmt(mannwhitney_p(tp_sub["age_at_MR"], fn_sub["age_at_MR"]), 3),
            fmt(mannwhitney_p(tp_sub["CDRTOT"], fn_sub["CDRTOT"]), 3),
            fmt(mannwhitney_p(tp_sub["CDRSUM"], fn_sub["CDRSUM"]), 3),
            fmt(mannwhitney_p(tp_sub["mean_fd"], fn_sub["mean_fd"]), 3),
            fmt(mannwhitney_p(tp_sub["prob_ensemble"], fn_sub["prob_ensemble"]), 3),
        ],
    }
    tests_df = pd.DataFrame(tests)

    return pd.DataFrame(rows), tests_df, ad[
        ["subject_id", "diagnosis", "age_at_MR", "sex_label", "CDRTOT", "CDRSUM",
         "mean_fd", "pct_frames_fd_gt0p2", "prob_ensemble",
         "prob_fold1", "prob_fold2", "prob_fold3", "prob_fold4", "prob_fold5",
         "outcome", "protocol_subset"]
    ].sort_values("prob_ensemble")


def section_cn_fp_vs_tn(new: pd.DataFrame) -> pd.DataFrame:
    cn = new[new["diagnosis"] == "CN"].copy()
    cn["predicted_positive"] = (cn["prob_ensemble"] >= ADNI_FIXED_THRESHOLD).astype(int)
    cn["outcome"] = cn["predicted_positive"].map({1: "FP", 0: "TN"})
    rows = []
    for outcome in ["FP", "TN"]:
        sub = cn[cn["outcome"] == outcome]
        n = len(sub)
        if n == 0:
            continue
        rows.append(
            {
                "outcome": outcome,
                "N": n,
                "age_mean_sd": mean_sd(sub["age_at_MR"]),
                "pct_female": fmt(100 * (sub["sex_label"] == "F").mean()),
                "mean_FD_mean_sd": mean_sd(sub["mean_fd"]),
                "prob_ensemble_mean_sd": mean_sd(sub["prob_ensemble"]),
            }
        )
    return pd.DataFrame(rows)


# ── comparison pilot vs new ── cross-cohort stat tests ───────────────────────

def section_cross_cohort_tests(pilot: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dx in ["CN", "AD_DEMENTIA"]:
        p = pilot[pilot["diagnosis"] == dx]
        n = new[new["diagnosis"] == dx]
        for metric, col in [
            ("age", "age_at_MR"),
            ("CDRTOT", "CDRTOT"),
            ("CDRSUM", "CDRSUM"),
            ("mean_FD", "mean_fd"),
            ("pct_frames_FD>0.2", "pct_frames_fd_gt0p2"),
            ("pct_frames_FD>0.5", "pct_frames_fd_gt0p5"),
        ]:
            rows.append(
                {
                    "diagnosis": dx,
                    "metric": metric,
                    "pilot_mean_sd": mean_sd(p[col]),
                    "new_mean_sd": mean_sd(n[col]),
                    "MW_p": fmt(mannwhitney_p(p[col], n[col]), 3),
                }
            )
    return pd.DataFrame(rows)


# ── writers ───────────────────────────────────────────────────────────────────

def write_pair(df: pd.DataFrame, stem: str) -> None:
    df.to_csv(OUT_DIR / f"{stem}.csv", index=False)
    (OUT_DIR / f"{stem}.md").write_text(md_table(df) + "\n")


# ── main ──────────────────────────────────────────────────────────────────────

def dry_run_check():
    missing = []
    for p in [
        TANDA_RUN_MANIFEST, TANDA_SUBJECT_MANIFEST, TANDA_SCORES,
        TANDA_RP_DIR,
        NEW_SPLIT, NEW_SELECTED, NEW_MOTION,
        NEW_SUBJECT_MANIFEST, NEW_SCORES,
    ]:
        if not Path(p).exists():
            missing.append(str(p))
    if missing:
        print("MISSING PATHS:")
        for m in missing:
            print(" ", m)
        sys.exit(1)
    print("DRY RUN OK — all paths present")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm", action="store_true", help="Run audit (omit for dry-run)")
    args = ap.parse_args()

    dry_run_check()
    if not args.confirm:
        print("Pass --confirm to run the audit.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Building pilot cohort dataframe …")
    pilot = build_pilot_df()
    print(f"  pilot: {len(pilot)} subjects, CN={( pilot.diagnosis=='CN').sum()} AD={(pilot.diagnosis=='AD_DEMENTIA').sum()}")

    print("Building new cohort dataframe …")
    new = build_new_df()
    print(f"  new:   {len(new)} subjects, CN={(new.diagnosis=='CN').sum()} AD={(new.diagnosis=='AD_DEMENTIA').sum()}")

    # ── 1. demographics ────────────────────────────────────────────────────────
    demo = section_demographics(pilot, new)
    write_pair(demo, "cohort_demographics")

    # ── 2. CDR distribution ────────────────────────────────────────────────────
    cdr = section_cdr(pilot, new)
    write_pair(cdr, "cdr_distribution")

    # ── 3. run counts ──────────────────────────────────────────────────────────
    runs = section_runs(pilot, new)
    write_pair(runs, "run_counts")

    # ── 4. motion ──────────────────────────────────────────────────────────────
    motion = section_motion(pilot, new)
    write_pair(motion, "cohort_motion")

    # ── 5. cross-cohort stat tests ─────────────────────────────────────────────
    tests = section_cross_cohort_tests(pilot, new)
    write_pair(tests, "cross_cohort_tests")

    # ── 6. FN vs TP in new AD ──────────────────────────────────────────────────
    fn_tp_summary, fn_tp_stats, fn_tp_detail = section_fn_vs_tp(new)
    write_pair(fn_tp_summary, "fn_vs_tp_ad_new_summary")
    write_pair(fn_tp_stats, "fn_vs_tp_ad_new_stats")
    write_pair(fn_tp_detail, "fn_vs_tp_ad_new_detail")

    # ── 7. FP vs TN in new CN ──────────────────────────────────────────────────
    cn_outcomes = section_cn_fp_vs_tn(new)
    write_pair(cn_outcomes, "fp_vs_tn_cn_new")

    # ── save per-subject data ──────────────────────────────────────────────────
    pilot_out = pilot[
        ["subject_id", "cohort", "diagnosis", "age_at_MR", "sex_label",
         "session_day", "CDRTOT", "CDRSUM", "n_runs_qc_ok",
         "mean_fd", "max_fd", "pct_frames_fd_gt0p2", "pct_frames_fd_gt0p5",
         "prob_ensemble"]
    ]
    new_out = new[
        ["subject_id", "cohort", "diagnosis", "age_at_MR", "sex_label",
         "session_day", "CDRTOT", "CDRSUM", "n_runs_qc_ok",
         "mean_fd", "max_fd", "pct_frames_fd_gt0p2", "pct_frames_fd_gt0p5",
         "prob_ensemble", "protocol_subset"]
    ]
    all_subjects = pd.concat([pilot_out, new_out], ignore_index=True)
    all_subjects.to_csv(OUT_DIR / "all_subjects.csv", index=False)

    # ── build summary.md ──────────────────────────────────────────────────────
    _write_summary(pilot, new, demo, cdr, motion, tests, fn_tp_summary, fn_tp_stats)

    # ── command log ───────────────────────────────────────────────────────────
    import datetime
    (OUT_DIR / "command_log.json").write_text(
        json.dumps(
            {
                "script": str(Path(__file__).name),
                "run_time": datetime.datetime.utcnow().isoformat() + "Z",
                "adni_fixed_threshold": ADNI_FIXED_THRESHOLD,
                "pilot_n": len(pilot),
                "new_n": len(new),
            },
            indent=2,
        )
    )
    print(f"\nDone. Outputs in {OUT_DIR}/")


def _write_summary(pilot, new, demo, cdr, motion, tests, fn_tp_sum, fn_tp_stats):
    lines = ["# OASIS Pilot vs New Cohort — Clinical/Motion Difficulty Audit", ""]
    lines += ["## 1. Cohort demographics", ""]
    lines += [md_table(demo), ""]

    # Age comparison highlight
    for dx in ["CN", "AD_DEMENTIA"]:
        pa = pilot[pilot.diagnosis == dx]["age_at_MR"]
        na = new[new.diagnosis == dx]["age_at_MR"]
        p_val = mannwhitney_p(pa, na)
        lines.append(
            f"  **{dx}** age: pilot {mean_sd(pa)} vs new {mean_sd(na)}"
            f" (MW p={fmt(p_val, 3)})"
        )
    lines.append("")

    lines += ["## 2. CDR distribution", ""]
    lines += [md_table(cdr), ""]
    lines.append(
        "Note: CDR reflects severity at time of MR session. "
        "New cohort CDR sourced from selection manifest (clinical_mapping_audit). "
        "Pilot CDR from DPABI run_manifest (OASIS UDSb4 closest visit)."
    )
    lines.append("")

    lines += ["## 3. Run counts", ""]
    new_runs = new.groupby("diagnosis")["n_runs_qc_ok"].value_counts().unstack(fill_value=0)
    pilot_runs = pilot.groupby("diagnosis")["n_runs_qc_ok"].value_counts().unstack(fill_value=0)
    lines.append(f"Pilot run counts per subject:\n```\n{pilot_runs.to_string()}\n```")
    lines.append(f"New run counts per subject:\n```\n{new_runs.to_string()}\n```")
    lines.append("")

    lines += ["## 4. Motion QC", ""]
    lines += [md_table(motion), ""]

    lines += ["## 5. Cross-cohort statistical tests (Mann-Whitney)", ""]
    lines += [md_table(tests), ""]

    lines += ["## 6. FN vs TP in new AD cohort", ""]
    lines += [
        f"ADNI fixed threshold: {ADNI_FIXED_THRESHOLD:.4f}  ",
        "",
        "**Summary by outcome:**",
        md_table(fn_tp_sum),
        "",
        "**Per-metric statistical tests (TP vs FN):**",
        md_table(fn_tp_stats),
        "",
    ]

    # Interpretation
    ad_new = new[new.diagnosis == "AD_DEMENTIA"].copy()
    ad_new["outcome"] = (ad_new["prob_ensemble"] >= ADNI_FIXED_THRESHOLD).map({True: "TP", False: "FN"})
    fn = ad_new[ad_new.outcome == "FN"]
    tp = ad_new[ad_new.outcome == "TP"]

    lines += ["### Interpretation notes", ""]

    # CDR comparison
    if fn["CDRTOT"].notna().any() and tp["CDRTOT"].notna().any():
        fn_cdr = fn["CDRTOT"].mean()
        tp_cdr = tp["CDRTOT"].mean()
        if fn_cdr < tp_cdr:
            lines.append(
                f"- FN AD subjects have **lower mean CDR-TOT** ({fn_cdr:.2f}) than TP ({tp_cdr:.2f}), "
                "consistent with milder disease at time of scan."
            )
        else:
            lines.append(
                f"- TP and FN AD subjects have similar mean CDR-TOT ({tp_cdr:.2f} vs {fn_cdr:.2f})."
            )

    # FD comparison
    if fn["mean_fd"].notna().any() and tp["mean_fd"].notna().any():
        fn_fd = fn["mean_fd"].mean()
        tp_fd = tp["mean_fd"].mean()
        if fn_fd > tp_fd * 1.1:
            lines.append(
                f"- FN subjects have **higher mean FD** ({fn_fd:.3f}) than TP ({tp_fd:.3f}), "
                "suggesting motion confounds classification."
            )
        elif tp_fd > fn_fd * 1.1:
            lines.append(
                f"- TP subjects have higher mean FD ({tp_fd:.3f}) than FN ({fn_fd:.3f}) — "
                "motion is not the primary driver of false negatives."
            )
        else:
            lines.append(
                f"- Motion is comparable between TP and FN groups (mean FD: {tp_fd:.3f} vs {fn_fd:.3f})."
            )

    # Pilot vs new AD CDR
    pa_cdr = pilot[pilot.diagnosis == "AD_DEMENTIA"]["CDRTOT"].mean()
    na_cdr = new[new.diagnosis == "AD_DEMENTIA"]["CDRTOT"].mean()
    if not pd.isna(pa_cdr) and not pd.isna(na_cdr):
        lines.append(
            f"- Pilot AD CDR-TOT mean = {pa_cdr:.2f}; new cohort AD CDR-TOT mean = {na_cdr:.2f}. "
            + ("New cohort has **milder AD** on average." if na_cdr < pa_cdr
               else "New cohort has similar or more severe AD.")
        )

    lines.append("")
    (OUT_DIR / "summary.md").write_text("\n".join(lines))
    print("  → summary.md written")


if __name__ == "__main__":
    main()
