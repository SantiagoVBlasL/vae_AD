#!/usr/bin/env python3
"""Read-only postmortem for OASIS 60CN/60AD external scoring.

This audit consumes completed OASIS tensor-build QC and frozen ADNI scoring
outputs. It does not train, tune, rebuild tensors, or select models.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

try:
    from scipy import stats
except Exception:  # pragma: no cover - scipy is expected, but keep audit robust.
    stats = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "revision_bspc_2026"

DEFAULT_SCORING_DIR = RESULTS_DIR / "oasis_next_60cn_60ad_external_scoring_20260530"
DEFAULT_TENSOR_DIR = RESULTS_DIR / "oasis_next_60cn_60ad_tensor_build_20260530"
DEFAULT_TENSOR_QC_DIR = RESULTS_DIR / "oasis_next_60cn_60ad_tensor_build_qc_20260530"
DEFAULT_HANDOFF_QC_DIR = RESULTS_DIR / "oasis_next_60cn_60ad_processed_aal3_handoff_qc_20260530"
DEFAULT_OUTPUT_DIR = RESULTS_DIR / "oasis_next_60cn_60ad_external_scoring_postmortem_20260530"

ADNI_TENSOR_PATH = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/"
    "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)
ADNI_METADATA_PATH = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/"
    "training_ready_metadata_v5_1_batch20260514b_no_pybandpass.csv"
)

TENSOR_FILES = {
    "concatenated_timeseries": "tensor_concatenated_timeseries.npz",
    "runwise_140TR_connectome_average": "tensor_runwise_140TR_connectome_average.npz",
    "runwise164_connectome_average": "tensor_runwise164_connectome_average.npz",
}
TENSOR_NAME_TO_BUILD = {v: k for k, v in TENSOR_FILES.items()}
CHANNELS = [
    "Pearson_Full_FisherZ_Signed",
    "Pearson_OMST_GCE_Signed_Weighted",
    "MI_KNN_Symmetric",
]
RNG_SEED = 20260530
MAX_KS_SAMPLE = 200_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--scoring-dir", type=Path, default=DEFAULT_SCORING_DIR)
    parser.add_argument("--tensor-dir", type=Path, default=DEFAULT_TENSOR_DIR)
    parser.add_argument("--tensor-qc-dir", type=Path, default=DEFAULT_TENSOR_QC_DIR)
    parser.add_argument("--handoff-qc-dir", type=Path, default=DEFAULT_HANDOFF_QC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--adni-tensor-path", type=Path, default=ADNI_TENSOR_PATH)
    parser.add_argument("--adni-metadata-path", type=Path, default=ADNI_METADATA_PATH)
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def md_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows).copy()
    lines = [
        "| " + " | ".join(map(str, view.columns)) + " |",
        "| " + " | ".join(["---"] * len(view.columns)) + " |",
    ]
    for _, row in view.iterrows():
        vals: List[str] = []
        for col in view.columns:
            val = row[col]
            if isinstance(val, (float, np.floating)):
                vals.append(f"{val:.8g}" if np.isfinite(val) else "")
            elif pd.isna(val):
                vals.append("")
            else:
                vals.append(str(val).replace("\n", " ").replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(df)} rows._")
    return "\n".join(lines) + "\n"


def write_table(outdir: Path, stem: str, df: pd.DataFrame, max_rows: int = 200) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def finite_quantiles(values: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(values, dtype=float).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "q01": np.nan,
            "q25": np.nan,
            "median": np.nan,
            "q75": np.nan,
            "q99": np.nan,
            "max": np.nan,
        }
    qs = np.quantile(arr, [0.01, 0.25, 0.5, 0.75, 0.99])
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "q01": float(qs[0]),
        "q25": float(qs[1]),
        "median": float(qs[2]),
        "q75": float(qs[3]),
        "q99": float(qs[4]),
        "max": float(np.max(arr)),
    }


def offdiag_flat(mats: np.ndarray) -> np.ndarray:
    n = mats.shape[-1]
    mask = ~np.eye(n, dtype=bool)
    return mats[..., mask].reshape(-1)


def binary_metrics(y_true: Iterable[int], y_score: Iterable[float], threshold: float) -> Dict[str, Any]:
    y = np.asarray(list(y_true), dtype=int)
    score = np.asarray(list(y_score), dtype=float)
    pred = (score >= float(threshold)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = safe_div(tp, tp + fn)
    spec = safe_div(tn, tn + fp)
    return {
        "n": int(len(y)),
        "n_cn": int((y == 0).sum()),
        "n_ad": int((y == 1).sum()),
        "threshold": float(threshold),
        "auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
        "reversed_score_auc": float(1.0 - roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "f1": safe_div(2 * tp, 2 * tp + fp + fn),
        "accuracy": safe_div(tp + tn, len(y)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def add_thresholded_predictions(pred: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    ens = pred[pred["fold"].astype(str).eq("ensemble")].copy()
    rows: List[pd.DataFrame] = []
    for _, m in metrics.iterrows():
        sub = ens[
            ens["model_label"].eq(m["model_label"])
            & ens["build_candidate"].eq(m["build_candidate"])
        ].copy()
        if sub.empty:
            continue
        sub["threshold_strategy"] = m["threshold_strategy"]
        sub["threshold"] = float(m["threshold"])
        sub["y_pred"] = (sub["y_score"].astype(float) >= float(m["threshold"])).astype(int)
        sub["error_type"] = np.select(
            [
                sub["y_true"].eq(0) & sub["y_pred"].eq(0),
                sub["y_true"].eq(0) & sub["y_pred"].eq(1),
                sub["y_true"].eq(1) & sub["y_pred"].eq(0),
                sub["y_true"].eq(1) & sub["y_pred"].eq(1),
            ],
            ["TN", "FP", "FN", "TP"],
            default="NA",
        )
        rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def compute_split_metrics(
    pred: pd.DataFrame,
    metrics_locked: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    ens = pred[pred["fold"].astype(str).eq("ensemble")].copy()
    rows: List[Dict[str, Any]] = []
    for _, m in metrics_locked.iterrows():
        sub = ens[
            ens["model_label"].eq(m["model_label"])
            & ens["build_candidate"].eq(m["build_candidate"])
        ]
        if sub.empty:
            continue
        row = {
            "split_subset": split_name,
            "model_label": m["model_label"],
            "build_candidate": m["build_candidate"],
            "threshold_strategy": m["threshold_strategy"],
            "calib_method": m.get("calib_method", ""),
            "auc_below_0p45": False,
        }
        row.update(binary_metrics(sub["y_true"], sub["y_score"], float(m["threshold"])))
        row["auc_below_0p45"] = bool(row["auc"] < 0.45) if np.isfinite(row["auc"]) else False
        row["source_threshold_from"] = "scoring_metrics_locked_test_csv"
        rows.append(row)
    return pd.DataFrame(rows)


def label_alignment_audit(
    pred_cal: pd.DataFrame,
    pred_test: pd.DataFrame,
    metadata_alignment: pd.DataFrame,
    subject_label_alignment: pd.DataFrame,
) -> pd.DataFrame:
    pred_all = pd.concat([pred_cal, pred_test], ignore_index=True)
    ens = pred_all[pred_all["fold"].astype(str).eq("ensemble")].copy()
    rows = []
    for (build, split), sub in ens.groupby(["build_candidate", "split_subset"], dropna=False):
        pred_subjects = set(sub["SubjectID"].astype(str))
        meta_sub = metadata_alignment[
            metadata_alignment["build_candidate"].astype(str).eq(str(build))
            & metadata_alignment["protocol_subset"].astype(str).eq(str(split))
        ]
        meta_subjects = set(meta_sub["SubjectID"].astype(str))
        tensor_name = next((k for k, v in TENSOR_NAME_TO_BUILD.items() if v == build), None)
        # TENSOR_NAME_TO_BUILD maps filename -> build; invert lookup.
        tensor_file = next((fname for b, fname in TENSOR_FILES.items() if b == build), None)
        tensor_sub = subject_label_alignment[
            subject_label_alignment["tensor_name"].astype(str).eq(str(tensor_file))
        ]
        tensor_subjects = set(tensor_sub["SubjectID"].astype(str))

        diag_map_ok = bool(
            sub.assign(expected_y=sub["diagnosis"].map({"CN": 0, "AD_DEMENTIA": 1}))
            .eval("expected_y == y_true")
            .all()
        )
        rows.append(
            {
                "build_candidate": build,
                "split_subset": split,
                "n_prediction_subjects": int(len(pred_subjects)),
                "n_metadata_subjects": int(len(meta_subjects)),
                "n_tensor_subjects_total": int(len(tensor_subjects)),
                "prediction_minus_metadata": ";".join(sorted(pred_subjects - meta_subjects)),
                "metadata_minus_prediction": ";".join(sorted(meta_subjects - pred_subjects)),
                "prediction_not_in_tensor": ";".join(sorted(pred_subjects - tensor_subjects)),
                "diagnosis_y_true_mapping_ok": diag_map_ok,
                "ad_dementia_encoded_positive": bool(
                    sub[sub["diagnosis"].eq("AD_DEMENTIA")]["y_true"].eq(1).all()
                ),
                "cn_encoded_negative": bool(sub[sub["diagnosis"].eq("CN")]["y_true"].eq(0).all()),
                "missing_age_n": int(sub["Age"].isna().sum()),
                "missing_sex_n": int(sub["Sex"].isna().sum()),
                "unknown_sex_n": int(sub["Sex"].astype(str).str.upper().eq("UNKNOWN").sum()),
            }
        )
    return pd.DataFrame(rows)


def load_subject_covariates(
    tensor_dir: Path,
    handoff_qc_dir: Path,
    metadata_alignment: pd.DataFrame,
) -> pd.DataFrame:
    manifest = pd.read_csv(tensor_dir / "subject_manifest.csv")
    run_selection = pd.read_csv(tensor_dir / "run_selection_used.csv")
    motion = pd.read_csv(handoff_qc_dir / "motion_qc_by_file.csv")

    run_agg = run_selection.groupby("subject_id", dropna=False).agg(
        selected_run_count=("run", "count"),
        selected_mean_fd_mean=("mean_fd_jenkinson", "mean"),
        selected_mean_fd_max=("mean_fd_jenkinson", "max"),
        selected_n_timepoints_total=("n_timepoints", "sum"),
    ).reset_index()
    motion_agg = motion.groupby("subject_id", dropna=False).agg(
        mean_fd_subject=("mean_fd_jenkinson", "mean"),
        max_fd_subject=("max_fd_jenkinson", "max"),
        pct_frames_fd_gt0p2_mean=("pct_frames_fd_gt0p2", "mean"),
        pct_frames_fd_gt0p5_mean=("pct_frames_fd_gt0p5", "mean"),
        severe_motion_run_n=("flag_severe_motion_fd", "sum"),
        mild_motion_run_n=("flag_mild_motion_fd", "sum"),
        martin_exclude_3mm3deg_run_n=("martins_pipeline_exclude_3mm3deg", "sum"),
    ).reset_index()
    meta_once = metadata_alignment.sort_values(["SubjectID", "protocol_subset"]).drop_duplicates("SubjectID")
    meta_once = meta_once.rename(columns={"SubjectID": "subject_id"})

    cov = manifest.merge(run_agg, on="subject_id", how="left").merge(motion_agg, on="subject_id", how="left")
    cov = cov.merge(
        meta_once[["subject_id", "Age", "Sex", "protocol_subset"]],
        on="subject_id",
        how="left",
        suffixes=("", "_alignment"),
    )
    return cov


def corr_pair(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    xx = pd.to_numeric(x, errors="coerce")
    yy = pd.to_numeric(y, errors="coerce")
    mask = xx.notna() & yy.notna()
    if int(mask.sum()) < 3 or xx[mask].nunique() < 2 or yy[mask].nunique() < 2:
        return {"n": int(mask.sum()), "pearson_r": np.nan, "pearson_p": np.nan, "spearman_r": np.nan, "spearman_p": np.nan}
    if stats is None:
        return {
            "n": int(mask.sum()),
            "pearson_r": float(np.corrcoef(xx[mask], yy[mask])[0, 1]),
            "pearson_p": np.nan,
            "spearman_r": float(pd.Series(xx[mask]).corr(pd.Series(yy[mask]), method="spearman")),
            "spearman_p": np.nan,
        }
    pr = stats.pearsonr(xx[mask], yy[mask])
    sr = stats.spearmanr(xx[mask], yy[mask])
    return {
        "n": int(mask.sum()),
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
        "spearman_r": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
    }


def score_correlates(pred_all: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    ens = pred_all[pred_all["fold"].astype(str).eq("ensemble")].copy()
    cov2 = cov.rename(columns={"subject_id": "SubjectID"})
    ens = ens.merge(cov2, on="SubjectID", how="left", suffixes=("", "_cov"))
    ens["sex_male"] = ens["Sex"].astype(str).str.upper().map({"M": 1, "F": 0})
    variables = [
        ("Age", "Age"),
        ("Sex_M_is_1", "sex_male"),
        ("mean_fd_subject", "mean_fd_subject"),
        ("max_fd_subject", "max_fd_subject"),
        ("selected_run_count", "selected_run_count"),
        ("diagnosis_AD_is_1", "y_true"),
    ]
    rows: List[Dict[str, Any]] = []
    for (build, model, split), sub in ens.groupby(["build_candidate", "model_label", "split_subset"], dropna=False):
        for label, col in variables:
            vals = corr_pair(sub[col], sub["y_score"]) if col in sub.columns else {
                "n": 0,
                "pearson_r": np.nan,
                "pearson_p": np.nan,
                "spearman_r": np.nan,
                "spearman_p": np.nan,
            }
            rows.append(
                {
                    "build_candidate": build,
                    "model_label": model,
                    "split_subset": split,
                    "variable": label,
                    **vals,
                }
            )
    return pd.DataFrame(rows)


def score_distribution(pred_all: pd.DataFrame, metrics_locked: pd.DataFrame) -> pd.DataFrame:
    ens = pred_all[pred_all["fold"].astype(str).eq("ensemble")].copy()
    threshold_lookup = metrics_locked.drop_duplicates(["model_label", "build_candidate"])[
        ["model_label", "build_candidate", "adni_threshold_ensemble", "oasis_cal_threshold"]
    ]
    rows: List[Dict[str, Any]] = []
    for (build, model, split, diagnosis), sub in ens.groupby(
        ["build_candidate", "model_label", "split_subset", "diagnosis"],
        dropna=False,
    ):
        q = finite_quantiles(sub["y_score"].to_numpy(float))
        thr = threshold_lookup[
            threshold_lookup["model_label"].eq(model)
            & threshold_lookup["build_candidate"].eq(build)
        ]
        adni_thr = float(thr["adni_threshold_ensemble"].iloc[0]) if not thr.empty else np.nan
        oasis_thr = float(thr["oasis_cal_threshold"].iloc[0]) if not thr.empty else np.nan
        rows.append(
            {
                "build_candidate": build,
                "model_label": model,
                "split_subset": split,
                "diagnosis": diagnosis,
                "n": int(len(sub)),
                **{f"score_{k}": v for k, v in q.items()},
                "adni_threshold": adni_thr,
                "oasis_calibration_threshold": oasis_thr,
                "fraction_ge_adni_threshold": float((sub["y_score"].astype(float) >= adni_thr).mean()) if np.isfinite(adni_thr) else np.nan,
                "fraction_ge_oasis_calibration_threshold": float((sub["y_score"].astype(float) >= oasis_thr).mean()) if np.isfinite(oasis_thr) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def plot_score_histograms(pred_all: pd.DataFrame, metrics_locked: pd.DataFrame, outdir: Path) -> None:
    ens = pred_all[pred_all["fold"].astype(str).eq("ensemble")].copy()
    plot_dir = outdir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    threshold_lookup = metrics_locked.drop_duplicates(["model_label", "build_candidate"])
    for (build, model), sub in ens.groupby(["build_candidate", "model_label"], dropna=False):
        thr = threshold_lookup[
            threshold_lookup["model_label"].eq(model)
            & threshold_lookup["build_candidate"].eq(build)
        ]
        adni_thr = float(thr["adni_threshold_ensemble"].iloc[0]) if not thr.empty else np.nan
        oasis_thr = float(thr["oasis_cal_threshold"].iloc[0]) if not thr.empty else np.nan
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, split in zip(axes, ["calibration", "locked_test"]):
            ss = sub[sub["split_subset"].eq(split)]
            for diagnosis, color in [("CN", "#377eb8"), ("AD_DEMENTIA", "#e41a1c")]:
                vals = ss[ss["diagnosis"].eq(diagnosis)]["y_score"].astype(float)
                ax.hist(vals, bins=np.linspace(0, 1, 21), alpha=0.45, label=diagnosis, color=color)
            if np.isfinite(adni_thr):
                ax.axvline(adni_thr, color="black", linestyle="--", linewidth=1.2, label="ADNI threshold")
            if np.isfinite(oasis_thr):
                ax.axvline(oasis_thr, color="purple", linestyle=":", linewidth=1.2, label="OASIS-cal threshold")
            ax.set_title(split)
            ax.set_xlabel("AD score")
            ax.set_ylabel("Subjects")
            ax.legend(fontsize=7)
        fig.suptitle(f"{model} | {build}", fontsize=10)
        fig.tight_layout()
        safe_name = f"score_hist_{model}_{build}".replace("/", "_")
        fig.savefig(plot_dir / f"{safe_name}.png", dpi=140)
        plt.close(fig)


def load_tensor(path: Path) -> Tuple[np.ndarray, List[str], np.ndarray]:
    with np.load(path, allow_pickle=True) as npz:
        tensor = np.asarray(npz["global_tensor_data"])
        channel_names = np.asarray(npz["channel_names"]).astype(str).tolist()
        subject_ids = np.asarray(npz["subject_ids"]).astype(str)
    return tensor, channel_names, subject_ids


def channel_distribution_comparison(
    adni_tensor_path: Path,
    adni_metadata_path: Path,
    tensor_dir: Path,
) -> pd.DataFrame:
    rng = np.random.default_rng(RNG_SEED)
    rows: List[Dict[str, Any]] = []

    adni_tensor, adni_channels, adni_subjects = load_tensor(adni_tensor_path)
    adni_meta = pd.read_csv(adni_metadata_path)
    adni_keep_all = pd.Series(adni_subjects).isin(adni_meta["SubjectID"].astype(str)).to_numpy()
    adni_dx_map = adni_meta.set_index("SubjectID")["ResearchGroup_Mapped"].astype(str).to_dict()
    adni_dx = np.asarray([adni_dx_map.get(str(s), "") for s in adni_subjects])
    adni_keep_clf = adni_keep_all & np.isin(adni_dx, ["CN", "AD"])
    adni_reference: Dict[str, np.ndarray] = {}

    for channel in CHANNELS:
        if channel not in adni_channels:
            continue
        idx = adni_channels.index(channel)
        for cohort_name, keep in [
            ("ADNI_training_ready_all_CN_MCI_AD", adni_keep_all),
            ("ADNI_classifier_CN_AD", adni_keep_clf),
        ]:
            vals = offdiag_flat(adni_tensor[keep, idx, :, :])
            stats_row = {
                "cohort": cohort_name,
                "build_candidate": "ADNI_v5_1b_140TR",
                "channel_name": channel,
                "n_subjects": int(keep.sum()),
                **finite_quantiles(vals),
            }
            rows.append(stats_row)
            if cohort_name == "ADNI_training_ready_all_CN_MCI_AD":
                adni_reference[channel] = vals[np.isfinite(vals)]

    for build, fname in TENSOR_FILES.items():
        tensor, channel_names, _subjects = load_tensor(tensor_dir / fname)
        for channel in CHANNELS:
            if channel not in channel_names:
                continue
            idx = channel_names.index(channel)
            vals = offdiag_flat(tensor[:, idx, :, :])
            q = finite_quantiles(vals)
            row: Dict[str, Any] = {
                "cohort": "OASIS_60CN_60AD",
                "build_candidate": build,
                "channel_name": channel,
                "n_subjects": int(tensor.shape[0]),
                **q,
            }
            ref = adni_reference.get(channel)
            if ref is not None and ref.size:
                finite_vals = vals[np.isfinite(vals)]
                row["mean_minus_adni_all"] = float(q["mean"] - np.mean(ref))
                row["std_ratio_vs_adni_all"] = float(q["std"] / np.std(ref)) if np.std(ref) else np.nan
                if stats is not None and finite_vals.size:
                    a = ref
                    b = finite_vals
                    if a.size > MAX_KS_SAMPLE:
                        a = rng.choice(a, size=MAX_KS_SAMPLE, replace=False)
                    if b.size > MAX_KS_SAMPLE:
                        b = rng.choice(b, size=MAX_KS_SAMPLE, replace=False)
                    ks = stats.ks_2samp(a, b)
                    row["ks_stat_vs_adni_all"] = float(ks.statistic)
                    row["ks_p_vs_adni_all"] = float(ks.pvalue)
            rows.append(row)
    return pd.DataFrame(rows)


def tensor_distribution_between_oasis_builds(tensor_dir: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    tensors: Dict[str, Tuple[np.ndarray, List[str]]] = {}
    for build, fname in TENSOR_FILES.items():
        tensor, channel_names, _subjects = load_tensor(tensor_dir / fname)
        tensors[build] = (tensor, channel_names)
    for a_name, b_name in [
        ("concatenated_timeseries", "runwise_140TR_connectome_average"),
        ("runwise164_connectome_average", "runwise_140TR_connectome_average"),
        ("concatenated_timeseries", "runwise164_connectome_average"),
    ]:
        a_tensor, a_channels = tensors[a_name]
        b_tensor, b_channels = tensors[b_name]
        for channel in CHANNELS:
            if channel not in a_channels or channel not in b_channels:
                continue
            a = a_tensor[:, a_channels.index(channel), :, :]
            b = b_tensor[:, b_channels.index(channel), :, :]
            diff = offdiag_flat(a - b)
            q = finite_quantiles(diff)
            rows.append(
                {
                    "comparison": f"{a_name}_minus_{b_name}",
                    "channel_name": channel,
                    **{f"diff_{k}": v for k, v in q.items()},
                    "abs_diff_mean": float(np.nanmean(np.abs(diff))),
                    "abs_diff_max": float(np.nanmax(np.abs(diff))),
                }
            )
    return pd.DataFrame(rows)


def motion_error_sensitivity(thresholded: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    df = thresholded[thresholded["split_subset"].eq("locked_test")].copy()
    df = df.merge(cov.rename(columns={"subject_id": "SubjectID"}), on="SubjectID", how="left", suffixes=("", "_cov"))
    rows: List[Dict[str, Any]] = []
    for fd_col in ["mean_fd_subject", "max_fd_subject"]:
        if fd_col not in df.columns:
            continue
        median_fd = float(pd.to_numeric(df[fd_col], errors="coerce").median())
        df[f"{fd_col}_group"] = np.where(pd.to_numeric(df[fd_col], errors="coerce") > median_fd, "high_fd", "low_fd")
        for (build, model, strategy, group), sub in df.groupby(
            ["build_candidate", "model_label", "threshold_strategy", f"{fd_col}_group"],
            dropna=False,
        ):
            y = sub["y_true"].astype(int)
            pred = sub["y_pred"].astype(int)
            fp = int(((y == 0) & (pred == 1)).sum())
            tn = int(((y == 0) & (pred == 0)).sum())
            fn = int(((y == 1) & (pred == 0)).sum())
            tp = int(((y == 1) & (pred == 1)).sum())
            rows.append(
                {
                    "build_candidate": build,
                    "model_label": model,
                    "threshold_strategy": strategy,
                    "fd_metric": fd_col,
                    "fd_group": group,
                    "fd_median_cut": median_fd,
                    "n": int(len(sub)),
                    "n_cn": int((y == 0).sum()),
                    "n_ad": int((y == 1).sum()),
                    "mean_fd_mean": float(pd.to_numeric(sub[fd_col], errors="coerce").mean()),
                    "error_rate": float((sub["error_type"].isin(["FP", "FN"])).mean()),
                    "fp_rate_among_cn": safe_div(fp, fp + tn),
                    "fn_rate_among_ad": safe_div(fn, fn + tp),
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp,
                }
            )
    return pd.DataFrame(rows)


def stable_errors(thresholded: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    df = thresholded[
        thresholded["split_subset"].eq("locked_test")
        & thresholded["threshold_strategy"].eq("adni_fixed")
    ].copy()
    cov_small = cov.rename(columns={"subject_id": "SubjectID"})
    df = df.merge(
        cov_small[["SubjectID", "age_at_MR", "sex", "Manufacturer", "ScannerModel", "selected_qc_runs", "mean_fd_subject", "max_fd_subject"]],
        on="SubjectID",
        how="left",
        suffixes=("", "_manifest"),
    )
    rows: List[Dict[str, Any]] = []
    for sid, sub in df.groupby("SubjectID", dropna=False):
        primary = sub[sub["model_label"].eq("v5_1b_locked_raw")]
        rows.append(
            {
                "SubjectID": sid,
                "diagnosis": sub["diagnosis"].iloc[0],
                "y_true": int(sub["y_true"].iloc[0]),
                "n_configs": int(len(sub)),
                "n_fp_all_configs": int(sub["error_type"].eq("FP").sum()),
                "n_fn_all_configs": int(sub["error_type"].eq("FN").sum()),
                "n_errors_all_configs": int(sub["error_type"].isin(["FP", "FN"]).sum()),
                "n_primary_builds": int(len(primary)),
                "primary_build_error_types": ";".join(primary.sort_values("build_candidate")["error_type"].astype(str)),
                "stable_primary_fp_all_3_builds": bool(len(primary) == 3 and primary["error_type"].eq("FP").all()),
                "stable_primary_fn_all_3_builds": bool(len(primary) == 3 and primary["error_type"].eq("FN").all()),
                "stable_error_all_9_configs": bool(len(sub) == 9 and sub["error_type"].isin(["FP", "FN"]).all()),
                "age_at_MR": sub["age_at_MR"].iloc[0],
                "sex": sub["sex"].iloc[0],
                "Manufacturer": sub["Manufacturer"].iloc[0],
                "ScannerModel": sub["ScannerModel"].iloc[0],
                "selected_qc_runs": sub["selected_qc_runs"].iloc[0],
                "mean_fd_subject": sub["mean_fd_subject"].iloc[0],
                "max_fd_subject": sub["max_fd_subject"].iloc[0],
            }
        )
    return pd.DataFrame(rows)


def calibration_vs_locked(metrics_all: pd.DataFrame) -> pd.DataFrame:
    base = metrics_all[metrics_all["threshold_strategy"].eq("adni_fixed")].copy()
    rows: List[Dict[str, Any]] = []
    for (build, model), sub in base.groupby(["build_candidate", "model_label"], dropna=False):
        cal = sub[sub["split_subset"].eq("calibration")]
        test = sub[sub["split_subset"].eq("locked_test")]
        if cal.empty or test.empty:
            continue
        c = cal.iloc[0]
        t = test.iloc[0]
        rows.append(
            {
                "build_candidate": build,
                "model_label": model,
                "calibration_auc": c["auc"],
                "locked_test_auc": t["auc"],
                "locked_minus_calibration_auc": t["auc"] - c["auc"],
                "calibration_pr_auc": c["pr_auc"],
                "locked_test_pr_auc": t["pr_auc"],
                "locked_minus_calibration_pr_auc": t["pr_auc"] - c["pr_auc"],
                "poor_performance_stable_both_auc_lt_0p55": bool(c["auc"] < 0.55 and t["auc"] < 0.55),
                "locked_auc_below_0p45": bool(t["auc"] < 0.45),
            }
        )
    return pd.DataFrame(rows)


def threshold_transfer_summary(distribution: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (build, model, split), sub in distribution.groupby(["build_candidate", "model_label", "split_subset"], dropna=False):
        cn = sub[sub["diagnosis"].eq("CN")]
        ad = sub[sub["diagnosis"].eq("AD_DEMENTIA")]
        rows.append(
            {
                "build_candidate": build,
                "model_label": model,
                "split_subset": split,
                "cn_median": float(cn["score_median"].iloc[0]) if not cn.empty else np.nan,
                "ad_median": float(ad["score_median"].iloc[0]) if not ad.empty else np.nan,
                "ad_minus_cn_median": float(ad["score_median"].iloc[0] - cn["score_median"].iloc[0]) if not cn.empty and not ad.empty else np.nan,
                "ad_fraction_ge_adni_threshold": float(ad["fraction_ge_adni_threshold"].iloc[0]) if not ad.empty else np.nan,
                "cn_fraction_ge_adni_threshold": float(cn["fraction_ge_adni_threshold"].iloc[0]) if not cn.empty else np.nan,
                "ad_fraction_ge_oasis_calibration_threshold": float(ad["fraction_ge_oasis_calibration_threshold"].iloc[0]) if not ad.empty else np.nan,
                "cn_fraction_ge_oasis_calibration_threshold": float(cn["fraction_ge_oasis_calibration_threshold"].iloc[0]) if not cn.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def make_recommendation(
    metrics_all: pd.DataFrame,
    alignment: pd.DataFrame,
    tensor_dist: pd.DataFrame,
    motion_sens: pd.DataFrame,
) -> Tuple[str, str]:
    locked = metrics_all[
        metrics_all["split_subset"].eq("locked_test")
        & metrics_all["threshold_strategy"].eq("adni_fixed")
    ]
    concat = locked[locked["build_candidate"].eq("concatenated_timeseries")]
    runwise = locked[locked["build_candidate"].isin(["runwise_140TR_connectome_average", "runwise164_connectome_average"])]
    any_alignment_fail = not alignment[
        [
            "diagnosis_y_true_mapping_ok",
            "ad_dementia_encoded_positive",
            "cn_encoded_negative",
        ]
    ].all().all()
    concat_below = bool((concat["auc"] < 0.45).any())
    runwise_best = float(runwise["auc"].max()) if not runwise.empty else np.nan
    best_auc = float(locked["auc"].max()) if not locked.empty else np.nan

    if any_alignment_fail:
        decision = "possible_preprocessing/tensor_mismatch"
        rationale = "Subject/label alignment checks did not all pass; do not interpret transfer metrics until alignment is resolved."
    elif concat_below and np.isfinite(runwise_best) and runwise_best > 0.50:
        decision = "needs_tensor_rebuild_check"
        rationale = (
            "The concatenated-timeseries build is below chance for at least one model while runwise builds are weakly above chance. "
            "This pattern is not a global label-polarity inversion; it points to a run-handling or preprocessing/tensor compatibility issue for the concatenated build, plus broader external domain shift."
        )
    elif np.isfinite(best_auc) and best_auc < 0.60:
        decision = "external_transfer_failed_due_to_domain_shift"
        rationale = (
            "All locked-test AUCs remain weak even when thresholding is ignored. "
            "This is not primarily a threshold-only failure."
        )
    else:
        decision = "ready_for_FD_sensitivity_scoring"
        rationale = (
            "Ranking signal is present and alignment passed; motion/FD sensitivity can be scored descriptively without model selection."
        )

    if not motion_sens.empty:
        fd_note = " Motion/error stratification was computed; high-FD sensitivity should be interpreted descriptively only."
        rationale += fd_note
    return decision, rationale


def main() -> int:
    args = parse_args()
    outdir = args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    required = {
        "predictions_locked_test": args.scoring_dir / "predictions_locked_test.csv",
        "predictions_calibration": args.scoring_dir / "predictions_calibration.csv",
        "metrics_locked_test": args.scoring_dir / "metrics_locked_test.csv",
        "metadata_alignment_audit": args.scoring_dir / "metadata_alignment_audit.csv",
        "subject_label_alignment": args.tensor_qc_dir / "subject_label_alignment.csv",
        "channel_statistics": args.tensor_qc_dir / "channel_statistics.csv",
        "tensor_qc_summary": args.tensor_qc_dir / "tensor_qc_summary.csv",
        "subject_manifest": args.tensor_dir / "subject_manifest.csv",
        "run_selection_used": args.tensor_dir / "run_selection_used.csv",
        "motion_qc_by_file": args.handoff_qc_dir / "motion_qc_by_file.csv",
        "adni_tensor": args.adni_tensor_path,
        "adni_metadata": args.adni_metadata_path,
    }
    for label, path in required.items():
        require(path, label)

    pred_test = pd.read_csv(required["predictions_locked_test"])
    pred_cal = pd.read_csv(required["predictions_calibration"])
    metrics_locked = pd.read_csv(required["metrics_locked_test"])
    metadata_alignment = pd.read_csv(required["metadata_alignment_audit"])
    subject_label_alignment = pd.read_csv(required["subject_label_alignment"])
    tensor_qc_summary = pd.read_csv(required["tensor_qc_summary"])
    channel_stats_existing = pd.read_csv(required["channel_statistics"])

    metrics_cal = compute_split_metrics(pred_cal, metrics_locked, "calibration")
    metrics_test = compute_split_metrics(pred_test, metrics_locked, "locked_test")
    metrics_all = pd.concat([metrics_cal, metrics_test], ignore_index=True)
    write_table(outdir, "metrics_all_splits", metrics_all)

    reversed_flags = metrics_all[
        ["split_subset", "model_label", "build_candidate", "threshold_strategy", "auc", "reversed_score_auc", "auc_below_0p45"]
    ].copy()
    reversed_flags["possible_polarity_concern"] = reversed_flags["auc_below_0p45"] & (reversed_flags["reversed_score_auc"] > 0.55)
    write_table(outdir, "reversed_auc_flags", reversed_flags)

    cal_vs_test = calibration_vs_locked(metrics_all)
    write_table(outdir, "calibration_vs_locked_test_auc", cal_vs_test)

    pred_all = pd.concat([pred_cal, pred_test], ignore_index=True)
    distribution = score_distribution(pred_all, metrics_locked)
    write_table(outdir, "score_distribution_by_diagnosis", distribution)
    write_table(outdir, "threshold_transfer_summary", threshold_transfer_summary(distribution))
    plot_score_histograms(pred_all, metrics_locked, outdir)

    alignment = label_alignment_audit(pred_cal, pred_test, metadata_alignment, subject_label_alignment)
    write_table(outdir, "label_polarity_metadata_alignment", alignment)

    tensor_dist = channel_distribution_comparison(args.adni_tensor_path, args.adni_metadata_path, args.tensor_dir)
    write_table(outdir, "tensor_distribution_comparison_adni_oasis", tensor_dist)
    write_table(outdir, "oasis_build_pairwise_tensor_differences", tensor_distribution_between_oasis_builds(args.tensor_dir))
    write_table(outdir, "existing_tensor_qc_summary_copy", tensor_qc_summary)
    write_table(outdir, "existing_channel_statistics_copy", channel_stats_existing)

    cov = load_subject_covariates(args.tensor_dir, args.handoff_qc_dir, metadata_alignment)
    write_table(outdir, "subject_motion_covariates", cov)

    score_corr = score_correlates(pred_all, cov)
    write_table(outdir, "score_correlates", score_corr)

    thresholded_test = add_thresholded_predictions(pred_test, metrics_locked)
    write_table(outdir, "locked_test_thresholded_predictions", thresholded_test, max_rows=400)

    motion_sens = motion_error_sensitivity(thresholded_test, cov)
    write_table(outdir, "motion_error_sensitivity", motion_sens)

    stable = stable_errors(thresholded_test, cov)
    write_table(outdir, "stable_error_subjects", stable, max_rows=400)
    stable_fpfn = stable[
        stable["stable_primary_fp_all_3_builds"] | stable["stable_primary_fn_all_3_builds"] | stable["stable_error_all_9_configs"]
    ].copy()
    write_table(outdir, "stable_false_positive_negative_subjects", stable_fpfn, max_rows=400)

    decision, rationale = make_recommendation(metrics_all, alignment, tensor_dist, motion_sens)
    best_rows = metrics_all[
        metrics_all["split_subset"].eq("locked_test")
        & metrics_all["threshold_strategy"].eq("adni_fixed")
    ].sort_values("auc", ascending=False)

    recommendation = [
        "# Final Recommendation",
        "",
        f"Decision: `{decision}`",
        "",
        rationale,
        "",
        "This postmortem is read-only. It performs no training, no threshold fitting for locked-test evaluation, no model selection, and no tensor modification.",
        "",
        "## Key Findings",
        "",
        f"- Best locked-test ADNI-fixed AUC observed in this audit: {best_rows['auc'].iloc[0]:.4f} "
        f"({best_rows['model_label'].iloc[0]} / {best_rows['build_candidate'].iloc[0]}).",
        f"- Concatenated-timeseries locked-test AUC range: {best_rows[best_rows['build_candidate'].eq('concatenated_timeseries')]['auc'].min():.4f} to "
        f"{best_rows[best_rows['build_candidate'].eq('concatenated_timeseries')]['auc'].max():.4f}.",
        f"- Runwise locked-test AUC range: {best_rows[best_rows['build_candidate'].ne('concatenated_timeseries')]['auc'].min():.4f} to "
        f"{best_rows[best_rows['build_candidate'].ne('concatenated_timeseries')]['auc'].max():.4f}.",
        f"- Label polarity/alignment checks passed: {bool(alignment[['diagnosis_y_true_mapping_ok', 'ad_dementia_encoded_positive', 'cn_encoded_negative']].all().all())}.",
        f"- Age/Sex missing in scoring metadata: Age={int(metadata_alignment['Age'].isna().sum())}, Sex={int(metadata_alignment['Sex'].isna().sum())}.",
        "",
        "## Interpretation",
        "",
        "The OASIS locked-test scores do not support OASIS-based model promotion. "
        "Below-chance concatenated-timeseries ranking alongside weakly above-chance runwise ranking argues against a simple global label polarity inversion. "
        "The most conservative interpretation is external domain shift with a specific run-handling/tensor compatibility concern for the concatenated build. "
        "Any FD or threshold recalibration analysis should remain explicitly external-calibration-only and should not alter ADNI model selection.",
        "",
    ]
    (outdir / "final_recommendation.md").write_text("\n".join(recommendation), encoding="utf-8")

    readme = [
        "# OASIS Next 60CN/60AD External Scoring Postmortem",
        "",
        "Read-only audit of frozen ADNI model transfer to the OASIS next-batch calibration/test split.",
        "",
        "Generated outputs:",
        "- `metrics_all_splits.csv/.md`",
        "- `reversed_auc_flags.csv/.md`",
        "- `calibration_vs_locked_test_auc.csv/.md`",
        "- `score_distribution_by_diagnosis.csv/.md` and `figures/score_hist_*.png`",
        "- `label_polarity_metadata_alignment.csv/.md`",
        "- `tensor_distribution_comparison_adni_oasis.csv/.md`",
        "- `score_correlates.csv/.md`",
        "- `motion_error_sensitivity.csv/.md`",
        "- `stable_false_positive_negative_subjects.csv/.md`",
        "- `final_recommendation.md`",
        "",
        "Guardrails: no training, no model selection, no OASIS-based promotion, no tensor modification.",
        "",
    ]
    (outdir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    write_json(
        outdir / "command_log.json",
        {
            "script": str(Path(__file__).resolve()),
            "timestamp_utc": now_utc(),
            "mode": "read_only_postmortem",
            "inputs": {k: str(v) for k, v in required.items()},
            "output_dir": str(outdir),
            "decision": decision,
            "no_training": True,
            "no_model_selection": True,
            "no_tensor_modification": True,
        },
    )
    print(f"Done. Outputs written to: {outdir}")
    print(f"Decision: {decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
