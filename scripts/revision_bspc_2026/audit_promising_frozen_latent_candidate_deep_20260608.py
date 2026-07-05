#!/usr/bin/env python3
"""Focused audit of promising frozen-latent correction candidates.

Reads the completed score-geometry correction batch and existing reference OOF
prediction files. It performs descriptive comparisons only: no VAE training, no
OASIS scoring, no threshold fitting on outer test labels, and no artifact
overwrite.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results/revision_bspc_2026"
INPUT_DEFAULT = RESULTS / "promoted_frozen_latent_score_geometry_correction_batch_20260608"
OUT_DEFAULT = RESULTS / "promising_frozen_latent_candidate_deep_audit_20260608"

PROMOTED_OOF = RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration/calib_predictions.csv"
CH1_OOF = RESULTS / "recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5_stageB_oof_score_calibration/calib_predictions.csv"
PROMOTED_CACHE = RESULTS / "recover035_latent384_beta3p75_T80_h10000_p560_full5x5/classifier_only_readout/latent_cache"

PRIMARY_MODEL = "logreg_l2_original"
PRIMARY_FEATURES = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"

FOCUS_CANDIDATES = [
    "promoted_plus_ch1_constrained_score_readout",
    "promoted_latent_mfr_direction_removal",
]
REFERENCE_PROMOTED = "reference_promoted_ch102_latent384_beta3p75"
REFERENCE_CH1 = "reference_ch1only_latent384_beta3p75"
MODEL_ORDER = [
    REFERENCE_PROMOTED,
    REFERENCE_CH1,
    "promoted_plus_ch1_constrained_score_readout",
    "promoted_latent_mfr_direction_removal",
]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def run_cmd(cmd: Sequence[str]) -> dict[str, Any]:
    proc = subprocess.run(
        list(cmd),
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {"cmd": list(cmd), "returncode": int(proc.returncode), "stdout": proc.stdout, "stderr": proc.stderr}


def md_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df.empty:
        return "_No rows._\n"
    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.6g}")
    text = view.to_markdown(index=False)
    if len(df) > max_rows:
        text += f"\n\n_Showing {max_rows} of {len(df)} rows; full table is in CSV._"
    return text + "\n"


def write_table(outdir: Path, stem: str, df: pd.DataFrame, max_rows: int = 200) -> None:
    df.to_csv(outdir / f"{stem}.csv", index=False)
    (outdir / f"{stem}.md").write_text(md_table(df, max_rows=max_rows), encoding="utf-8")


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def primary_oof(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    mask = (
        df["model_name"].astype(str).eq(PRIMARY_MODEL)
        & df["feature_set"].astype(str).eq(PRIMARY_FEATURES)
        & df["calib_method"].astype(str).eq(PRIMARY_CALIB)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    )
    out = df.loc[mask].copy()
    if out.empty:
        raise ValueError(f"No primary rows found in {path}")
    out["SubjectID"] = out["SubjectID"].astype(str)
    return out


def load_model_predictions(input_dir: Path) -> pd.DataFrame:
    err = pd.read_csv(input_dir / "promoted_vs_candidate_subject_errors.csv")
    err["SubjectID"] = err["SubjectID"].astype(str)
    parts: list[pd.DataFrame] = []
    base = err[err["candidate_id"].eq(FOCUS_CANDIDATES[0])].copy()
    promoted = base[
        ["SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex", "fold", "y_true", "promoted_score", "promoted_pred"]
    ].rename(columns={"promoted_score": "score", "promoted_pred": "pred"})
    promoted["model_id"] = REFERENCE_PROMOTED
    parts.append(promoted)

    ch1 = primary_oof(CH1_OOF)
    ch1 = ch1[
        ["SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex", "fold", "y_true", "y_score", "y_pred"]
    ].rename(columns={"y_score": "score", "y_pred": "pred"})
    ch1["model_id"] = REFERENCE_CH1
    parts.append(ch1)

    for cand in FOCUS_CANDIDATES:
        sub = err[err["candidate_id"].eq(cand)].copy()
        pred = sub[
            ["SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Age", "Sex", "fold", "y_true", "candidate_score", "candidate_pred"]
        ].rename(columns={"candidate_score": "score", "candidate_pred": "pred"})
        pred["model_id"] = cand
        parts.append(pred)

    all_pred = pd.concat(parts, ignore_index=True)
    all_pred["y_true"] = pd.to_numeric(all_pred["y_true"], errors="raise").astype(int)
    all_pred["pred"] = pd.to_numeric(all_pred["pred"], errors="raise").astype(int)
    all_pred["score"] = pd.to_numeric(all_pred["score"], errors="raise")
    all_pred["Age"] = pd.to_numeric(all_pred["Age"], errors="coerce")
    return all_pred


def metric_summary(pred: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id, sub in pred.groupby("model_id", sort=False):
        y = sub["y_true"].to_numpy(dtype=int)
        score = sub["score"].to_numpy(dtype=float)
        p = sub["pred"].to_numpy(dtype=int)
        tn, fp, fn, tp = confusion_matrix(y, p, labels=[0, 1]).ravel()
        rows.append(
            {
                "model_id": model_id,
                "n": int(len(sub)),
                "n_cn": int((y == 0).sum()),
                "n_ad": int((y == 1).sum()),
                "auc": float(roc_auc_score(y, score)),
                "pr_auc": float(average_precision_score(y, score)),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "balanced_accuracy": 0.5 * (safe_div(tp, tp + fn) + safe_div(tn, tn + fp)),
                "sensitivity": safe_div(tp, tp + fn),
                "specificity": safe_div(tn, tn + fp),
                "f1": safe_div(2 * tp, 2 * tp + fp + fn),
            }
        )
    return pd.DataFrame(rows)


def roc_pr_curves(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    roc_rows: list[dict[str, Any]] = []
    pr_rows: list[dict[str, Any]] = []
    for model_id, sub in pred.groupby("model_id", sort=False):
        y = sub["y_true"].to_numpy(dtype=int)
        score = sub["score"].to_numpy(dtype=float)
        fpr, tpr, thr = roc_curve(y, score)
        for i, (a, b, c) in enumerate(zip(fpr, tpr, thr)):
            roc_rows.append({"model_id": model_id, "point_idx": i, "fpr": a, "tpr": b, "threshold": c})
        prec, rec, pr_thr = precision_recall_curve(y, score)
        thresholds = list(pr_thr) + [np.nan]
        for i, (p, r, t) in enumerate(zip(prec, rec, thresholds)):
            pr_rows.append({"model_id": model_id, "point_idx": i, "precision": p, "recall": r, "threshold": t})
    return pd.DataFrame(roc_rows), pd.DataFrame(pr_rows)


def fixed_pr_operating_points(pred: pd.DataFrame) -> pd.DataFrame:
    recall_levels = [0.50, 0.60, 0.70, 0.80]
    precision_levels = [0.30, 0.40, 0.50, 0.60]
    rows: list[dict[str, Any]] = []
    for model_id, sub in pred.groupby("model_id", sort=False):
        y = sub["y_true"].to_numpy(dtype=int)
        score = sub["score"].to_numpy(dtype=float)
        precision, recall, _ = precision_recall_curve(y, score)
        for level in recall_levels:
            eligible = precision[recall >= level]
            rows.append(
                {
                    "model_id": model_id,
                    "operating_query": f"precision_at_recall_ge_{level:.2f}",
                    "target": level,
                    "value": float(np.max(eligible)) if len(eligible) else float("nan"),
                }
            )
        for level in precision_levels:
            eligible = recall[precision >= level]
            rows.append(
                {
                    "model_id": model_id,
                    "operating_query": f"recall_at_precision_ge_{level:.2f}",
                    "target": level,
                    "value": float(np.max(eligible)) if len(eligible) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def topk_enrichment(pred: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model_id, sub in pred.groupby("model_id", sort=False):
        prevalence = float(sub["y_true"].mean())
        ranked = sub.sort_values("score", ascending=False)
        for k in [20, 40, 60, 80]:
            top = ranked.head(k)
            n_ad = int(top["y_true"].sum())
            rows.append(
                {
                    "model_id": model_id,
                    "k": k,
                    "ad_in_top_k": n_ad,
                    "cn_in_top_k": int(k - n_ad),
                    "precision_at_k": safe_div(n_ad, k),
                    "lift_vs_prevalence": safe_div(safe_div(n_ad, k), prevalence),
                    "philips_cn_in_top_k": int(((top["y_true"] == 0) & top["Manufacturer"].eq("Philips")).sum()),
                    "ge_ad_in_top_k": int(((top["y_true"] == 1) & top["Manufacturer"].eq("GE")).sum()),
                }
            )
    return pd.DataFrame(rows)


def score_distribution(pred: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, sub in pred.groupby(["model_id", "ResearchGroup_Mapped", "Manufacturer"], dropna=False):
        model_id, dx, mfr = keys
        s = sub["score"].to_numpy(dtype=float)
        rows.append(
            {
                "model_id": model_id,
                "diagnosis": dx,
                "Manufacturer": mfr,
                "n": int(len(sub)),
                "mean": float(np.mean(s)),
                "std": float(np.std(s, ddof=1)) if len(s) > 1 else float("nan"),
                "median": float(np.median(s)),
                "p10": float(np.percentile(s, 10)),
                "p25": float(np.percentile(s, 25)),
                "p75": float(np.percentile(s, 75)),
                "p90": float(np.percentile(s, 90)),
                "predicted_ad_rate": float(sub["pred"].mean()),
            }
        )
    return pd.DataFrame(rows)


def subject_error_summaries(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    err = pd.read_csv(input_dir / "promoted_vs_candidate_subject_errors.csv")
    err = err[err["candidate_id"].isin(FOCUS_CANDIDATES)].copy()
    err["age_bin"] = pd.cut(
        pd.to_numeric(err["Age"], errors="coerce"),
        bins=[0, 65, 75, 85, 120],
        labels=["<65", "65-75", "75-85", "85+"],
        include_lowest=True,
    )
    detail = err.copy()
    rows: list[dict[str, Any]] = []
    group_sets = {
        "overall": ["candidate_id", "error_change"],
        "diagnosis": ["candidate_id", "error_change", "ResearchGroup_Mapped"],
        "manufacturer": ["candidate_id", "error_change", "Manufacturer"],
        "sex": ["candidate_id", "error_change", "Sex"],
        "age_bin": ["candidate_id", "error_change", "age_bin"],
        "fold": ["candidate_id", "error_change", "fold"],
    }
    for context, cols in group_sets.items():
        for keys, sub in err.groupby(cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {"context": context, "n": int(len(sub)), "age_mean": float(pd.to_numeric(sub["Age"], errors="coerce").mean())}
            row.update(dict(zip(cols, keys)))
            rows.append(row)
    return pd.DataFrame(rows), detail


def load_train_fold_with_scores(fold: int, promoted_scores: pd.DataFrame, ch1_scores: pd.DataFrame) -> pd.DataFrame:
    path = PROMOTED_CACHE / f"fold_{fold}_trainDev_latent_mu.csv"
    df = pd.read_csv(path)
    df["SubjectID"] = df["SubjectID"].astype(str)
    df = df[df["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
    df["y"] = df["ResearchGroup_Mapped"].map({"CN": 0, "AD": 1}).astype(int)
    df["Age"] = pd.to_numeric(df["Age"], errors="raise")
    df["Sex_M"] = df["Sex"].astype(str).str.upper().str.startswith("M").astype(float)
    df = df.merge(promoted_scores[["SubjectID", "promoted_oof_ecdf_score"]], on="SubjectID", how="left")
    df = df.merge(ch1_scores[["SubjectID", "ch1_oof_ecdf_score"]], on="SubjectID", how="left")
    if df[["promoted_oof_ecdf_score", "ch1_oof_ecdf_score"]].isna().any().any():
        raise ValueError(f"Missing OOF score features for fold {fold}")
    return df.reset_index(drop=True)


def plus_ch1_coefficients(input_dir: Path) -> pd.DataFrame:
    selected = pd.read_csv(input_dir / "selected_hyperparameters.csv")
    selected = selected[selected["candidate_id"].eq("promoted_plus_ch1_constrained_score_readout")].copy()
    promoted_oof = primary_oof(PROMOTED_OOF)[["SubjectID", "y_score"]].rename(columns={"y_score": "promoted_oof_ecdf_score"})
    ch1_oof = primary_oof(CH1_OOF)[["SubjectID", "y_score"]].rename(columns={"y_score": "ch1_oof_ecdf_score"})
    rows: list[dict[str, Any]] = []
    for _, hp in selected.iterrows():
        fold = int(hp["fold"])
        params = json.loads(hp["selected_params_json"])
        c = float(params["C"])
        train = load_train_fold_with_scores(fold, promoted_oof, ch1_oof)
        features = ["promoted_oof_ecdf_score", "ch1_oof_ecdf_score", "Age", "Sex_M"]
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=c, class_weight="balanced", solver="liblinear", max_iter=10000, random_state=42)),
            ]
        )
        model.fit(train[features], train["y"].to_numpy(dtype=int))
        coef = model.named_steps["model"].coef_.ravel()
        row = {
            "fold": fold,
            "selected_C": c,
            "intercept": float(model.named_steps["model"].intercept_[0]),
        }
        row.update({f"coef_standardized_{name}": float(value) for name, value in zip(features, coef)})
        row["abs_promoted_score_coef"] = abs(row["coef_standardized_promoted_oof_ecdf_score"])
        row["abs_ch1_score_coef"] = abs(row["coef_standardized_ch1_oof_ecdf_score"])
        row["dominant_score_source"] = (
            "promoted_score"
            if row["abs_promoted_score_coef"] > row["abs_ch1_score_coef"]
            else "ch1_score"
            if row["abs_ch1_score_coef"] > row["abs_promoted_score_coef"]
            else "tie"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def direction_k_audit(input_dir: Path) -> pd.DataFrame:
    selected = pd.read_csv(input_dir / "selected_hyperparameters.csv")
    grid = pd.read_csv(input_dir / "inner_grid_search_results.csv")
    selected = selected[selected["candidate_id"].eq("promoted_latent_mfr_direction_removal")].copy()
    rows: list[dict[str, Any]] = []
    for _, hp in selected.iterrows():
        fold = int(hp["fold"])
        params = json.loads(hp["selected_params_json"])
        sub = grid[(grid["candidate_id"].eq("promoted_latent_mfr_direction_removal")) & (grid["fold"].eq(fold))].copy()
        sub["params"] = sub["params_json"].map(json.loads)
        sub["k"] = sub["params"].map(lambda d: d.get("k"))
        by_k = sub.groupby("k", dropna=False).agg(best_inner_auc=("inner_oof_auc", "max"), best_inner_pr_auc=("inner_oof_pr_auc", "max")).reset_index()
        for _, krow in by_k.iterrows():
            rows.append(
                {
                    "fold": fold,
                    "selected_k": params.get("k"),
                    "selected_C": params.get("C"),
                    "k": int(krow["k"]),
                    "best_inner_auc_for_k": float(krow["best_inner_auc"]),
                    "best_inner_pr_auc_for_k": float(krow["best_inner_pr_auc"]),
                    "actual_direction_removal_selected": bool(params.get("k", 0) > 0),
                }
            )
    return pd.DataFrame(rows)


def manufacturer_operating_profile(pred: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, sub in pred.groupby(["model_id", "Manufacturer"], dropna=False):
        model_id, mfr = keys
        cn = sub[sub["y_true"] == 0]
        ad = sub[sub["y_true"] == 1]
        rows.append(
            {
                "model_id": model_id,
                "Manufacturer": mfr,
                "n_cn": int(len(cn)),
                "cn_fp": int((cn["pred"] == 1).sum()),
                "cn_fpr": safe_div((cn["pred"] == 1).sum(), len(cn)),
                "n_ad": int(len(ad)),
                "ad_fn": int((ad["pred"] == 0).sum()),
                "ad_fnr": safe_div((ad["pred"] == 0).sum(), len(ad)),
            }
        )
    return pd.DataFrame(rows)


def plus_ch1_high_scoring_cn(pred: pd.DataFrame) -> pd.DataFrame:
    sub = pred[pred["model_id"].eq("promoted_plus_ch1_constrained_score_readout")].copy()
    cn = sub[sub["y_true"].eq(0)].copy()
    return cn.sort_values("score", ascending=False).head(60)


def write_recommendation(outdir: Path, metrics: pd.DataFrame, coeffs: pd.DataFrame, k_audit: pd.DataFrame, mfr: pd.DataFrame) -> None:
    m = metrics.set_index("model_id")
    plus = m.loc["promoted_plus_ch1_constrained_score_readout"]
    dirs = m.loc["promoted_latent_mfr_direction_removal"]
    prom = m.loc[REFERENCE_PROMOTED]
    ch1 = m.loc[REFERENCE_CH1]
    all_k_zero = bool(k_audit["selected_k"].fillna(0).eq(0).all())
    promoted_dom = int((coeffs["dominant_score_source"] == "promoted_score").sum())
    ch1_dom = int((coeffs["dominant_score_source"] == "ch1_score").sum())
    lines = [
        "# Recommendation",
        "",
        "Decision: **supplementary sensitivity only; neither candidate should replace the promoted model.**",
        "",
        "The two-source constrained readout is the strongest high-AUC sensitivity: "
        f"AUC={plus['auc']:.6f} versus promoted {prom['auc']:.6f} and ch1-only {ch1['auc']:.6f}. "
        f"However PR-AUC={plus['pr_auc']:.6f} falls below the promoted PR-AUC={prom['pr_auc']:.6f}, so it does not meet the primary correction gate.",
        "",
        "The latent manufacturer-direction candidate is also sensitivity-only: "
        f"AUC={dirs['auc']:.6f}, PR-AUC={dirs['pr_auc']:.6f}, Philips CN FPR improved, but BA/F1 and GE AD FNR do not cleanly improve the promoted operating profile.",
        "",
    ]
    if all_k_zero:
        lines.append(
            "For `promoted_latent_mfr_direction_removal`, selected k was 0 in every fold. Therefore no actual latent manufacturer direction was removed; the observed changes are attributable to the fold-local logreg re-readout/regularization and OOF calibration, not nuisance-direction removal."
        )
    else:
        lines.append("At least one fold selected k>0, so the candidate includes actual nuisance-direction removal in those folds.")
    lines.extend(
        [
            "",
            f"For the two-source meta-logreg, promoted-score coefficient magnitude dominated in {promoted_dom}/5 folds and ch1-score magnitude dominated in {ch1_dom}/5 folds.",
            "",
            "Recommended reporting:",
            "- Report `promoted_plus_ch1_constrained_score_readout` as a supplementary high-AUC frozen-readout sensitivity only.",
            "- Report `promoted_latent_mfr_direction_removal` only as a negative/neutral nuisance-removal sensitivity, because k=0 means no true direction removal was selected.",
            "- OASIS scoring is optional external stress-test only; it should not be used to reopen model selection.",
            "- These results do not motivate a new FULL model because the correction is either score/readout-only or failed to select an actual nuisance-removal transform.",
        ]
    )
    (outdir / "final_recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    command_log: list[dict[str, Any]] = [
        {
            "timestamp": now_iso(),
            "event": "start",
            "input_dir": str(args.input_dir),
            "output_dir": str(out),
            "guardrails": [
                "no VAE training",
                "no OASIS scoring",
                "no outer-test threshold fitting",
                "no tensor/metadata/model artifact modification",
            ],
        }
    ]

    required = [
        args.input_dir / "promoted_vs_candidate_subject_errors.csv",
        args.input_dir / "selected_hyperparameters.csv",
        args.input_dir / "inner_grid_search_results.csv",
        PROMOTED_OOF,
        CH1_OOF,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    pred = load_model_predictions(args.input_dir)
    metrics = metric_summary(pred)
    roc_tbl, pr_tbl = roc_pr_curves(pred)
    fixed_pr = fixed_pr_operating_points(pred)
    topk = topk_enrichment(pred)
    score_dist = score_distribution(pred)
    err_summary, err_detail = subject_error_summaries(args.input_dir)
    coeffs = plus_ch1_coefficients(args.input_dir)
    k_audit = direction_k_audit(args.input_dir)
    mfr = manufacturer_operating_profile(pred)
    high_cn = plus_ch1_high_scoring_cn(pred)

    write_table(out, "candidate_auc_pr_tradeoff", metrics)
    write_table(out, "roc_curve_points", roc_tbl, max_rows=400)
    write_table(out, "pr_curve_points", pr_tbl, max_rows=400)
    write_table(out, "precision_recall_fixed_levels", fixed_pr)
    write_table(out, "topk_ad_enrichment", topk)
    write_table(out, "score_distribution_by_diagnosis_manufacturer", score_dist)
    write_table(out, "subject_error_summary", err_summary, max_rows=500)
    write_table(out, "subject_error_detail", err_detail, max_rows=400)
    write_table(out, "plus_ch1_meta_coefficients_by_fold", coeffs)
    write_table(out, "plus_ch1_high_scoring_cn_false_positives", high_cn)
    write_table(out, "latent_mfr_direction_removal_k_audit", k_audit)
    write_table(out, "manufacturer_operating_profile", mfr)
    write_recommendation(out, metrics, coeffs, k_audit, mfr)

    readme = [
        "# Promising Frozen-Latent Candidate Deep Audit",
        "",
        "Focused read-only audit of:",
        "- `promoted_plus_ch1_constrained_score_readout`",
        "- `promoted_latent_mfr_direction_removal`",
        "",
        "References:",
        "- `reference_promoted_ch102_latent384_beta3p75`",
        "- `reference_ch1only_latent384_beta3p75`",
        "",
        "All comparisons use existing OOF-ECDF predictions and frozen latent/score artifacts. No VAE training, OASIS scoring, threshold refitting on outer test, tensor modification, metadata modification, or model artifact modification was performed.",
    ]
    (out / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    command_log.append(
        {
            "timestamp": now_iso(),
            "event": "completed",
            "n_models": int(pred["model_id"].nunique()),
            "n_subject_model_rows": int(len(pred)),
            "outputs": [
                "candidate_auc_pr_tradeoff.csv",
                "roc_curve_points.csv",
                "pr_curve_points.csv",
                "precision_recall_fixed_levels.csv",
                "topk_ad_enrichment.csv",
                "score_distribution_by_diagnosis_manufacturer.csv",
                "subject_error_summary.csv",
                "plus_ch1_meta_coefficients_by_fold.csv",
                "latent_mfr_direction_removal_k_audit.csv",
                "manufacturer_operating_profile.csv",
                "final_recommendation.md",
            ],
        }
    )
    (out / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
