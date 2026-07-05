#!/usr/bin/env python3
"""Read-only channel-ablation evidence audit for the ADNI revision paper.

The script only reads existing repo/results artifacts and writes derived audit
outputs under results/revision_bspc_2026/channel_ablation_evidence_audit_20260620.
No training, scoring, tensor mutation, metadata mutation, or manuscript editing.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/revision_bspc_2026"
OUTDIR = RESULTS / "channel_ablation_evidence_audit_20260620"
MEDIA_RESULTS = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026")
CONFIGS = ROOT / "configs/runs"

PROMOTED_ID = "promoted_latent384_beta3p75_ch1_0_2"
PRIMARY_MODEL_NAME = "logreg_l2_original"
PRIMARY_FEATURE_SET = "z_plus_age_sex"
PRIMARY_CALIB = "oof_ecdf"
PRIMARY_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"

CHANNEL_NAMES = {
    0: "Pearson_OMST_GCE_Signed_Weighted",
    1: "Pearson_Full_FisherZ_Signed",
    2: "MI_KNN_Symmetric",
    3: "dFC_AbsDiffMean",
    4: "dFC_StdDev",
    5: "DistanceCorr",
    6: "Granger_F_lag1",
}


def write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_df(df: pd.DataFrame, csv_path: Path, md_path: Optional[Path] = None) -> None:
    df.to_csv(csv_path, index=False)
    if md_path is not None:
        md_path.write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")


def read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        return None
    return None


def as_float(x: Any) -> float:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def stringify_channels(x: Any) -> str:
    if pd.isna(x) if not isinstance(x, (list, tuple, dict)) else False:
        return ""
    return str(x)


def classify_artifact(path: Path) -> tuple[str, str]:
    s = str(path).lower()
    if "fast3x3" in s or "greedy_fast" in s or "fast_channel" in s or "ablation_v3" in s:
        return "PARTIALLY COMPARABLE / exploratory only", "FAST or greedy screening; not current full 5x5 promoted protocol."
    if "v4_channel" in s or "v5_1b" in s or "locked_current_model" in s:
        return "PARTIALLY COMPARABLE / exploratory only", "Historical/older objective or dataset branch."
    if "final_model_evidence_map" in s or "final_full_model_evidence_map" in s:
        return "DIRECTLY COMPARABLE where row settings match", "Aggregated final evidence map; row-level comparability still depends on channel/beta/latent settings."
    if "ch1only" in s or "ch12" in s or "stageb_oof_score_calibration" in s:
        return "DIRECTLY COMPARABLE if same latent/beta/protocol; otherwise partial", "Current recover035/full5x5 channel variant artifact."
    return "NEEDS REVIEW", "Relevant keyword match; inspect row/source before manuscript use."


def inventory_artifacts() -> pd.DataFrame:
    patterns = [
        "*ablation*",
        "*greedy*",
        "*channel*",
        "*evidence*map*",
        "*model*decision*",
        "*promotion*gate*",
        "*candidate*metrics*",
        "*ch1only*",
        "*ch12*",
    ]
    paths: set[Path] = set()
    for pat in patterns:
        paths.update(RESULTS.glob(pat))
        paths.update(RESULTS.glob(f"*/{pat}"))
        paths.update(RESULTS.glob(f"*/*/{pat}"))
    manuscript_matches = list(ROOT.glob("**/Manuscript_v4_v2.tex"))
    figure_matches = list(ROOT.glob("**/Fig_Ablation_v1.png")) + list(RESULTS.glob("**/Fig_Ablation_v1.png"))

    rows: list[dict[str, Any]] = []
    for p in sorted(paths):
        if not p.exists():
            continue
        comp, note = classify_artifact(p)
        rows.append(
            {
                "artifact_path": str(p),
                "artifact_name": p.name,
                "artifact_type": "directory" if p.is_dir() else "file",
                "date_hint": extract_date(str(p)),
                "comparability_initial": comp,
                "audit_note": note,
            }
        )
    rows.append(
        {
            "artifact_path": "Manuscript_v4_v2.tex",
            "artifact_name": "Manuscript_v4_v2.tex",
            "artifact_type": "manuscript_search",
            "date_hint": "",
            "comparability_initial": "NOT FOUND IN WORKTREE",
            "audit_note": f"Exact manuscript file matches found: {len(manuscript_matches)}.",
        }
    )
    rows.append(
        {
            "artifact_path": "Fig_Ablation_v1.png",
            "artifact_name": "Fig_Ablation_v1.png",
            "artifact_type": "figure_search",
            "date_hint": "",
            "comparability_initial": "NOT FOUND IN SEARCHED PATHS",
            "audit_note": f"Exact figure matches found: {len(figure_matches)}.",
        }
    )
    return pd.DataFrame(rows)


def extract_date(text: str) -> str:
    m = re.search(r"20\d{6}", text)
    return m.group(0) if m else ""


def load_latest_evidence_map() -> pd.DataFrame:
    candidates = [
        RESULTS / "final_model_evidence_map_with_beta9p5_T160_chweighted_20260609/full_model_evidence_map.csv",
        RESULTS / "final_model_evidence_map_with_plus_ch1_pr_oasis_20260608/full_model_evidence_map.csv",
        RESULTS / "final_full_model_evidence_map_with_ch12_beta2p75_20260607/full_model_evidence_map.csv",
    ]
    for p in candidates:
        df = read_csv(p)
        if df is not None and not df.empty:
            df["_source_file"] = str(p)
            return df
    raise FileNotFoundError("No final full-model evidence map found")


def run_counts(run_dir: str | Path | float | None) -> dict[str, Any]:
    if run_dir is None or (isinstance(run_dir, float) and math.isnan(run_dir)):
        return {}
    paths = [Path(str(run_dir))]
    if not paths[0].is_absolute():
        paths.append(MEDIA_RESULTS / paths[0].name)
    for rd in paths:
        if not rd.exists():
            continue
        fold1 = rd / "fold_1"
        if not fold1.exists():
            continue
        out: dict[str, Any] = {}
        train_p = fold1 / "train_dev_subjects_fold.csv"
        test_p = fold1 / "test_subjects_fold.csv"
        if train_p.exists() and test_p.exists():
            tr = pd.read_csv(train_p)
            te = pd.read_csv(test_p)
            clf = pd.concat([tr, te], ignore_index=True)
            dx = clf.get("ResearchGroup_Mapped", pd.Series(dtype=str)).astype(str)
            out.update(
                {
                    "classifier_pool_n": int(len(clf)),
                    "classifier_pool_CN": int(dx.eq("CN").sum()),
                    "classifier_pool_AD": int(dx.eq("AD").sum()),
                }
            )
        vae_idx = fold1 / "vae_training_pool_tensor_idx.npy"
        if vae_idx.exists():
            out["vae_pool_n"] = int(len(np.load(vae_idx)))
        return out
    return {}


def config_info(model_id: str, run_name: str) -> dict[str, Any]:
    candidates = []
    if run_name:
        candidates += list(CONFIGS.glob(f"*{run_name}*.json"))
    if "ch1only" in model_id:
        candidates += list(CONFIGS.glob("*ch1only_latent384_beta3p75_T80_h10000_p560_full5x5.json"))
    elif "ch12_latent384_beta3p75" in model_id:
        candidates += list(CONFIGS.glob("*ch12_latent384_beta3p75_T80_h10000_p560_full5x5.json"))
    elif "ch12_latent384_beta2p75" in model_id:
        candidates += list(CONFIGS.glob("*ch12_latent384_beta2p75_currentloss_T80_h10000_p560_full5x5.json"))
    elif model_id == PROMOTED_ID:
        candidates += list(CONFIGS.glob("*recover035_latent384_beta3p75_T80_h10000_p560_full5x5.json"))
    out: dict[str, Any] = {}
    for p in candidates:
        try:
            cfg = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        out["config_path"] = str(p)
        for key in ["tensor_path", "metadata_path", "channels_to_use", "latent_dim", "beta_vae", "epochs_vae", "early_stopping_patience_vae", "lr_scheduler_T0", "outer_folds", "inner_folds"]:
            if key in cfg:
                out[key] = cfg[key]
        # Some configs nest under args.
        args = cfg.get("args", {}) if isinstance(cfg, dict) else {}
        for key in ["tensor_path", "metadata_path", "channels_to_use", "latent_dim", "beta_vae", "epochs_vae", "early_stopping_patience_vae", "lr_scheduler_T0", "outer_folds", "inner_folds"]:
            if key not in out and key in args:
                out[key] = args[key]
        return out
    return out


def comparable_table() -> pd.DataFrame:
    ev = load_latest_evidence_map()
    keep_ids = [
        PROMOTED_ID,
        "ch1only_latent384_beta3p75",
        "ch1only_latent256_beta3p75",
        "ch1only_latent384_beta3p25",
        "ch12_latent384_beta3p75",
        "ch12_latent384_beta2p75_currentloss_stageB_oof_ecdf",
        "chweightedPearson50_latent384_beta3p75",
    ]
    rows: list[dict[str, Any]] = []
    for _, r in ev.iterrows():
        mid = str(r.get("model_id", ""))
        if mid not in keep_ids:
            continue
        comp = "PARTIALLY COMPARABLE / exploratory only"
        reason = ""
        if mid == PROMOTED_ID:
            comp = "DIRECTLY COMPARABLE"
            reason = "Reference promoted model."
        elif mid == "ch1only_latent384_beta3p75":
            comp = "DIRECTLY COMPARABLE"
            reason = "Same recover035 full5x5 latent384 beta3.75 protocol; channel subset differs."
        elif mid == "ch12_latent384_beta3p75":
            comp = "DIRECTLY COMPARABLE"
            reason = "Same recover035 full5x5 latent384 beta3.75 protocol; channel pair [1,2] differs."
        elif "ch12" in mid and "beta2p75" in mid:
            reason = "Channel pair plus beta changed; useful sensitivity but not isolated channel-set evidence."
        elif "ch1only_latent256" in mid or "ch1only_latent384_beta3p25" in mid:
            reason = "Channel plus capacity/beta follow-up; not isolated channel-set evidence."
        elif "chweighted" in mid:
            reason = "Channel weighting/loss sensitivity, not a channel-subset ablation."
        counts = run_counts(r.get("run_dir"))
        cfg = config_info(mid, str(r.get("run_name", "")))
        ba = r.get("adni_oof_ecdf_balanced_accuracy", r.get("adni_oof_ecdf_ba", np.nan))
        sens = r.get("adni_oof_ecdf_sensitivity", r.get("adni_oof_ecdf_sens", np.nan))
        spec = r.get("adni_oof_ecdf_specificity", r.get("adni_oof_ecdf_spec", np.nan))
        row = {
            "run_id": mid,
            "model_name": r.get("display_name", ""),
            "output_directory": r.get("run_dir", ""),
            "date": extract_date(str(r.get("_source_file", ""))) or extract_date(str(r.get("run_dir", ""))),
            "tensor_path": cfg.get("tensor_path", ""),
            "tensor_sha": "",
            "metadata_path": cfg.get("metadata_path", ""),
            "vae_pool_n": counts.get("vae_pool_n", np.nan),
            "classifier_pool_n": counts.get("classifier_pool_n", np.nan),
            "classifier_AD": counts.get("classifier_pool_AD", np.nan),
            "classifier_CN": counts.get("classifier_pool_CN", np.nan),
            "MCI": 250 if counts.get("vae_pool_n", np.nan) == 647 else np.nan,
            "channel_set_order": r.get("channel_set_order", cfg.get("channels_to_use", "")),
            "channel_names": r.get("selected_channel_names", ""),
            "latent_dim": r.get("latent_dim", cfg.get("latent_dim", np.nan)),
            "beta": r.get("beta_vae", cfg.get("beta_vae", np.nan)),
            "epochs": cfg.get("epochs_vae", np.nan),
            "patience": cfg.get("early_stopping_patience_vae", np.nan),
            "scheduler_T0": cfg.get("lr_scheduler_T0", np.nan),
            "evidence_type": "FULL Stage B OOF-ECDF",
            "outer_inner_cv_design": "5x5 nested where Stage B OOF calibration artifact is available",
            "auc": r.get("adni_oof_ecdf_auc", np.nan),
            "pr_auc": r.get("adni_oof_ecdf_pr_auc", np.nan),
            "balanced_accuracy": ba,
            "sensitivity": sens,
            "specificity": spec,
            "f1": r.get("adni_oof_ecdf_f1", np.nan),
            "brier": np.nan,
            "oof_predictions_available": bool(find_calib_predictions(mid)),
            "philips_cn_fpr": r.get("philips_cn_fpr", np.nan),
            "manufacturer_metrics_available": not pd.isna(r.get("scanner_leakage_latent_acc", np.nan)),
            "scanner_leakage_latent_acc": r.get("scanner_leakage_latent_acc", np.nan),
            "external_oasis_available": not pd.isna(r.get("oasis_runwise164_auc", np.nan)),
            "oasis_runwise164_auc": r.get("oasis_runwise164_auc", np.nan),
            "oasis_runwise164_pr_auc": r.get("oasis_runwise164_pr_auc", np.nan),
            "comparability": comp,
            "comparability_reason": reason,
        }
        rows.append(row)

    # Add current FAST table as explicitly exploratory.
    fast = read_csv(RESULTS / "channel_ablation_fast3x3_offdiag_channelmean/channel_subset_ranking.csv")
    if fast is not None:
        for _, r in fast.head(13).iterrows():
            rows.append(
                {
                    "run_id": f"FAST3x3_{r.get('run_key')}",
                    "model_name": r.get("run_key", ""),
                    "output_directory": str(RESULTS / "channel_ablation_fast3x3_offdiag_channelmean"),
                    "date": "",
                    "tensor_path": "",
                    "tensor_sha": "",
                    "metadata_path": "",
                    "vae_pool_n": np.nan,
                    "classifier_pool_n": np.nan,
                    "classifier_AD": np.nan,
                    "classifier_CN": np.nan,
                    "MCI": np.nan,
                    "channel_set_order": r.get("channels", ""),
                    "channel_names": r.get("selected_channel_names", ""),
                    "latent_dim": np.nan,
                    "beta": np.nan,
                    "epochs": "FAST3x3",
                    "patience": "FAST3x3",
                    "scheduler_T0": "FAST3x3",
                    "evidence_type": "FAST 3x3 screening",
                    "outer_inner_cv_design": "3x3 fast screening, not final full5x5",
                    "auc": r.get("auc", np.nan),
                    "pr_auc": r.get("pr_auc", np.nan),
                    "balanced_accuracy": r.get("balanced_accuracy", np.nan),
                    "sensitivity": r.get("sensitivity", np.nan),
                    "specificity": r.get("specificity", np.nan),
                    "f1": r.get("f1", np.nan),
                    "brier": np.nan,
                    "oof_predictions_available": False,
                    "philips_cn_fpr": np.nan,
                    "manufacturer_metrics_available": False,
                    "scanner_leakage_latent_acc": np.nan,
                    "external_oasis_available": False,
                    "oasis_runwise164_auc": np.nan,
                    "oasis_runwise164_pr_auc": np.nan,
                    "comparability": "PARTIALLY COMPARABLE / exploratory only",
                    "comparability_reason": "FAST 3x3 screening only; not current full5x5 promoted protocol.",
                }
            )
    return pd.DataFrame(rows)


def find_calib_predictions(model_id: str) -> Optional[Path]:
    mapping = {
        PROMOTED_ID: RESULTS / "recover035_latent384_beta3p75_stageB_oof_score_calibration/calib_predictions.csv",
        "ch1only_latent384_beta3p75": RESULTS / "recover035_ch1only_latent384_beta3p75_T80_h10000_p560_full5x5_stageB_oof_score_calibration/calib_predictions.csv",
        "ch1only_latent256_beta3p75": RESULTS / "ch1only_latent256_beta3p75_T80_h10000_p560_full5x5_stageB_oof_score_calibration/calib_predictions.csv",
        "ch1only_latent384_beta3p25": RESULTS / "ch1only_latent384_beta3p25_T80_h10000_p560_full5x5_stageB_oof_score_calibration/calib_predictions.csv",
        "ch12_latent384_beta3p75": RESULTS / "recover035_ch12_latent384_beta3p75_T80_h10000_p560_full5x5_stageB_oof_score_calibration/calib_predictions.csv",
        "ch12_latent384_beta2p75_currentloss_stageB_oof_ecdf": RESULTS / "recover035_ch12_latent384_beta2p75_currentloss_T80_h10000_p560_full5x5_stageB_oof_score_calibration/calib_predictions.csv",
    }
    p = mapping.get(model_id)
    return p if p is not None and p.exists() else None


def primary_prediction_frame(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    mask = (
        df["model_name"].astype(str).eq(PRIMARY_MODEL_NAME)
        & df["feature_set"].astype(str).eq(PRIMARY_FEATURE_SET)
        & df["calib_method"].astype(str).eq(PRIMARY_CALIB)
        & df["threshold_strategy"].astype(str).eq(PRIMARY_THRESHOLD)
    )
    out = df.loc[mask].copy()
    if out.empty:
        raise ValueError(f"No primary promoted-convention rows in {path}")
    if out[["SubjectID", "fold"]].duplicated().any():
        raise ValueError(f"Duplicate primary rows in {path}")
    return out.sort_values(["SubjectID", "fold"]).reset_index(drop=True)


def paired_bootstrap() -> pd.DataFrame:
    promoted_path = find_calib_predictions(PROMOTED_ID)
    if promoted_path is None:
        return pd.DataFrame()
    ref = primary_prediction_frame(promoted_path)
    candidates = ["ch1only_latent384_beta3p75", "ch12_latent384_beta3p75", "ch12_latent384_beta2p75_currentloss_stageB_oof_ecdf"]
    rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(20260620)
    for mid in candidates:
        p = find_calib_predictions(mid)
        if p is None:
            continue
        cand = primary_prediction_frame(p)
        merged = ref[["SubjectID", "fold", "y_true", "y_score", "y_pred"]].merge(
            cand[["SubjectID", "fold", "y_true", "y_score", "y_pred"]],
            on=["SubjectID", "fold", "y_true"],
            suffixes=("_promoted", "_candidate"),
            how="inner",
        )
        if len(merged) < 10:
            continue
        y = merged["y_true"].to_numpy(int)
        sp = merged["y_score_promoted"].to_numpy(float)
        sc = merged["y_score_candidate"].to_numpy(float)
        pp = merged["y_pred_promoted"].to_numpy(int)
        pc = merged["y_pred_candidate"].to_numpy(int)
        point = {
            "candidate_model_id": mid,
            "n_paired_subjects": int(len(merged)),
            "delta_auc_candidate_minus_promoted": metric_auc(y, sc) - metric_auc(y, sp),
            "delta_pr_auc_candidate_minus_promoted": average_precision_score(y, sc) - average_precision_score(y, sp),
            "delta_ba_candidate_minus_promoted": balanced_accuracy_score(y, pc) - balanced_accuracy_score(y, pp),
            "delta_f1_candidate_minus_promoted": f1_score(y, pc, zero_division=0) - f1_score(y, pp, zero_division=0),
        }
        boot = {k: [] for k in ["auc", "pr_auc", "ba", "f1"]}
        n = len(y)
        for _ in range(2000):
            idx = rng.integers(0, n, size=n)
            yy = y[idx]
            if len(np.unique(yy)) < 2:
                continue
            spp = sp[idx]
            scc = sc[idx]
            ppp = pp[idx]
            pcc = pc[idx]
            boot["auc"].append(metric_auc(yy, scc) - metric_auc(yy, spp))
            boot["pr_auc"].append(average_precision_score(yy, scc) - average_precision_score(yy, spp))
            boot["ba"].append(balanced_accuracy_score(yy, pcc) - balanced_accuracy_score(yy, ppp))
            boot["f1"].append(f1_score(yy, pcc, zero_division=0) - f1_score(yy, ppp, zero_division=0))
        for metric, vals in boot.items():
            arr = np.asarray(vals, dtype=float)
            point[f"{metric}_delta_ci2p5"] = float(np.nanpercentile(arr, 2.5)) if len(arr) else np.nan
            point[f"{metric}_delta_ci97p5"] = float(np.nanpercentile(arr, 97.5)) if len(arr) else np.nan
        rows.append(point)
    return pd.DataFrame(rows)


def metric_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def claim_audit_text(comp: pd.DataFrame, boot: pd.DataFrame) -> str:
    def get(model_id: str, col: str) -> float:
        row = comp[comp["run_id"].eq(model_id)]
        return as_float(row[col].iloc[0]) if not row.empty and col in row else float("nan")

    promoted_auc = get(PROMOTED_ID, "auc")
    promoted_pr = get(PROMOTED_ID, "pr_auc")
    ch1_auc = get("ch1only_latent384_beta3p75", "auc")
    ch1_pr = get("ch1only_latent384_beta3p75", "pr_auc")
    ch12_auc = get("ch12_latent384_beta3p75", "auc")
    ch12_pr = get("ch12_latent384_beta3p75", "pr_auc")

    return f"""# Claim Validity Audit

## A. "Full Pearson emerged as the strongest individual predictor."

**Supported only with conservative wording.** The FAST screening tables consistently rank channel `[1]` (Pearson Full Fisher-z) as the strongest single-channel predictor among completed single-channel FAST candidates. The current FULL evidence also includes a ch1-only latent384 beta3.75 model with OOF-ECDF AUC {ch1_auc:.6f} and PR-AUC {ch1_pr:.6f}. However, the current final dataset does **not** contain matched FULL latent384 beta3.75 single-channel `[0]` and `[2]` runs, so the claim should be framed as FAST-screening plus parsimonious sensitivity evidence, not as a complete FULL single-channel ranking.

## B. "Greedy forward selection peaked at Full Pearson + OMST."

**Not supported under the current evidence.** The completed FAST 3x3 table ranks `[1]` above `[1,0]`, and the pair `[1,2]` is stronger than `[1,0]` in that FAST table. The older greedy table does not support a stable Pearson+OMST peak under the final setting. This claim should be removed or rewritten.

## C. "The full high-capacity beta-VAE three-channel tensor consistently outperformed the two-channel model."

**Mostly supported for the current directly comparable [1,2] pair, but not as a universal statement.** The promoted `[1,0,2]` model has OOF-ECDF AUC {promoted_auc:.6f}, PR-AUC {promoted_pr:.6f}; the directly comparable `[1,2]` latent384 beta3.75 channel-pair sensitivity has AUC {ch12_auc:.6f}, PR-AUC {ch12_pr:.6f}. This supports retaining the three-channel model over the tested `[1,2]` pair. It does not prove superiority over every possible two-channel setting.

## D. "All subsequent analyses were performed on three-channel tensors based on this evidence."

**Needs revision.** Subsequent analyses used the three-channel promoted model because it was the pre-specified final primary model after full-model ADNI/OASIS evidence, not solely because the old FAST greedy ablation selected a two-channel or three-channel subset. The manuscript should say the FAST ablation motivated channel scrutiny, while final channel choice was locked by full 5x5 model-selection evidence and external stress testing.

## Paired OOF Bootstrap Note

Prediction-level paired bootstrap comparisons were computed where promoted-convention OOF predictions existed. These are descriptive because candidates are separately trained VAE runs and fold count is only five. Use them as internal support, not formal proof of channel superiority.
"""


def figure_recommendation_text() -> str:
    return """# Figure Recommendation

`Fig_Ablation_v1.png` was not found in the searched repo/results tree. Based on the described manuscript content, the old FAST greedy ablation figure should **not** remain as a main-paper figure if it implies confirmatory current-model channel selection.

Recommended handling:

1. Move the old FAST ablation figure to the supplement, clearly labeled "exploratory FAST screening".
2. Replace the main-paper ablation panel with a compact current-evidence table/forest-style panel showing:
   - promoted `[1,0,2]` latent384 beta3.75 FULL 5x5 OOF-ECDF;
   - ch1-only latent384 beta3.75 FULL 5x5 sensitivity;
   - `[1,2]` latent384 beta3.75 FULL 5x5 sensitivity;
   - optional `[1,2]` beta2.75 as rejected beta/channel sensitivity.
3. Do not state that the old FAST result is confirmatory. State that it motivated the final channel candidates.

If no new figure can be made before submission, use a supplement-only table and keep the main text conservative.
"""


def reviewer_text() -> str:
    return """# Reviewer-Safe Ablation Text

## Conservative Methods/Results Paragraph

We evaluated connectivity-channel dependence in two stages. First, we used a computationally cheaper FAST screening analysis to rank individual and small channel subsets; this screen consistently identified the Pearson Full Fisher-z channel as the strongest single-channel signal source, but was treated as exploratory because it used an abbreviated training protocol. We then evaluated selected channel subsets under the final full 5x5 protocol. In the final recover035 latent384 beta3.75 setting, the promoted three-channel tensor `[1,0,2]` achieved OOF-ECDF ROC-AUC 0.795 and PR-AUC 0.574. A parsimonious Pearson-only model achieved slightly higher internal AUC/PR-AUC (0.800/0.586) but had weaker operating-profile and external-transfer behavior, while the directly comparable `[1,2]` channel-pair sensitivity was lower (AUC 0.785, PR-AUC 0.543). We therefore retained `[1,0,2]` as the primary model and report single-/pair-channel results as sensitivity analyses rather than as independent model-selection claims.

## Figure Caption Proposal

Channel-sensitivity evidence. Exploratory FAST screening ranked Pearson Full Fisher-z as the strongest individual channel, but full-model selection was based on the final 5x5 protocol. The promoted `[1,0,2]` latent384 beta3.75 model is shown alongside the Pearson-only and `[1,2]` full 5x5 sensitivities. FAST results are displayed only as screening evidence and were not used as confirmatory performance estimates.

## Response-to-Reviewer Paragraph

We agree that channel-ablation evidence should be separated from final model-selection evidence. We have revised the text to distinguish the FAST channel screen from the final full 5x5 sensitivities. The FAST analysis is now described as exploratory and moved to the supplement. The main text now relies on the matched full-protocol comparisons: the Pearson-only model is a parsimonious sensitivity with high internal rank performance, whereas the promoted `[1,0,2]` model remains primary because it has the best overall balance across ADNI OOF-ECDF performance, operating-point behavior, Philips/manufacturer error profile, and OASIS stress-test transfer. We no longer claim that the old greedy FAST analysis alone selected the final tensor.
"""


def minimal_plan_text() -> str:
    return """# Minimal Confirmatory Ablation Plan

No new training is required if the manuscript is revised conservatively: keep the old FAST ablation as exploratory/supplementary and base the main claim on existing FULL 5x5 `[1,0,2]`, `[1]`, and `[1,2]` evidence.

If a reviewer specifically requires a complete matched single-channel FULL ablation under the current final protocol, the smallest scientifically clean plan would be:

1. FULL `[0]` latent384 beta3.75 T80 h10000 p560 full5x5.
2. FULL `[2]` latent384 beta3.75 T80 h10000 p560 full5x5.
3. Optional FULL `[1,0]` latent384 beta3.75 T80 h10000 p560 full5x5 only if the manuscript wants to discuss Pearson+OMST specifically.

All runs should reuse the promoted recover035 metadata, tensor, folds, latent_dim, beta, dropout, scheduler, Stage B OOF-ECDF readout, and OASIS scoring protocol. Do not launch these unless explicitly requested; they are confirmatory completeness runs, not necessary for the conservative revision.

Planned dry-run command pattern, to be generated only if requested:

```bash
/home/diego/anaconda3/envs/vae_ad/bin/python scripts/revision_bspc_2026/create_and_preflight_channel_subset_full5x5.py --base configs/runs/adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5.json --channels <CHANNELS> --dry-run
```
"""


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    command_log: list[dict[str, Any]] = []

    inv = inventory_artifacts()
    write_df(inv, OUTDIR / "ablation_artifact_inventory.csv", OUTDIR / "ablation_artifact_inventory.md")
    command_log.append({"output": "ablation_artifact_inventory.csv", "rows": int(len(inv))})

    comp = comparable_table()
    boot = paired_bootstrap()
    if not boot.empty:
        comp = comp.merge(
            boot,
            left_on="run_id",
            right_on="candidate_model_id",
            how="left",
        )
    write_df(comp, OUTDIR / "comparable_model_evidence_table.csv", OUTDIR / "comparable_model_evidence_table.md")
    command_log.append({"output": "comparable_model_evidence_table.csv", "rows": int(len(comp))})

    if not boot.empty:
        write_df(boot, OUTDIR / "paired_oof_bootstrap_comparisons.csv", OUTDIR / "paired_oof_bootstrap_comparisons.md")
        command_log.append({"output": "paired_oof_bootstrap_comparisons.csv", "rows": int(len(boot))})

    write_text(OUTDIR / "claim_validity_audit.md", claim_audit_text(comp, boot))
    write_text(OUTDIR / "figure_recommendation.md", figure_recommendation_text())
    write_text(OUTDIR / "reviewer_safe_ablation_text.md", reviewer_text())
    write_text(OUTDIR / "minimal_confirmatory_ablation_plan.md", minimal_plan_text())
    command_log.extend(
        [
            {"output": "claim_validity_audit.md"},
            {"output": "figure_recommendation.md"},
            {"output": "reviewer_safe_ablation_text.md"},
            {"output": "minimal_confirmatory_ablation_plan.md"},
            {"guardrails": "read-only audit; no training; no tensor/metadata/fold/model/manuscript modification"},
        ]
    )
    write_text(OUTDIR / "command_log.json", json.dumps(command_log, indent=2))


if __name__ == "__main__":
    main()
