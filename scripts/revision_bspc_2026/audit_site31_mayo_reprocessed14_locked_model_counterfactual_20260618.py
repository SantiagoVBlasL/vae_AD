#!/usr/bin/env python3
"""Locked-model counterfactual audit for Site31 Mayo slice-order replacement tensor.

Read-only with respect to source tensors, metadata, and trained model artifacts.
Writes only derived audit tables under results/revision_bspc_2026.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)

from betavae_xai.data.preprocessing import apply_normalization_params
from betavae_xai.models import ConvolutionalVAE


PROMOTED_RUN_DIR = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/"
    "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
)
ORIG_TENSOR = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/"
    "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)
REPL_TENSOR = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass_site31_mayo_reprocessed14/"
    "subject_tensors/"
    "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass_site31_mayo_reprocessed14.npz"
)
METADATA = PROJECT_ROOT / "results/revision_bspc_2026/adni_035_metadata_rescue_preflight/patched_metadata_candidate.csv"
OUTDIR = PROJECT_ROOT / "results/revision_bspc_2026/site31_mayo_reprocessed14_locked_model_counterfactual_20260618"

SITE31_SUBJECTS = [
    "031_S_4024",
    "031_S_4029",
    "031_S_4032",
    "031_S_4042",
    "031_S_4149",
    "031_S_4194",
    "031_S_4203",
    "031_S_4218",
    "031_S_4474",
    "031_S_4476",
    "031_S_4496",
    "031_S_4590",
    "031_S_4721",
    "031_S_4947",
]
KNOWN_REVERSE = {"031_S_4021", "031_S_4032", "031_S_4218", "031_S_4474", "031_S_4496"}
SELECTED_CHANNELS = [1, 0, 2]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {label}: {path}")


def md_from_df(df: pd.DataFrame, path: Path, max_rows: int | None = None) -> None:
    view = df if max_rows is None else df.head(max_rows)
    path.write_text(view.to_markdown(index=False) + "\n", encoding="utf-8")


def load_npz_tensor(path: Path) -> tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    require(path, "tensor NPZ")
    npz = np.load(path, allow_pickle=True)
    if "global_tensor_data" not in npz or "subject_ids" not in npz:
        raise KeyError(f"{path} missing global_tensor_data or subject_ids")
    tensor = np.asarray(npz["global_tensor_data"])
    subject_ids = np.asarray(npz["subject_ids"]).astype(str)
    meta = {k: npz[k] for k in npz.files if k not in {"global_tensor_data", "subject_ids"}}
    return tensor, subject_ids, meta


def normalize_dx(x: Any) -> str:
    s = str(x).strip().upper()
    if s in {"CN", "NORMAL", "CONTROL"}:
        return "CN"
    if s in {"AD", "DEMENTIA", "AD_OR_DEMENTIA"}:
        return "AD"
    if s == "MCI":
        return "MCI"
    return str(x).strip()


def label_from_dx(dx: Any) -> float:
    d = normalize_dx(dx)
    if d == "CN":
        return 0.0
    if d == "AD":
        return 1.0
    return np.nan


def build_model() -> ConvolutionalVAE:
    return ConvolutionalVAE(
        input_channels=3,
        latent_dim=384,
        image_size=131,
        final_activation="tanh",
        intermediate_fc_dim_config="quarter",
        dropout_rate=0.15,
        use_layernorm_fc=False,
        num_conv_layers_encoder=4,
        decoder_type="convtranspose",
    )


def load_fold_vae(fold_dir: Path, device: torch.device) -> ConvolutionalVAE:
    model_path = fold_dir / f"vae_model_fold_{fold_dir.name.split('_')[-1]}.pt"
    require(model_path, "fold VAE checkpoint")
    model = build_model().to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def compute_mu(
    model: ConvolutionalVAE,
    tensor_rows: np.ndarray,
    norm_params: List[Dict[str, Any]],
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    selected = tensor_rows[:, SELECTED_CHANNELS, :, :].astype(np.float32, copy=True)
    normed = apply_normalization_params(selected, norm_params).astype(np.float32, copy=False)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, normed.shape[0], batch_size):
            xb = torch.from_numpy(normed[start : start + batch_size]).to(device)
            mu, _logvar = model.encode(xb)
            out.append(mu.detach().cpu().numpy())
    return np.vstack(out)


def get_score(estimator: Any, x: pd.DataFrame) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        p = estimator.predict_proba(x)
        if np.asarray(p).ndim == 2 and p.shape[1] >= 2:
            return np.asarray(p)[:, 1].astype(float)
        return np.asarray(p).ravel().astype(float)
    if hasattr(estimator, "decision_function"):
        return np.asarray(estimator.decision_function(x)).ravel().astype(float)
    return np.asarray(estimator.predict(x)).ravel().astype(float)


def make_feature_frame(mu: np.ndarray, base: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    latent_cols = [f"latent_{i}" for i in range(mu.shape[1])]
    df = pd.DataFrame(mu, columns=latent_cols)
    df["Age"] = pd.to_numeric(base["Age"], errors="coerce").to_numpy()
    df["Sex"] = base["Sex"].astype(str).to_numpy()
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Feature frame missing columns: {missing[:10]}")
    return df[feature_columns].copy()


def find_split_role(subject_id: str, tensor_idx: int, fold_dir: Path) -> str:
    test = pd.read_csv(fold_dir / "test_subjects_fold.csv")
    train = pd.read_csv(fold_dir / "train_dev_subjects_fold.csv")
    if subject_id in set(test["SubjectID"].astype(str)):
        return "classifier_test"
    if subject_id in set(train["SubjectID"].astype(str)):
        return "classifier_train_dev"
    vae_pool_path = fold_dir / "vae_training_pool_tensor_idx.npy"
    if vae_pool_path.exists():
        vae_pool = set(np.load(vae_pool_path).astype(int).tolist())
        if int(tensor_idx) in vae_pool:
            return "vae_pool_only"
    return "not_in_fold_pool"


def metrics_from_scores(y: np.ndarray, score: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    y = np.asarray(y).astype(int)
    score = np.asarray(score).astype(float)
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    out = {
        "n": int(len(y)),
        "threshold": float(threshold),
        "auc": float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
        "pr_auc": float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "specificity": float(recall_score(y, pred, pos_label=0, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, score)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    return out


def subgroup_summary(rows: pd.DataFrame) -> pd.DataFrame:
    groups = []
    specs = [
        ("all_14_subject_fold_rows", rows.index == rows.index),
        ("classifier_CN_AD_subject_fold_rows", rows["diagnosis"].isin(["CN", "AD"])),
        ("cn_subject_fold_rows", rows["diagnosis"].eq("CN")),
        ("ad_subject_fold_rows", rows["diagnosis"].eq("AD")),
        ("known_reverse_reprocessed_fold_rows", rows["known_reverse_nondefault_reprocessed"].eq(True)),
        ("fold_test_only_rows", rows["split_role"].eq("classifier_test")),
    ]
    for name, mask in specs:
        sub = rows.loc[mask].copy()
        if sub.empty:
            groups.append({"subgroup": name, "n_rows": 0})
            continue
        favorable = sub["favorable_direction"].dropna()
        groups.append(
            {
                "subgroup": name,
                "n_rows": int(len(sub)),
                "n_subjects": int(sub["SubjectID"].nunique()),
                "old_score_mean": float(sub["old_score"].mean()),
                "new_score_mean": float(sub["new_score"].mean()),
                "delta_score_mean": float(sub["delta_score"].mean()),
                "delta_score_median": float(sub["delta_score"].median()),
                "delta_score_min": float(sub["delta_score"].min()),
                "delta_score_max": float(sub["delta_score"].max()),
                "n_label_flips_at_0p5": int((sub["old_pred_0p5"] != sub["new_pred_0p5"]).sum()),
                "n_favorable_direction": int(favorable.sum()) if len(favorable) else 0,
                "favorable_fraction": float(favorable.mean()) if len(favorable) else np.nan,
            }
        )
    return pd.DataFrame(groups)


def compute_fpr(df: pd.DataFrame, score_col: str, pred_col: str, manufacturer: str | None = None, site3: Any | None = None) -> Dict[str, Any]:
    sub = df[df["diagnosis"].eq("CN")].copy()
    if manufacturer is not None and "Manufacturer" in sub:
        sub = sub[sub["Manufacturer"].astype(str).str.upper().str.contains(manufacturer.upper(), na=False)]
    if site3 is not None and "Site3" in sub:
        site_num = pd.to_numeric(sub["Site3"], errors="coerce")
        try:
            site_value = float(site3)
            sub = sub[site_num.eq(site_value)]
        except Exception:
            sub = sub[sub["Site3"].astype(str).eq(str(site3))]
    denom = int(len(sub))
    numer = int(sub[pred_col].sum()) if denom else 0
    return {"numerator": numer, "denominator": denom, "fpr": float(numer / denom) if denom else np.nan}


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    command_log: list[dict[str, Any]] = []

    for path, label in [
        (PROMOTED_RUN_DIR, "promoted run directory"),
        (ORIG_TENSOR, "original tensor"),
        (REPL_TENSOR, "replacement tensor"),
        (METADATA, "metadata"),
    ]:
        require(path, label)
        command_log.append({"check": label, "path": str(path), "exists": True})

    orig_x, orig_ids, _orig_meta = load_npz_tensor(ORIG_TENSOR)
    repl_x, repl_ids, _repl_meta = load_npz_tensor(REPL_TENSOR)
    if orig_x.shape != repl_x.shape:
        raise ValueError(f"Tensor shapes differ: original={orig_x.shape}, replacement={repl_x.shape}")
    if not np.array_equal(orig_ids, repl_ids):
        raise ValueError("Original and replacement subject_ids differ")
    tensor_index = {sid: int(i) for i, sid in enumerate(orig_ids)}

    metadata = pd.read_csv(METADATA)
    metadata["SubjectID"] = metadata["SubjectID"].astype(str).str.strip()
    metadata = metadata.drop_duplicates("SubjectID", keep="first")
    subj_rows = []
    missing_subjects = [sid for sid in SITE31_SUBJECTS if sid not in tensor_index]
    if missing_subjects:
        raise ValueError(f"Site31 subjects missing from tensor: {missing_subjects}")
    for sid in SITE31_SUBJECTS:
        row = metadata.loc[metadata["SubjectID"].eq(sid)]
        if row.empty:
            raise ValueError(f"Site31 subject missing from metadata: {sid}")
        rec = row.iloc[0].to_dict()
        rec["tensor_idx"] = tensor_index[sid]
        rec["diagnosis"] = normalize_dx(rec.get("ResearchGroup_Mapped", rec.get("Diagnosis", "")))
        rec["y_true"] = label_from_dx(rec["diagnosis"])
        rec["known_reverse_nondefault_reprocessed"] = sid in KNOWN_REVERSE
        subj_rows.append(rec)
    subj_df = pd.DataFrame(subj_rows)
    subj_df = subj_df.sort_values("tensor_idx").reset_index(drop=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows: list[pd.DataFrame] = []

    for fold in range(1, 6):
        fold_dir = PROMOTED_RUN_DIR / f"fold_{fold}"
        require(fold_dir, f"fold {fold} directory")
        for req in [
            fold_dir / f"vae_model_fold_{fold}.pt",
            fold_dir / "vae_norm_params.joblib",
            fold_dir / "feature_columns.json",
            fold_dir / f"classifier_logreg_final_pipeline_fold_{fold}.joblib",
            fold_dir / "train_dev_subjects_fold.csv",
            fold_dir / "test_subjects_fold.csv",
        ]:
            require(req, f"fold {fold} artifact")

        norm_params = joblib.load(fold_dir / "vae_norm_params.joblib")
        feature_info = json.loads((fold_dir / "feature_columns.json").read_text(encoding="utf-8"))
        feature_columns = feature_info["final_feature_columns"]
        clf = joblib.load(fold_dir / f"classifier_logreg_final_pipeline_fold_{fold}.joblib")
        model = load_fold_vae(fold_dir, device=device)

        idx = subj_df["tensor_idx"].astype(int).to_numpy()
        mu_old = compute_mu(model, orig_x[idx], norm_params, device=device)
        mu_new = compute_mu(model, repl_x[idx], norm_params, device=device)
        x_old = make_feature_frame(mu_old, subj_df, feature_columns)
        x_new = make_feature_frame(mu_new, subj_df, feature_columns)
        score_old = get_score(clf, x_old)
        score_new = get_score(clf, x_new)

        fold_rows = subj_df[
            [
                "SubjectID",
                "tensor_idx",
                "diagnosis",
                "y_true",
                "Manufacturer",
                "Site3",
                "Age",
                "Sex",
                "ImageID",
                "n_timepoints_raw",
                "known_reverse_nondefault_reprocessed",
            ]
        ].copy()
        fold_rows.insert(0, "fold", fold)
        fold_rows["split_role"] = [
            find_split_role(str(sid), int(tidx), fold_dir)
            for sid, tidx in zip(fold_rows["SubjectID"], fold_rows["tensor_idx"])
        ]
        fold_rows["old_score"] = score_old
        fold_rows["new_score"] = score_new
        fold_rows["delta_score"] = fold_rows["new_score"] - fold_rows["old_score"]
        fold_rows["old_pred_0p5"] = (fold_rows["old_score"] >= 0.5).astype(int)
        fold_rows["new_pred_0p5"] = (fold_rows["new_score"] >= 0.5).astype(int)
        fold_rows["old_correct_if_cn_ad_fold_test"] = np.where(
            fold_rows["split_role"].eq("classifier_test") & fold_rows["y_true"].isin([0.0, 1.0]),
            fold_rows["old_pred_0p5"].eq(fold_rows["y_true"].astype("Int64")).astype(object),
            np.nan,
        )
        fold_rows["new_correct_if_cn_ad_fold_test"] = np.where(
            fold_rows["split_role"].eq("classifier_test") & fold_rows["y_true"].isin([0.0, 1.0]),
            fold_rows["new_pred_0p5"].eq(fold_rows["y_true"].astype("Int64")).astype(object),
            np.nan,
        )
        fold_rows["favorable_direction"] = np.where(
            fold_rows["diagnosis"].eq("CN"),
            fold_rows["delta_score"] < 0,
            np.where(fold_rows["diagnosis"].eq("AD"), fold_rows["delta_score"] > 0, np.nan),
        )
        all_rows.append(fold_rows)
        command_log.append({"fold": fold, "counterfactual_scored_rows": int(len(fold_rows))})

    counter = pd.concat(all_rows, ignore_index=True)
    counter.to_csv(OUTDIR / "subject_level_counterfactual_scores.csv", index=False)
    md_from_df(counter, OUTDIR / "subject_level_counterfactual_scores.md")

    summary = subgroup_summary(counter)
    summary.to_csv(OUTDIR / "subgroup_score_shift_summary.csv", index=False)
    md_from_df(summary, OUTDIR / "subgroup_score_shift_summary.md")

    all_preds_paths = sorted(PROMOTED_RUN_DIR.glob("all_folds_clf_predictions_MULTI_logreg*.csv"))
    if not all_preds_paths:
        raise FileNotFoundError("Missing promoted all_folds_clf_predictions_MULTI_logreg*.csv")
    oof = pd.read_csv(all_preds_paths[0])
    if "classifier_type" in oof.columns:
        oof = oof[oof["classifier_type"].astype(str).eq("logreg")].copy()
    if oof[["fold", "SubjectID"]].duplicated().any():
        dup = oof.loc[oof[["fold", "SubjectID"]].duplicated(keep=False), ["fold", "SubjectID"]]
        raise ValueError(f"Duplicate logreg OOF fold/subject rows remain, first={dup.head().to_dict('records')}")
    oof["SubjectID"] = oof["SubjectID"].astype(str)
    meta_cols = ["SubjectID", "ResearchGroup_Mapped", "Manufacturer", "Site3", "Age", "Sex"]
    oof = oof.merge(metadata[meta_cols], on="SubjectID", how="left")
    oof["diagnosis"] = oof["ResearchGroup_Mapped"].map(normalize_dx)
    oof["orig_pred_0p5"] = (oof["y_score_final"].astype(float) >= 0.5).astype(int)
    oof["patched_score_final"] = oof["y_score_final"].astype(float)
    patch_rows = counter[counter["split_role"].eq("classifier_test") & counter["diagnosis"].isin(["CN", "AD"])].copy()
    validation_rows = []
    for _, row in patch_rows.iterrows():
        mask = oof["fold"].eq(int(row["fold"])) & oof["SubjectID"].eq(str(row["SubjectID"]))
        if int(mask.sum()) != 1:
            validation_rows.append(
                {
                    "fold": int(row["fold"]),
                    "SubjectID": row["SubjectID"],
                    "patch_status": "missing_or_duplicate_oof_row",
                    "n_oof_matches": int(mask.sum()),
                }
            )
            continue
        original_score = float(oof.loc[mask, "y_score_final"].iloc[0])
        oof.loc[mask, "patched_score_final"] = float(row["new_score"])
        validation_rows.append(
            {
                "fold": int(row["fold"]),
                "SubjectID": row["SubjectID"],
                "patch_status": "patched",
                "n_oof_matches": 1,
                "oof_original_score": original_score,
                "recomputed_old_score": float(row["old_score"]),
                "new_counterfactual_score": float(row["new_score"]),
                "oof_minus_recomputed_old": original_score - float(row["old_score"]),
            }
        )

    oof["patched_pred_0p5"] = (oof["patched_score_final"] >= 0.5).astype(int)
    y = oof["y_true"].astype(int).to_numpy()
    original_metrics = metrics_from_scores(y, oof["y_score_final"].astype(float).to_numpy())
    patched_metrics = metrics_from_scores(y, oof["patched_score_final"].astype(float).to_numpy())

    phil_orig = compute_fpr(oof, "y_score_final", "orig_pred_0p5", manufacturer="Philips")
    phil_patch = compute_fpr(oof, "patched_score_final", "patched_pred_0p5", manufacturer="Philips")
    site31_orig = compute_fpr(oof, "y_score_final", "orig_pred_0p5", site3=31)
    site31_patch = compute_fpr(oof, "patched_score_final", "patched_pred_0p5", site3=31)

    metric_rows = []
    for label, vals in [("original_locked_oof", original_metrics), ("patched_site31_mayo_reprocessed14", patched_metrics)]:
        r = {"score_set": label}
        r.update(vals)
        if label.startswith("original"):
            r.update(
                {
                    "philips_cn_fp": phil_orig["numerator"],
                    "philips_cn_n": phil_orig["denominator"],
                    "philips_cn_fpr": phil_orig["fpr"],
                    "site31_cn_fp": site31_orig["numerator"],
                    "site31_cn_n": site31_orig["denominator"],
                    "site31_cn_fpr": site31_orig["fpr"],
                }
            )
        else:
            r.update(
                {
                    "philips_cn_fp": phil_patch["numerator"],
                    "philips_cn_n": phil_patch["denominator"],
                    "philips_cn_fpr": phil_patch["fpr"],
                    "site31_cn_fp": site31_patch["numerator"],
                    "site31_cn_n": site31_patch["denominator"],
                    "site31_cn_fpr": site31_patch["fpr"],
                }
            )
        metric_rows.append(r)
    metrics_df = pd.DataFrame(metric_rows)
    for col in [
        "auc",
        "pr_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "brier",
        "philips_cn_fpr",
        "site31_cn_fpr",
    ]:
        if col in metrics_df:
            metrics_df[f"delta_{col}_vs_original"] = metrics_df[col] - float(metrics_df.loc[0, col])
    metrics_df.to_csv(OUTDIR / "fold_test_counterfactual_patch_metrics.csv", index=False)
    md_from_df(metrics_df, OUTDIR / "fold_test_counterfactual_patch_metrics.md")

    validation_df = pd.DataFrame(validation_rows)
    validation_df.to_csv(OUTDIR / "fold_test_patch_validation.csv", index=False)
    md_from_df(validation_df, OUTDIR / "fold_test_patch_validation.md")
    oof.to_csv(OUTDIR / "patched_oof_score_table.csv", index=False)

    changed_rows = counter[counter["split_role"].eq("classifier_test")].copy()
    cn_changed = changed_rows[changed_rows["diagnosis"].eq("CN")]
    ad_changed = changed_rows[changed_rows["diagnosis"].eq("AD")]
    n_fav = int(changed_rows["favorable_direction"].fillna(False).sum())
    final_lines = [
        "# Locked-Model Counterfactual Recommendation",
        "",
        "This audit used saved promoted fold VAE checkpoints, fold normalization parameters, and saved logreg final classifier pipelines. No VAE or classifier was retrained.",
        "",
        f"- Original tensor SHA256: `{sha256_file(ORIG_TENSOR)}`",
        f"- Replacement tensor SHA256: `{sha256_file(REPL_TENSOR)}`",
        f"- Reprocessed subjects audited: {len(SITE31_SUBJECTS)}",
        f"- Fold-test CN/AD rows patched in OOF table: {len(patch_rows)}",
        f"- Favorable fold-test directions: {n_fav}/{len(changed_rows)}",
        "",
        "## Metric Delta",
    ]
    for col in ["auc", "pr_auc", "balanced_accuracy", "sensitivity", "specificity", "f1", "philips_cn_fpr", "site31_cn_fpr"]:
        delta = metrics_df.loc[1, col] - metrics_df.loc[0, col]
        final_lines.append(f"- {col}: {metrics_df.loc[0, col]:.6f} -> {metrics_df.loc[1, col]:.6f} (delta {delta:+.6f})")
    final_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Scores are interpreted directionally: lower is favorable for CN and higher is favorable for AD. This is a locked-model counterfactual sensitivity, not a promotion run.",
        ]
    )
    if len(cn_changed):
        final_lines.append(
            f"- CN fold-test rows: mean delta {cn_changed['delta_score'].mean():+.6f}; favorable downward rows {int((cn_changed['delta_score'] < 0).sum())}/{len(cn_changed)}."
        )
    if len(ad_changed):
        final_lines.append(
            f"- AD fold-test rows: mean delta {ad_changed['delta_score'].mean():+.6f}; favorable upward rows {int((ad_changed['delta_score'] > 0).sum())}/{len(ad_changed)}."
        )
    if float(metrics_df.loc[1, "auc"]) > float(metrics_df.loc[0, "auc"]) and float(metrics_df.loc[1, "philips_cn_fpr"]) <= float(metrics_df.loc[0, "philips_cn_fpr"]):
        final_lines.append("- Recommendation: the replacement tensor improves the locked-model sensitivity signal and can justify a controlled retraining/predefined sensitivity, but it still is not a trained replacement model.")
    else:
        final_lines.append("- Recommendation: do not treat the replacement tensor as evidence of immediate model improvement; use it to decide whether a controlled retraining sensitivity is scientifically worth launching.")
    (OUTDIR / "final_recommendation.md").write_text("\n".join(final_lines) + "\n", encoding="utf-8")

    command_log.extend(
        [
            {"output": "subject_level_counterfactual_scores.csv", "rows": int(len(counter))},
            {"output": "subgroup_score_shift_summary.csv", "rows": int(len(summary))},
            {"output": "fold_test_counterfactual_patch_metrics.csv", "rows": int(len(metrics_df))},
            {"output": "patched_oof_score_table.csv", "rows": int(len(oof))},
            {"readonly_guardrail": "no model training, no tensor edits, no metadata edits, no prediction edits to source artifacts"},
        ]
    )
    (OUTDIR / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
