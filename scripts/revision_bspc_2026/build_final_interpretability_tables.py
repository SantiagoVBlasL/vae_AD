#!/usr/bin/env python3
"""Build final BSPC interpretability tables from final SHAP/IG artifacts.

This script is read-only with respect to model artifacts, tensors, metadata, and
manuscript files. It consumes the final strict K=200 consensus and fold-level
IG rankings from the promoted model, then writes derived paper tables to a new
results package.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from betavae_xai.interpretability.paper_figures import (  # noqa: E402
    cohen_d,
    edge_lateralization,
    write_csv_md,
)

OUT = PROJECT_ROOT / "results/revision_bspc_2026/final_interpretability_figures_20260624"
AUDIT = PROJECT_ROOT / "results/revision_bspc_2026/final_interpretability_figure_rebuild_audit_20260624"
PREFLIGHT = PROJECT_ROOT / "results/revision_bspc_2026/final_model_shap_ig_preflight_20260624"
STRICT = PROJECT_ROOT / "results/revision_bspc_2026/final_model_shap_ig_strict_consensus_audit_20260624"
MANUSCRIPT_AUDIT = PROJECT_ROOT / "results/revision_bspc_2026/final_shap_ig_manuscript_update_audit_20260624"
RUN_DIR = PROJECT_ROOT / "results/revision_bspc_2026/recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
TENSOR_PATH = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/"
    "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)
METADATA_PATH = PROJECT_ROOT / "results/revision_bspc_2026/adni_035_metadata_rescue_preflight/patched_metadata_candidate.csv"

EXPECTED_CHANNELS = [1, 0, 2]
EXPECTED_CHANNEL_NAMES = [
    "Pearson_Full_FisherZ_Signed",
    "Pearson_OMST_GCE_Signed_Weighted",
    "MI_KNN_Symmetric",
]
EXPECTED_LATENT_DIM = 384
EXPECTED_BETA = 3.75
EXPECTED_K = 200
EXPECTED_N_EDGES = 8
EXPECTED_N_ROIS = 131
FOLDS = [1, 2, 3, 4, 5]
EDGE_KS = [50, 100, 200]
STALE_TOKENS = ["vae_3channels_beta65_pro", "latent_dim=256", "TOP_K_STABILITY=250"]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def fail(msg: str) -> None:
    raise RuntimeError(msg)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        fail(f"Missing required artifact: {path}")
    return pd.read_csv(path)


def validate_no_stale_selected_artifacts() -> None:
    selection = read_csv(AUDIT / "final_artifact_selection.csv")
    selected = selection[selection["selected_for_final_pipeline"].isin(["yes", "supporting"])]
    for _, row in selected.iterrows():
        path_text = str(row["path"])
        for token in STALE_TOKENS:
            if token in path_text:
                fail(f"Selected artifact path contains stale token {token}: {path_text}")
        path = PROJECT_ROOT / path_text
        if not path.exists():
            fail(f"Selected artifact does not exist: {path}")


def validate_final_identity() -> dict:
    identity = read_csv(PREFLIGHT / "final_model_identity_check.csv")
    observed = dict(zip(identity["parameter"], identity["observed"].astype(str)))
    matches = identity["match"].astype(str).str.upper().eq("YES")
    if not matches.all():
        fail("Final model identity check contains non-YES rows.")
    if observed.get("channels_to_use") != "[1, 0, 2]":
        fail(f"Unexpected final channels: {observed.get('channels_to_use')}")
    if int(float(observed.get("latent_dim", "nan"))) != EXPECTED_LATENT_DIM:
        fail(f"Unexpected latent_dim: {observed.get('latent_dim')}")
    if abs(float(observed.get("beta_vae", "nan")) - EXPECTED_BETA) > 1e-9:
        fail(f"Unexpected beta_vae: {observed.get('beta_vae')}")
    if observed.get("vae_final_activation") != "tanh":
        fail(f"Unexpected final activation: {observed.get('vae_final_activation')}")

    summary_path = PREFLIGHT / "full_outputs/full_run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "COMPLETE":
        fail("Full SHAP/IG summary is not COMPLETE.")
    if summary.get("folds") != FOLDS:
        fail(f"Unexpected folds in SHAP/IG summary: {summary.get('folds')}")
    if summary.get("shap_summary", {}).get("shap_all_shape") != [397, 386]:
        fail(f"Unexpected SHAP all shape: {summary.get('shap_summary', {}).get('shap_all_shape')}")
    return summary


def validate_consensus(strict: pd.DataFrame) -> None:
    if len(strict) != EXPECTED_N_EDGES:
        fail(f"Strict consensus must have 8 edges; observed {len(strict)}")
    if set(strict["K"].astype(int)) != {EXPECTED_K}:
        fail(f"Strict consensus K must be 200; observed {sorted(strict['K'].unique())}")
    if (strict["replication_frequency"] < 0.6).any():
        fail("Strict consensus contains replication_frequency < 0.6")
    if (strict["abs_mean_signed_direction"] < 0.6).any():
        fail("Strict consensus contains sign consistency < 0.6")
    if len(strict) == 11:
        fail("Detected stale 11-edge consensus table.")


def copy_final_tables(strict: pd.DataFrame, channels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    table_s5 = read_csv(MANUSCRIPT_AUDIT / "table_s5_consensus_edges_final.csv")
    if len(table_s5) != EXPECTED_N_EDGES:
        fail("Manuscript-ready Table S5 does not contain the expected 8 edges.")
    if set(table_s5["K"].astype(int)) != {EXPECTED_K}:
        fail("Manuscript-ready Table S5 is not K=200.")
    write_csv_md(table_s5, OUT / "table_s5_consensus_edges_final.csv", OUT / "table_s5_consensus_edges_final.md")
    (OUT / "table_s5_consensus_edges_final.tex").write_text(
        table_s5.to_latex(index=False, float_format="%.3f"),
        encoding="utf-8",
    )

    channels_final = read_csv(MANUSCRIPT_AUDIT / "channel_contributions_final.csv")
    if len(channels_final) != 3:
        fail("Final channel contribution table must have exactly 3 rows.")
    write_csv_md(channels_final, OUT / "channel_contributions_final.csv", OUT / "channel_contributions_final.md")
    return table_s5, channels_final


def load_fold_ranking(fold: int) -> pd.DataFrame:
    fold_dir = RUN_DIR / f"fold_{fold}" / "interpretability_logreg"
    args_path = fold_dir / "run_args_saliency_integrated_gradients_top50.json"
    if not args_path.exists():
        fail(f"Missing final fold run args: {args_path}")
    args = json.loads(args_path.read_text(encoding="utf-8"))
    if args.get("channels_to_use") != EXPECTED_CHANNELS:
        fail(f"Fold {fold} has unexpected channels: {args.get('channels_to_use')}")
    if int(args.get("latent_dim")) != EXPECTED_LATENT_DIM:
        fail(f"Fold {fold} has unexpected latent_dim: {args.get('latent_dim')}")
    if args.get("ig_baseline") != "cn_median_train":
        fail(f"Fold {fold} has unexpected IG baseline: {args.get('ig_baseline')}")
    if int(args.get("top_k")) != 50:
        fail(f"Fold {fold} expected IG latent top_k=50 run args.")
    ranking_path = fold_dir / "ranking_conexiones_ANOTADO_integrated_gradients_top50.csv"
    if not ranking_path.exists():
        fail(f"Missing fold ranking: {ranking_path}")
    df = pd.read_csv(ranking_path)
    expected_cols = {"Rank", "ROI_i_name", "ROI_j_name", "Saliency_Signed", "Saliency_Abs"}
    if not expected_cols.issubset(df.columns):
        fail(f"Fold {fold} ranking lacks required columns: {ranking_path}")
    df["fold"] = fold
    return df


def build_lateralization() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for fold in FOLDS:
        ranking = load_fold_ranking(fold).sort_values("Rank")
        for k in EDGE_KS:
            top = ranking.head(k).copy()
            top["lateralization"] = [
                edge_lateralization(a, b) for a, b in zip(top["ROI_i_name"], top["ROI_j_name"])
            ]
            counts = top["lateralization"].value_counts()
            for cat in ["intra_left", "intra_right", "interhemispheric", "midline_or_unknown"]:
                n = int(counts.get(cat, 0))
                rows.append(
                    {
                        "fold": fold,
                        "K": k,
                        "lateralization": cat,
                        "n_edges": n,
                        "fraction": n / float(k),
                    }
                )
    by_fold = pd.DataFrame(rows)
    summary = (
        by_fold.groupby(["K", "lateralization"], as_index=False)
        .agg(
            n_edges_mean=("n_edges", "mean"),
            n_edges_sd=("n_edges", "std"),
            fraction_mean=("fraction", "mean"),
            fraction_sd=("fraction", "std"),
        )
        .sort_values(["K", "lateralization"])
    )
    write_csv_md(by_fold, OUT / "lateralization_by_fold_final.csv", None)
    write_csv_md(summary, OUT / "lateralization_final.csv", OUT / "lateralization_final.md")
    return summary, by_fold


def normalize_edge_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted([str(a), str(b)]))


def lookup_saliency(strict: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for edge_id, row in strict.reset_index(drop=True).iterrows():
        key = normalize_edge_key(row["roi_i"], row["roi_j"])
        fold_abs = []
        fold_signed = []
        fold_ranks = []
        for fold in FOLDS:
            ranking = load_fold_ranking(fold)
            ranking["edge_key"] = [
                normalize_edge_key(a, b) for a, b in zip(ranking["ROI_i_name"], ranking["ROI_j_name"])
            ]
            sub = ranking[ranking["edge_key"] == key]
            if sub.empty:
                continue
            hit = sub.sort_values("Rank").iloc[0]
            fold_abs.append(float(hit["Saliency_Abs"]))
            fold_signed.append(float(hit["Saliency_Signed"]))
            fold_ranks.append(int(hit["Rank"]))
        if len(fold_abs) != len(FOLDS):
            fail(f"Consensus edge missing from at least one fold ranking: {row['roi_i']} - {row['roi_j']}")
        sal_mean = float(np.mean(fold_abs))
        sal_max = float(np.max(fold_abs))
        source = "fold_ranking_saliency_abs_integrated_gradients_top50"
        rows.append(
            {
                "edge_id": edge_id + 1,
                "roi_i": row["roi_i"],
                "roi_j": row["roi_j"],
                "saliency_abs": sal_mean,
                "saliency_abs_max_fold": sal_max,
                "saliency_signed_mean": float(np.mean(fold_signed)),
                "rank_mean": float(np.mean(fold_ranks)),
                "rank_min": int(np.min(fold_ranks)),
                "rank_max": int(np.max(fold_ranks)),
                "saliency_source": source,
            }
        )
    return pd.DataFrame(rows)


def load_tensor_and_metadata() -> tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    if not TENSOR_PATH.exists():
        fail(f"Missing tensor: {TENSOR_PATH}")
    z = np.load(TENSOR_PATH, allow_pickle=True)
    tensor = z["global_tensor_data"]
    subject_ids = z["subject_ids"].astype(str)
    channel_names = list(z["channel_names"].astype(str))
    if tensor.shape != (648, 7, EXPECTED_N_ROIS, EXPECTED_N_ROIS):
        fail(f"Unexpected tensor shape: {tensor.shape}")
    if [channel_names[i] for i in EXPECTED_CHANNELS] != EXPECTED_CHANNEL_NAMES:
        fail(f"Unexpected channel names for [1,0,2]: {[channel_names[i] for i in EXPECTED_CHANNELS]}")
    metadata = read_csv(METADATA_PATH)
    if len(metadata) != 647:
        fail(f"Unexpected metadata N: {len(metadata)}")
    roi_info = read_csv(RUN_DIR / "roi_info_from_tensor.csv")
    if len(roi_info) != EXPECTED_N_ROIS:
        fail(f"Unexpected ROI count: {len(roi_info)}")
    return tensor, subject_ids, metadata, roi_info


def build_cohen_tables(strict: pd.DataFrame, table_s5: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    tensor, _subject_ids, _metadata, roi_info = load_tensor_and_metadata()
    roi_to_idx = dict(zip(roi_info["roi_name_in_tensor"], range(len(roi_info))))
    saliency = lookup_saliency(strict)

    edge_rows = []
    fold_rows = []
    for edge_id, row in table_s5.iterrows():
        roi_i = str(row["roi_i"])
        roi_j = str(row["roi_j"])
        if roi_i not in roi_to_idx or roi_j not in roi_to_idx:
            fail(f"Consensus edge ROI missing from ROI map: {roi_i}, {roi_j}")
        i = roi_to_idx[roi_i]
        j = roi_to_idx[roi_j]
        pooled_ad = []
        pooled_cn = []
        for fold in FOLDS:
            subjects_path = RUN_DIR / f"fold_{fold}" / "test_subjects_fold.csv"
            test_df = read_csv(subjects_path)
            test_df = test_df[test_df["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
            vals = tensor[test_df["tensor_idx"].astype(int).values][:, EXPECTED_CHANNELS, i, j].mean(axis=1)
            test_df["edge_value_channel_mean_102"] = vals
            ad_vals = test_df.loc[test_df["ResearchGroup_Mapped"] == "AD", "edge_value_channel_mean_102"].values
            cn_vals = test_df.loc[test_df["ResearchGroup_Mapped"] == "CN", "edge_value_channel_mean_102"].values
            d_fold = cohen_d(ad_vals, cn_vals)
            fold_rows.append(
                {
                    "edge_id": int(row["edge_id"]),
                    "fold": fold,
                    "roi_i": roi_i,
                    "roi_j": roi_j,
                    "n_ad": len(ad_vals),
                    "n_cn": len(cn_vals),
                    "cohen_d_ad_minus_cn": d_fold,
                    "mean_ad": float(np.mean(ad_vals)) if len(ad_vals) else np.nan,
                    "mean_cn": float(np.mean(cn_vals)) if len(cn_vals) else np.nan,
                }
            )
            pooled_ad.extend(ad_vals.tolist())
            pooled_cn.extend(cn_vals.tolist())
        d_pool = cohen_d(pooled_ad, pooled_cn)
        edge_rows.append(
            {
                "edge_id": int(row["edge_id"]),
                "roi_i": roi_i,
                "roi_j": roi_j,
                "network_i": row["network_i"],
                "network_j": row["network_j"],
                "direction": row["direction"],
                "replication_frequency": row["replication_frequency"],
                "mean_signed_direction": row["mean_signed_direction"],
                "n_ad_pooled": len(pooled_ad),
                "n_cn_pooled": len(pooled_cn),
                "cohen_d_pooled_ad_minus_cn": d_pool,
                "abs_cohen_d_pooled": abs(d_pool) if np.isfinite(d_pool) else np.nan,
                "effect_value_definition": "mean selected-channel [1,0,2] edge value in held-out outer-test subjects",
            }
        )

    cohen_df = pd.DataFrame(edge_rows)
    fold_df = pd.DataFrame(fold_rows)
    fold_summary = (
        fold_df.groupby("edge_id", as_index=False)
        .agg(
            cohen_d_fold_mean=("cohen_d_ad_minus_cn", "mean"),
            cohen_d_fold_sd=("cohen_d_ad_minus_cn", "std"),
        )
    )
    cohen_df = cohen_df.merge(fold_summary, on="edge_id", how="left")
    cohen_df = cohen_df.merge(saliency, on=["edge_id", "roi_i", "roi_j"], how="left")

    write_csv_md(cohen_df, OUT / "cohen_d_consensus8_final.csv", OUT / "cohen_d_consensus8_final.md")

    x = cohen_df["saliency_abs"].astype(float).values
    y = cohen_df["abs_cohen_d_pooled"].astype(float).values
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() != EXPECTED_N_EDGES:
        rho = np.nan
        pval = np.nan
    else:
        rho, pval = spearmanr(x[finite], y[finite])
    saliency_vs = cohen_df[
        [
            "edge_id",
            "roi_i",
            "roi_j",
            "direction",
            "replication_frequency",
            "saliency_abs",
            "saliency_abs_max_fold",
            "saliency_signed_mean",
            "rank_mean",
            "rank_min",
            "rank_max",
            "abs_cohen_d_pooled",
            "cohen_d_pooled_ad_minus_cn",
            "saliency_source",
        ]
    ].copy()
    saliency_vs["spearman_rho"] = float(rho) if np.isfinite(rho) else np.nan
    saliency_vs["spearman_p"] = float(pval) if np.isfinite(pval) else np.nan
    saliency_vs["n_edges_for_correlation"] = int(finite.sum())
    write_csv_md(
        saliency_vs,
        OUT / "saliency_vs_cohen_d_consensus8_final.csv",
        OUT / "saliency_vs_cohen_d_consensus8_final.md",
    )
    return cohen_df, saliency_vs


def build_network_signature(table_s5: pd.DataFrame) -> pd.DataFrame:
    df = table_s5.copy()
    df["signed_weight"] = df["mean_signed_direction"].astype(float)
    out = (
        df.groupby(["network_i", "network_j"], as_index=False)
        .agg(
            n_edges=("edge_id", "count"),
            signed_weight_sum=("signed_weight", "sum"),
            pro_ad_edges=("direction", lambda s: int((s == "Pro-AD").sum())),
            pro_cn_edges=("direction", lambda s: int((s == "Pro-CN").sum())),
        )
        .sort_values(["n_edges", "signed_weight_sum"], ascending=[False, False])
    )
    write_csv_md(out, OUT / "network_pair_signature_final.csv", OUT / "network_pair_signature_final.md")
    return out


def build_outputs_index() -> pd.DataFrame:
    rows = []
    for path in sorted(OUT.iterdir()):
        if path.name == "__pycache__":
            continue
        if path.is_file():
            rows.append(
                {
                    "path": rel(path),
                    "bytes": path.stat().st_size,
                    "role": "generated table/report/figure package artifact",
                }
            )
    df = pd.DataFrame(rows)
    write_csv_md(df, OUT / "outputs_index.csv", OUT / "outputs_index.md")
    return df


def write_final_report(
    table_s5: pd.DataFrame,
    channels_final: pd.DataFrame,
    lateralization: pd.DataFrame,
    saliency_vs: pd.DataFrame,
    summary: dict,
) -> None:
    strict_summary = read_csv(STRICT / "strict_consensus_summary.csv")
    k200 = strict_summary[strict_summary["K"].astype(int) == 200].iloc[0]
    pro_ad = int((table_s5["direction"] == "Pro-AD").sum())
    pro_cn = int((table_s5["direction"] == "Pro-CN").sum())
    rho = saliency_vs["spearman_rho"].dropna().iloc[0]
    pval = saliency_vs["spearman_p"].dropna().iloc[0]
    channel_lines = [
        f"- {row.channel_label}: {row.pct:.1f}%"
        for row in channels_final.itertuples(index=False)
    ]
    lat_text = lateralization.pivot(index="lateralization", columns="K", values="fraction_mean").fillna(0)
    report = f"""# Final Interpretability Figure Report

## Provenance

- Model: promoted FULL `[1,0,2]`, latent_dim=384, beta=3.75, classifier=logreg.
- SHAP/IG status: {summary.get('status')}; folds={summary.get('folds')}.
- SHAP shape: {summary.get('shap_summary', {}).get('shap_all_shape')}.
- IG baseline: `cn_median_train`.
- Consensus rule: strict Top-K=200, pi>=0.6, sign consistency>=0.6.

## Manuscript Numbers

- Consensus edges: {len(table_s5)}
- Direction split: {pro_ad} Pro-AD, {pro_cn} Pro-CN
- Mean Top-200 Jaccard: {float(k200['mean_jaccard']):.6f}
- Saliency-vs-effect Spearman rho: {float(rho):.3f}
- Saliency-vs-effect Spearman p-value: {float(pval):.3f}

## Channel Contributions

{chr(10).join(channel_lines)}

## Lateralization Fractions

{lat_text.to_markdown()}

## Output Figure Paths

- `Figure3_main_signature_final.pdf/.png/.svg`
- `Figure4_glass_brain_final.pdf/.png/.svg`

## Interpretation Guardrail

Consensus edge saliency is multivariate model-derived. Cohen's d is an
empirical univariate effect computed on held-out outer-test subjects. The two
quantities are complementary and should not be described as equivalent.
"""
    (OUT / "final_report.md").write_text(report, encoding="utf-8")


def write_command_log() -> None:
    log = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": rel(Path(__file__)),
        "output_dir": rel(OUT),
        "guardrails": {
            "did_train_vae": False,
            "did_retrain_classifier": False,
            "did_run_shap_ig": False,
            "did_modify_tensors": False,
            "did_modify_metadata": False,
            "did_modify_manuscript": False,
        },
    }
    (OUT / "command_log_tables.json").write_text(json.dumps(log, indent=2), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    validate_no_stale_selected_artifacts()
    summary = validate_final_identity()
    strict = read_csv(STRICT / "strict_consensus_topK200.csv")
    validate_consensus(strict)
    table_s5, channels_final = copy_final_tables(strict, read_csv(STRICT / "channel_contributions.csv"))
    lateralization, _by_fold = build_lateralization()
    cohen_df, saliency_vs = build_cohen_tables(strict, table_s5)
    _network = build_network_signature(table_s5)
    write_final_report(table_s5, channels_final, lateralization, saliency_vs, summary)
    write_command_log()
    build_outputs_index()
    print(f"Tables written to: {OUT}")
    print((OUT / "outputs_index.csv").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
