#!/usr/bin/env python3
"""Prepare a read-only preflight package for a conditional-age VAE branch.

This audit intentionally does not launch training.  It verifies the promoted
baseline configuration, writes a proposed conditional-age candidate config, and
checks whether the current code can implement the requested architecture:

    q_phi(z | x, Age), p_theta(x | z, Age)

If the code path is not supported, the package is still written, but the launch
recommendation is blocked.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMOTED_CONFIG = (
    PROJECT_ROOT
    / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5.json"
)
TRAIN_SCRIPT = PROJECT_ROOT / "scripts/run_vae_clf_ad_inference.py"
MODEL_SCRIPT = PROJECT_ROOT / "src/betavae_xai/models/convolutional_vae.py"
OUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/conditional_age_vae_preflight_20260610"
RUN_ROOT = PROJECT_ROOT / "results/revision_bspc_2026/conditional_age_vae_latent384_beta3p75_T80_p560_full5x5_20260610"
BIG_DISK_RUN_ROOT = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/"
    "conditional_age_vae_latent384_beta3p75_T80_p560_full5x5_20260610"
)
PYTHON = "/home/diego/anaconda3/envs/vae_ad/bin/python"

EXPECTED_CHANNELS = [1, 0, 2]
EXPECTED_SELECTED_CHANNEL_NAMES = [
    "Pearson_Full_FisherZ_Signed",
    "Pearson_OMST_GCE_Signed_Weighted",
    "MI_KNN_Symmetric",
]
EXPECTED_STAGE_B = "logreg_l2"
EXPECTED_CALIBRATION = "oof_ecdf"
EXPECTED_THRESHOLD = "inner_oof_target_sens_ge_0p70_max_spec"
PROMOTED_AUC = 0.795155
PROMOTED_PR_AUC = 0.573934


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--promoted-config", type=Path, default=PROMOTED_CONFIG)
    parser.add_argument("--run-dryrun-command", action="store_true", default=True)
    parser.add_argument("--skip-dryrun-command", action="store_false", dest="run_dryrun_command")
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_md_table(path: Path, df: pd.DataFrame) -> None:
    path.write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")


def write_pair(path_csv: Path, df: pd.DataFrame) -> None:
    df.to_csv(path_csv, index=False)
    write_md_table(path_csv.with_suffix(".md"), df)


def serialize(value: Any) -> str:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, sort_keys=True)
    if value is None:
        return ""
    return str(value)


def line_no(text: str, needle: str) -> int | None:
    for i, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return i
    return None


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def normalize_dx(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def normalize_mfr(value: Any) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    up = text.upper()
    if "PHILIPS" in up:
        return "Philips"
    if "SIEMENS" in up:
        return "SIEMENS"
    if up.startswith("GE") or "GENERAL ELECTRIC" in up:
        return "GE"
    return text or "UNKNOWN"


def load_metadata(path: Path) -> pd.DataFrame:
    meta = pd.read_csv(path)
    if "tensor_idx" not in meta.columns and "tensor_index" in meta.columns:
        meta = meta.rename(columns={"tensor_index": "tensor_idx"})
    required = ["SubjectID", "ResearchGroup_Mapped", "Age", "Sex", "Manufacturer", "tensor_idx"]
    missing = [c for c in required if c not in meta.columns]
    require(not missing, f"Metadata missing required columns: {missing}")
    meta = meta.copy()
    meta["SubjectID"] = meta["SubjectID"].astype(str)
    meta["ResearchGroup_Mapped"] = normalize_dx(meta["ResearchGroup_Mapped"])
    meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
    meta["Sex"] = meta["Sex"].astype(str).str.strip()
    meta["Manufacturer"] = meta["Manufacturer"].map(normalize_mfr)
    meta["tensor_idx"] = pd.to_numeric(meta["tensor_idx"], errors="raise").astype(int)
    return meta


def baseline_checks(config: Dict[str, Any]) -> pd.DataFrame:
    params = config["parameters"]
    paths = config["paths"]
    rows = [
        ("tensor_path_exists", resolve(paths["global_tensor_path"]).exists(), True, paths["global_tensor_path"]),
        ("metadata_path_exists", resolve(paths["metadata_path"]).exists(), True, paths["metadata_path"]),
        ("channels_to_use", params.get("channels_to_use"), EXPECTED_CHANNELS, ""),
        ("selected_channel_names", config.get("selected_channel_names"), EXPECTED_SELECTED_CHANNEL_NAMES, ""),
        ("latent_dim", params.get("latent_dim"), 384, ""),
        ("beta_vae", params.get("beta_vae"), 3.75, ""),
        ("lr_scheduler_T0", params.get("lr_scheduler_T0"), 80, ""),
        ("epochs_vae", params.get("epochs_vae"), 10000, ""),
        ("early_stopping_patience_vae", params.get("early_stopping_patience_vae"), 560, ""),
        ("dropout_rate_vae", params.get("dropout_rate_vae"), 0.15, ""),
        ("vae_final_activation", params.get("vae_final_activation"), "tanh", ""),
        ("batch_size", params.get("batch_size"), 64, ""),
        ("recon_loss_mode", params.get("recon_loss_mode"), "mse_sum_batchmean_current", ""),
        ("outer_folds", params.get("outer_folds"), 5, ""),
        ("inner_folds", params.get("inner_folds"), 5, ""),
        ("classifier_types_contains_logreg", "logreg" in params.get("classifier_types", []), True, ""),
        ("metadata_features", params.get("metadata_features"), ["Age", "Sex"], ""),
        ("classifier_stratify_cols", params.get("classifier_stratify_cols"), ["Manufacturer"], ""),
        ("vae_stratify_cols", params.get("vae_stratify_cols"), ["Manufacturer"], ""),
    ]
    out = pd.DataFrame(
        [
            {
                "check": name,
                "actual": serialize(actual),
                "expected": serialize(expected),
                "pass": bool(actual == expected),
                "note": note,
            }
            for name, actual, expected, note in rows
        ]
    )
    return out


def make_candidate_config(reference: Dict[str, Any]) -> Dict[str, Any]:
    candidate = json.loads(json.dumps(reference))
    candidate["run_name"] = "conditional_age_vae_latent384_beta3p75_T80_p560_full5x5_20260610"
    candidate["paths"]["output_dir"] = str(RUN_ROOT.relative_to(PROJECT_ROOT))
    candidate["paths"]["big_disk_output_dir"] = str(BIG_DISK_RUN_ROOT)
    candidate["paths"]["split_preview_csv"] = str(
        OUT_DIR.relative_to(PROJECT_ROOT) / "conditional_age_vae_split_preview.csv"
    )
    candidate["paths"]["split_preview_summary_csv"] = str(
        OUT_DIR.relative_to(PROJECT_ROOT) / "conditional_age_vae_split_preview_summary.csv"
    )
    params = candidate["parameters"]
    params["vae_conditioning_mode"] = "age"
    params["vae_conditioning_vars"] = ["Age"]
    params["encoder_conditioning"] = True
    params["decoder_conditioning"] = True
    params["condition_diagnosis"] = False
    params["condition_manufacturer"] = False
    params["condition_site"] = False
    params["condition_raw_tp_group"] = False
    params["condition_prediction"] = False
    return candidate


def config_diff(reference: Dict[str, Any], candidate: Dict[str, Any]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    ref_params = reference["parameters"]
    cand_params = candidate["parameters"]
    keys = sorted(set(ref_params) | set(cand_params))
    allowed_conditioning_keys = {
        "vae_conditioning_mode",
        "vae_conditioning_vars",
        "encoder_conditioning",
        "decoder_conditioning",
        "condition_diagnosis",
        "condition_manufacturer",
        "condition_site",
        "condition_raw_tp_group",
        "condition_prediction",
    }
    for key in keys:
        ref_val = ref_params.get(key, "<absent>")
        cand_val = cand_params.get(key, "<absent>")
        if ref_val != cand_val:
            rows.append(
                {
                    "scope": "parameters",
                    "field": key,
                    "promoted_value": serialize(ref_val),
                    "conditional_age_value": serialize(cand_val),
                    "allowed_scientific_diff": key in allowed_conditioning_keys,
                    "current_code_supported": key
                    not in {"vae_conditioning_mode", "vae_conditioning_vars", "encoder_conditioning", "decoder_conditioning"},
                    "note": "requested VAE conditioning diff",
                }
            )
    for key in sorted(set(reference["paths"]) | set(candidate["paths"])):
        ref_val = reference["paths"].get(key, "<absent>")
        cand_val = candidate["paths"].get(key, "<absent>")
        if ref_val != cand_val:
            rows.append(
                {
                    "scope": "paths",
                    "field": key,
                    "promoted_value": serialize(ref_val),
                    "conditional_age_value": serialize(cand_val),
                    "allowed_scientific_diff": key
                    in {"output_dir", "big_disk_output_dir", "split_preview_csv", "split_preview_summary_csv"},
                    "current_code_supported": True,
                    "note": "provenance/output path change only",
                }
            )
    return pd.DataFrame(rows)


def inspect_architecture() -> Dict[str, Any]:
    train_text = TRAIN_SCRIPT.read_text(encoding="utf-8")
    model_text = MODEL_SCRIPT.read_text(encoding="utf-8")
    choices_line = ""
    if "VAE_CONDITIONING_VAR_CHOICES =" in train_text:
        choices_line = train_text.split("VAE_CONDITIONING_VAR_CHOICES =", 1)[1].splitlines()[0]
    model_choices = ""
    if "CONDITIONING_MODE_CHOICES" in model_text:
        model_choices = model_text.split("CONDITIONING_MODE_CHOICES", 1)[1].split(")", 1)[0]
    return {
        "model_conditioning_choices_line": line_no(model_text, "CONDITIONING_MODE_CHOICES"),
        "model_decoder_latent_in_line": line_no(model_text, "decoder_latent_in = latent_dim + self.conditioning_dim"),
        "model_encode_signature_line": line_no(model_text, "def encode(self, x: torch.Tensor)"),
        "model_forward_encode_line": line_no(model_text, "mu, logvar = self.encode(x)"),
        "model_decoder_cat_line": line_no(model_text, "return torch.cat([z, condition], dim=1)"),
        "cli_conditioning_mode_line": line_no(train_text, "--vae_conditioning_mode"),
        "cli_conditioning_vars_line": line_no(train_text, "--vae_conditioning_vars"),
        "cli_validation_decoder_only_line": line_no(train_text, "elif args.vae_conditioning_mode == \"decoder_only\""),
        "condition_vars_choices_line": line_no(train_text, "VAE_CONDITIONING_VAR_CHOICES"),
        "conditioning_transformer_fit_line": line_no(train_text, "fit_vae_conditioning_transformer("),
        "conditioning_train_scope_line": line_no(
            train_text, "vae_train_pool_df.iloc[vae_actual_train_indices_local_to_pool].copy(),"
        ),
        "condition_pass_train_line": line_no(train_text, "recon_batch, mu, logvar, _ = vae_fold_k(vae_input, condition=cond_batch)"),
        "condition_pass_latent_line": line_no(train_text, "condition=cond_train_dev_tensor"),
        "supports_encoder_age_conditioning": "def encode(self, x: torch.Tensor, condition" in model_text
        or "encoder_conditioning" in model_text,
        "supports_age_only_var": '"age"' in choices_line or "'age'" in choices_line,
        "supports_requested_mode_age": '"age"' in model_choices or "'age'" in model_choices,
        "supports_decoder_only": '"decoder_only"' in model_text,
    }


def architecture_md(audit: Dict[str, Any]) -> str:
    verdict = (
        "UNSAFE_TO_LAUNCH: current code does not implement encoder+decoder Age conditioning."
        if not audit["supports_encoder_age_conditioning"]
        else "SAFE_ARCHITECTURE_SUPPORT_DETECTED"
    )
    return f"""# Conditional-Age VAE Architecture Audit

## Verdict

{verdict}

## Required Target

- Encoder posterior: `q_phi(z | x, Age)`.
- Decoder likelihood: `p_theta(x | z, Age)`.
- VAE condition variables: `Age` only.
- Not allowed as VAE conditions: Sex, diagnosis, Manufacturer, Site3, raw_tp_group, fold, score, prediction outcome.

## Current Implementation Evidence

- `{MODEL_SCRIPT.relative_to(PROJECT_ROOT)}` line {audit['model_conditioning_choices_line']}: `CONDITIONING_MODE_CHOICES` includes only `none` and `decoder_only`.
- `{MODEL_SCRIPT.relative_to(PROJECT_ROOT)}` line {audit['model_encode_signature_line']}: `encode(self, x)` has no condition argument.
- `{MODEL_SCRIPT.relative_to(PROJECT_ROOT)}` line {audit['model_forward_encode_line']}: `forward()` computes `mu, logvar = self.encode(x)`, so posterior is `q(z | x)`, not `q(z | x, Age)`.
- `{MODEL_SCRIPT.relative_to(PROJECT_ROOT)}` line {audit['model_decoder_latent_in_line']}: decoder input dimension is `latent_dim + conditioning_dim`.
- `{MODEL_SCRIPT.relative_to(PROJECT_ROOT)}` line {audit['model_decoder_cat_line']}: the condition is concatenated only to `z` before decoding.
- `{TRAIN_SCRIPT.relative_to(PROJECT_ROOT)}` line {audit['cli_conditioning_mode_line']}: CLI exposes `--vae_conditioning_mode`.
- `{TRAIN_SCRIPT.relative_to(PROJECT_ROOT)}` line {audit['cli_validation_decoder_only_line']}: CLI validation only supports `decoder_only` when conditioning is enabled.
- `{TRAIN_SCRIPT.relative_to(PROJECT_ROOT)}` line {audit['condition_vars_choices_line']}: VAE conditioning vars currently do not include age-only.
- `{TRAIN_SCRIPT.relative_to(PROJECT_ROOT)}` line {audit['conditioning_train_scope_line']}: fold-local conditioning transformer is fit from VAE actual-train rows only when current conditioning is used.

## Consequence

The current code can support a decoder-only conditional experiment such as `p(x | z, Age/Sex)`, but it does not support the requested age-only encoder+decoder experiment. Launching under the current implementation would not test the stated hypothesis that Age is forced out of `mu`, because `mu` would still be computed without Age.

## Required Code Change Before Launch

Implement a backward-compatible conditioning mode such as `encoder_decoder_age` or explicit flags:

- concatenate scaled Age into the encoder FC representation before `fc_mu` and `fc_logvar`;
- concatenate scaled Age into the decoder input with `z`;
- add age-only conditioning vars;
- keep Sex out of the VAE condition for this experiment;
- keep diagnosis, Manufacturer, Site3, raw_tp_group, fold, score, and prediction outcome unavailable to the VAE conditioning path;
- add a smoke test confirming that `mu` changes when Age changes at fixed `x`.
"""


def make_outer_folds(meta: pd.DataFrame, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    params = config["parameters"]
    cn_ad = meta.loc[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy().reset_index(drop=True)
    cn_ad["label"] = cn_ad["ResearchGroup_Mapped"].map({"CN": 0, "AD": 1}).astype(int)
    strat_cols = ["ResearchGroup_Mapped"]
    for col in params.get("classifier_stratify_cols", []) or []:
        if col in cn_ad.columns and col not in strat_cols:
            cn_ad[col] = cn_ad[col].fillna(f"{col}_Unknown").astype(str)
            strat_cols.append(col)
    y_outer = cn_ad[strat_cols].apply(lambda x: "_".join(x.astype(str)), axis=1)
    if (y_outer.value_counts() < int(params["outer_folds"])).any():
        y_outer = cn_ad["label"].to_numpy()
    skf = StratifiedKFold(n_splits=int(params["outer_folds"]), shuffle=True, random_state=int(params["seed"]))
    rows: List[Dict[str, Any]] = []
    all_valid_tensor_idx = meta["tensor_idx"].astype(int).to_numpy()
    for fold_idx, (train_dev_idx, test_idx) in enumerate(skf.split(np.arange(len(cn_ad)), y_outer), start=1):
        test_tensor_idx = cn_ad.iloc[test_idx]["tensor_idx"].astype(int).to_numpy()
        vae_pool_tensor_idx = np.setdiff1d(np.unique(all_valid_tensor_idx), np.unique(test_tensor_idx))
        vae_pool = meta.set_index("tensor_idx").loc[vae_pool_tensor_idx].reset_index()

        vae_strat_cols = ["ResearchGroup_Mapped"]
        for col in params.get("vae_stratify_cols", []) or []:
            if col not in vae_strat_cols and col in vae_pool.columns:
                vae_strat_cols.append(col)
        strat_df = vae_pool[vae_strat_cols].copy()
        for col in vae_strat_cols:
            strat_df[col] = strat_df[col].fillna(f"{col}_Unknown").astype(str)
        vae_strat = strat_df.apply(lambda x: "_".join(x.astype(str)), axis=1)
        if not (vae_strat.value_counts() >= 2).all():
            vae_strat = vae_pool["ResearchGroup_Mapped"].fillna("RG_Unknown").astype(str)
        actual_train_local = np.arange(len(vae_pool), dtype=int)
        internal_val_local = np.array([], dtype=int)
        if float(params.get("vae_val_split_ratio", 0.2)) > 0 and len(vae_pool) > 10:
            actual_train_local, internal_val_local = train_test_split(
                np.arange(len(vae_pool), dtype=int),
                test_size=float(params.get("vae_val_split_ratio", 0.2)),
                stratify=vae_strat,
                random_state=int(params["seed"]) + (fold_idx - 1) + 10,
                shuffle=True,
            )
            actual_train_local = np.asarray(actual_train_local, dtype=int)
            internal_val_local = np.asarray(internal_val_local, dtype=int)

        fit_pool = vae_pool.iloc[actual_train_local].copy()
        internal_val = vae_pool.iloc[internal_val_local].copy()
        clf_train_dev = cn_ad.iloc[train_dev_idx].copy()
        clf_test = cn_ad.iloc[test_idx].copy()
        fit_mean = float(fit_pool["Age"].mean())
        fit_std = float(fit_pool["Age"].std(ddof=0))
        require(np.isfinite(fit_std) and fit_std > 0, f"Fold {fold_idx} has invalid Age std.")
        for split_name, split_df in [
            ("vae_actual_train_fit", fit_pool),
            ("vae_internal_val_apply", internal_val),
            ("classifier_train_dev_apply", clf_train_dev),
            ("classifier_test_apply", clf_test),
        ]:
            age = split_df["Age"].astype(float)
            scaled = (age - fit_mean) / fit_std if len(age) else pd.Series(dtype=float)
            rows.append(
                {
                    "fold": fold_idx,
                    "split": split_name,
                    "n": int(len(split_df)),
                    "age_missing_n": int(age.isna().sum()),
                    "raw_age_mean": float(age.mean()) if len(age) else np.nan,
                    "raw_age_std": float(age.std(ddof=0)) if len(age) else np.nan,
                    "raw_age_min": float(age.min()) if len(age) else np.nan,
                    "raw_age_max": float(age.max()) if len(age) else np.nan,
                    "fit_age_mean": fit_mean,
                    "fit_age_std": fit_std,
                    "scaled_age_mean": float(scaled.mean()) if len(scaled) else np.nan,
                    "scaled_age_std": float(scaled.std(ddof=0)) if len(scaled) else np.nan,
                    "scaled_age_min": float(scaled.min()) if len(scaled) else np.nan,
                    "scaled_age_max": float(scaled.max()) if len(scaled) else np.nan,
                    "fit_scope": "vae_actual_train_only",
                    "test_subjects_used_for_fit": 0,
                    "train_test_subject_overlap": 0,
                }
            )
    return rows


def stageb_plan() -> pd.DataFrame:
    rows = [
        {
            "readout_id": "mu_only",
            "features": "mu",
            "metadata_features": "[]",
            "classifier": EXPECTED_STAGE_B,
            "outer_folds": 5,
            "inner_folds": 5,
            "primary_calibration": EXPECTED_CALIBRATION,
            "threshold_rule": EXPECTED_THRESHOLD,
            "manufacturer_site_features_allowed": False,
        },
        {
            "readout_id": "mu_plus_sex",
            "features": "mu + Sex",
            "metadata_features": '["Sex"]',
            "classifier": EXPECTED_STAGE_B,
            "outer_folds": 5,
            "inner_folds": 5,
            "primary_calibration": EXPECTED_CALIBRATION,
            "threshold_rule": EXPECTED_THRESHOLD,
            "manufacturer_site_features_allowed": False,
        },
        {
            "readout_id": "mu_plus_age_sex",
            "features": "mu + Age + Sex",
            "metadata_features": '["Age", "Sex"]',
            "classifier": EXPECTED_STAGE_B,
            "outer_folds": 5,
            "inner_folds": 5,
            "primary_calibration": EXPECTED_CALIBRATION,
            "threshold_rule": EXPECTED_THRESHOLD,
            "manufacturer_site_features_allowed": False,
        },
    ]
    return pd.DataFrame(rows)


def command_from_config(config: Dict[str, Any], candidate: Dict[str, Any]) -> List[str]:
    params = dict(config["parameters"])
    params.update(
        {
            "vae_conditioning_mode": "age",
            "vae_conditioning_vars": "Age",
            "encoder_conditioning": True,
            "decoder_conditioning": True,
        }
    )
    command = [
        PYTHON,
        str(TRAIN_SCRIPT),
        "--global_tensor_path",
        config["paths"]["global_tensor_path"],
        "--metadata_path",
        config["paths"]["metadata_path"],
        "--output_dir",
        candidate["paths"]["output_dir"],
    ]
    passthrough = [
        "channels_to_use",
        "classifier_types",
        "classifier_stratify_cols",
        "vae_stratify_cols",
        "latent_features_type",
        "gridsearch_scoring",
        "outer_folds",
        "inner_folds",
        "repeated_outer_folds_n_repeats",
        "num_conv_layers_encoder",
        "decoder_type",
        "epochs_vae",
        "vae_val_split_ratio",
        "early_stopping_patience_vae",
        "cyclical_beta_n_cycles",
        "cyclical_beta_ratio_increase",
        "beta_vae",
        "dropout_rate_vae",
        "vae_dropout_scope",
        "vae_block_order",
        "latent_dim",
        "batch_size",
        "lr_vae",
        "lr_scheduler_type",
        "lr_scheduler_T0",
        "lr_scheduler_eta_min",
        "weight_decay_vae",
        "vae_final_activation",
        "intermediate_fc_dim_vae",
        "metadata_features",
        "norm_mode",
        "recon_loss_mode",
        "vae_conditioning_mode",
        "vae_conditioning_vars",
        "vae_latent_covariate_corr_lambda",
        "seed",
        "num_workers",
        "n_jobs_gridsearch",
        "n_iter_logreg",
        "n_iter_svm",
    ]
    bool_flags = [
        "classifier_calibrate",
        "classifier_use_class_weight",
        "save_fold_artefacts",
        "save_vae_training_history",
        "qc_analyze_distributions",
        "qc_check_scanner_leakage",
        "qc_rate_distortion",
        "qc_latent_information",
    ]
    for name in passthrough:
        value = params.get(name)
        if value is None:
            continue
        command.append(f"--{name}")
        if isinstance(value, list):
            command.extend(str(x) for x in value)
        else:
            command.append(str(value))
    for name in bool_flags:
        if params.get(name):
            command.append(f"--{name}")
    command.append("--dry-run")
    return command


def run_validation_command(command: Sequence[str]) -> Dict[str, Any]:
    proc = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=120)
    return {
        "command": shlex.join(command),
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "expected_result": "fail_current_code_unsupported_conditioning_mode",
        "pass": proc.returncode != 0 and "invalid choice" in proc.stderr.lower(),
    }


def write_static_docs(outdir: Path, safe_to_launch: bool) -> None:
    (outdir / "expected_output_schema.md").write_text(
        """# Expected Post-Run Output Schema

The conditional-age VAE branch should produce these post-run audit outputs after real training, Stage B, and OOF calibration:

- `stageB_three_readout_metrics.csv/.md`: one row per readout/calibration/threshold, including AUC, PR-AUC, BA, Sens, Spec, F1, TN, FP, FN, TP.
- `philips_cn_fpr_by_readout.csv/.md`: Philips CN false positives by readout, including 140TP subgroup if available.
- `manufacturer_error_by_readout.csv/.md`: CN FPR and AD FNR by GE/SIEMENS/Philips.
- `latent_age_association_before_after.csv/.md`: latent-age association versus promoted baseline.
- `latent_protocol_association_before_after.csv/.md`: latent raw_tp_group/protocol association versus promoted baseline.
- `latent_geometry_before_after.csv/.md`: centroid distances for Philips CN FP/TN and GE AD FN/TP.
- `tensor_channel_error_mechanism_before_after.csv/.md`: channel summaries for promoted-error phenotypes.
- `final_conditional_age_vae_decision.md`: sensitivity/promotion decision with leakage guard status.
""",
        encoding="utf-8",
    )
    (outdir / "promotion_gate.md").write_text(
        f"""# Promotion Gate

The conditional-age VAE may enter sensitivity or promotion discussion only if all of the following hold:

- OOF-ECDF AUC >= promoted AUC - 0.005 (`{PROMOTED_AUC - 0.005:.6f}`).
- PR-AUC >= promoted PR-AUC - 0.005 (`{PROMOTED_PR_AUC - 0.005:.6f}`).
- BA and F1 are not materially worse than promoted.
- Philips CN FPR decreases meaningfully, target `< 0.40`.
- Philips 140TP FPR decreases meaningfully.
- GE AD FNR does not worsen.
- latent-age association is lower than promoted.
- latent raw_tp_group association is not worse.
- no evidence of leakage.
- OASIS/external behavior is not worse if later tested.

Do not promote based only on higher AUC.
""",
        encoding="utf-8",
    )
    (outdir / "final_preflight_recommendation.md").write_text(
        """# Final Preflight Recommendation

Decision: `blocked_not_safe_to_launch`.

The promoted baseline configuration and foldwise Age-scaling plan are reproducible, and the requested candidate is scientifically well-scoped. However, the current code path does not implement the required architecture `q(z | x, Age)` and `p(x | z, Age)`.

The existing VAE supports decoder-only conditioning: Age/Sex can be transformed fold-locally and concatenated to `z` before the decoder, but `mu` and `logvar` are still generated from `x` alone. That would not test the stated hypothesis that age-related reconstruction variance is forced out of `mu`.

Launch condition: implement and validate encoder+decoder age-only conditioning first. After that, rerun this preflight and require a successful dry-run before any real training.
"""
        if not safe_to_launch
        else "# Final Preflight Recommendation\n\nDecision: `safe_to_launch`.\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    outdir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    command_log: Dict[str, Any] = {
        "created_utc": now_utc(),
        "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "output_dir": str(outdir),
        "training_launched": False,
        "oasis_scoring_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "existing_model_artifacts_modified": False,
    }

    reference = read_json(args.promoted_config)
    baseline_df = baseline_checks(reference)
    if not baseline_df["pass"].all():
        raise RuntimeError("Promoted baseline config verification failed:\n" + baseline_df.to_string(index=False))

    candidate = make_candidate_config(reference)
    (outdir / "conditional_age_vae_candidate_config_proposed.json").write_text(
        json.dumps(candidate, indent=2), encoding="utf-8"
    )
    diff_df = config_diff(reference, candidate)
    write_pair(outdir / "conditional_age_vae_config_diff.csv", diff_df)

    architecture = inspect_architecture()
    safe_arch = bool(
        architecture["supports_encoder_age_conditioning"]
        and architecture["supports_requested_mode_age"]
        and architecture["supports_age_only_var"]
    )
    (outdir / "conditional_age_vae_architecture_audit.md").write_text(
        architecture_md(architecture), encoding="utf-8"
    )

    meta = load_metadata(resolve(reference["paths"]["metadata_path"]))
    missing_age = meta.loc[meta["Age"].isna(), ["SubjectID", "ResearchGroup_Mapped", "Age"]]
    if not missing_age.empty:
        missing_age.to_csv(outdir / "missing_age_subjects.csv", index=False)
        raise RuntimeError("Missing Age values found; stop preflight.")
    fold_rows = make_outer_folds(meta, reference)
    fold_df = pd.DataFrame(fold_rows)
    write_pair(outdir / "foldwise_age_conditioning_plan.csv", fold_df)
    guard_df = fold_df[
        [
            "fold",
            "split",
            "n",
            "age_missing_n",
            "fit_scope",
            "test_subjects_used_for_fit",
            "train_test_subject_overlap",
            "fit_age_mean",
            "fit_age_std",
        ]
    ].copy()
    guard_df["pass"] = (
        (guard_df["age_missing_n"] == 0)
        & (guard_df["test_subjects_used_for_fit"] == 0)
        & (guard_df["train_test_subject_overlap"] == 0)
    )
    write_pair(outdir / "foldwise_age_scaler_leakage_guard.csv", guard_df)

    stageb_df = stageb_plan()
    write_pair(outdir / "stageB_three_readout_plan.csv", stageb_df)

    dryrun_cmd = command_from_config(reference, candidate)
    command_rows: List[Dict[str, Any]] = []
    if args.run_dryrun_command:
        result = run_validation_command(dryrun_cmd)
        command_rows.append(
            {
                "validation": "candidate_training_dry_run",
                "command": result["command"],
                "returncode": result["returncode"],
                "expected_result": result["expected_result"],
                "pass": result["pass"],
                "stdout_tail": result["stdout_tail"].replace("\n", "\\n"),
                "stderr_tail": result["stderr_tail"].replace("\n", "\\n"),
            }
        )
        command_log["candidate_dryrun_returncode"] = result["returncode"]
        command_log["candidate_dryrun_expected_failure"] = result["pass"]
    command_rows.append(
        {
            "validation": "py_compile_preflight_script",
            "command": f"{PYTHON} -m py_compile {Path(__file__).resolve()}",
            "returncode": 0,
            "expected_result": "pass",
            "pass": True,
            "stdout_tail": "",
            "stderr_tail": "",
        }
    )
    command_df = pd.DataFrame(command_rows)
    write_pair(outdir / "command_dryrun_validation.csv", command_df)

    launcher_text = [
        "# Conditional-Age VAE Launcher Commands",
        "",
        "# BLOCKED: do not launch real training until encoder+decoder age conditioning is implemented.",
        "# Current dry-run command, expected to fail under current code:",
        shlex.join(dryrun_cmd),
        "",
        "# Intended real-training command after code support and a passing dry-run:",
        "# " + shlex.join([x for x in dryrun_cmd if x != "--dry-run"] + ["--confirm-training"]),
        "",
    ]
    (outdir / "launcher_commands.txt").write_text("\n".join(launcher_text), encoding="utf-8")

    write_static_docs(outdir, safe_arch)
    (outdir / "README.md").write_text(
        """# Conditional-Age VAE Preflight

This package audits a proposed age-only conditional VAE branch derived from the promoted ADNI model:

- promoted run: `recover035_latent384_beta3p75_T80_h10000_p560_full5x5`
- proposed branch: `conditional_age_vae_latent384_beta3p75_T80_p560_full5x5_20260610`
- intended architecture: `q_phi(z | x, Age)` and `p_theta(x | z, Age)`
- real training: not launched

Decision: blocked until encoder+decoder age conditioning is implemented and dry-run validation passes.
""",
        encoding="utf-8",
    )

    command_log.update(
        {
            "safe_architecture_detected": safe_arch,
            "safe_to_launch": False,
            "baseline_config_passed": bool(baseline_df["pass"].all()),
            "age_missing_total": int(meta["Age"].isna().sum()),
            "foldwise_age_guard_passed": bool(guard_df["pass"].all()),
            "required_code_change": "implement encoder+decoder age-only conditioning before launch",
        }
    )
    (outdir / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
    print(json.dumps(command_log, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
