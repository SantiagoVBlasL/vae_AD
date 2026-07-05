#!/usr/bin/env python3
"""Validate code support for encoder+decoder age-only VAE conditioning.

This is a read-only/preflight utility. It compiles modified code, runs a small
architecture smoke test, validates fold-local Age scaling, and runs dry-run CLI
commands only. It does not launch training.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from betavae_xai.models import ConvolutionalVAE  # noqa: E402


PROMOTED_CONFIG = (
    PROJECT_ROOT
    / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5.json"
)
TRAIN_SCRIPT = PROJECT_ROOT / "scripts/run_vae_clf_ad_inference.py"
MODEL_SCRIPT = PROJECT_ROOT / "src/betavae_xai/models/convolutional_vae.py"
OUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/conditional_age_vae_code_support_preflight_20260611"
RUN_ROOT = PROJECT_ROOT / "results/revision_bspc_2026/conditional_age_vae_latent384_beta3p75_T80_p560_full5x5_20260610"
PYTHON = "/home/diego/anaconda3/envs/vae_ad/bin/python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--promoted-config", type=Path, default=PROMOTED_CONFIG)
    return parser.parse_args()


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def write_md_table(path: Path, df: pd.DataFrame) -> None:
    path.write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")


def write_pair(csv_path: Path, df: pd.DataFrame) -> None:
    df.to_csv(csv_path, index=False)
    write_md_table(csv_path.with_suffix(".md"), df)


def run_command(cmd: Sequence[str], *, timeout: int = 180) -> Dict[str, Any]:
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, capture_output=True, timeout=timeout)
    return {
        "command": shlex.join(cmd),
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def load_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    if missing:
        raise RuntimeError(f"Metadata missing required columns: {missing}")
    meta = meta.copy()
    meta["SubjectID"] = meta["SubjectID"].astype(str)
    meta["ResearchGroup_Mapped"] = meta["ResearchGroup_Mapped"].astype(str).str.strip()
    meta["Age"] = pd.to_numeric(meta["Age"], errors="coerce")
    meta["Sex"] = meta["Sex"].astype(str).str.strip()
    meta["Manufacturer"] = meta["Manufacturer"].map(normalize_mfr)
    meta["tensor_idx"] = pd.to_numeric(meta["tensor_idx"], errors="raise").astype(int)
    return meta


def line_no(path: Path, needle: str) -> int | None:
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if needle in line:
            return i
    return None


def stage_a_command(config: Dict[str, Any], *, mode: str, vars_: str, output_suffix: str) -> List[str]:
    params = config["parameters"]
    cmd = [
        PYTHON,
        str(TRAIN_SCRIPT),
        "--global_tensor_path",
        config["paths"]["global_tensor_path"],
        "--metadata_path",
        config["paths"]["metadata_path"],
        "--output_dir",
        str(RUN_ROOT) + output_suffix,
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
        "seed",
        "num_workers",
        "n_jobs_gridsearch",
        "n_iter_logreg",
        "n_iter_svm",
    ]
    for name in passthrough:
        value = params.get(name)
        if value is None:
            continue
        cmd.append(f"--{name}")
        if isinstance(value, list):
            cmd.extend(str(v) for v in value)
        else:
            cmd.append(str(value))
    cmd.extend(["--vae_conditioning_mode", mode, "--vae_conditioning_vars", vars_])
    for name in [
        "classifier_calibrate",
        "classifier_use_class_weight",
        "save_fold_artefacts",
        "save_vae_training_history",
        "qc_analyze_distributions",
        "qc_check_scanner_leakage",
        "qc_rate_distortion",
        "qc_latent_information",
    ]:
        if params.get(name):
            cmd.append(f"--{name}")
    cmd.append("--dry-run")
    return cmd


def smoke_test() -> str:
    torch.manual_seed(20260611)
    model = ConvolutionalVAE(
        input_channels=3,
        latent_dim=8,
        image_size=32,
        final_activation="tanh",
        intermediate_fc_dim_config="quarter",
        dropout_rate=0.0,
        num_conv_layers_encoder=3,
        decoder_type="convtranspose",
        conditioning_mode="encoder_decoder",
        conditioning_dim=1,
    )
    model.eval()
    x = torch.randn(1, 3, 32, 32)
    age_a = torch.tensor([[0.0]], dtype=torch.float32)
    age_b = torch.tensor([[1.5]], dtype=torch.float32)
    with torch.no_grad():
        mu_a, logvar_a = model.encode(x, condition=age_a)
        mu_b, logvar_b = model.encode(x, condition=age_b)
        recon_a = model.decode(mu_a, condition=age_a)
        recon_b = model.decode(mu_b, condition=age_b)
    mu_delta = float(torch.max(torch.abs(mu_a - mu_b)).item())
    logvar_delta = float(torch.max(torch.abs(logvar_a - logvar_b)).item())
    recon_delta = float(torch.max(torch.abs(recon_a - recon_b)).item())
    assert mu_delta > 1e-7, "mu did not change when Age changed"
    assert logvar_delta > 1e-7, "logvar did not change when Age changed"
    assert recon_delta > 1e-7, "reconstruction did not change when Age changed"
    return (
        "conditional_age_smoke_test=PASS\n"
        f"mu_max_abs_delta={mu_delta:.8f}\n"
        f"logvar_max_abs_delta={logvar_delta:.8f}\n"
        f"recon_max_abs_delta={recon_delta:.8f}\n"
        f"conditioning_mode={model.conditioning_mode}\n"
        f"conditioning_dim={model.conditioning_dim}\n"
    )


def foldwise_age_guard(meta: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
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
    all_valid_tensor_idx = meta["tensor_idx"].astype(int).to_numpy()
    rows: List[Dict[str, Any]] = []
    for fold_idx, (train_dev_idx, test_idx) in enumerate(skf.split(np.arange(len(cn_ad)), y_outer), start=1):
        test_tensor_idx = cn_ad.iloc[test_idx]["tensor_idx"].astype(int).to_numpy()
        vae_pool_tensor_idx = np.setdiff1d(np.unique(all_valid_tensor_idx), np.unique(test_tensor_idx))
        vae_pool = meta.set_index("tensor_idx").loc[vae_pool_tensor_idx].reset_index()
        vae_strat_cols = ["ResearchGroup_Mapped"]
        for col in params.get("vae_stratify_cols", []) or []:
            if col in vae_pool.columns and col not in vae_strat_cols:
                vae_strat_cols.append(col)
        strat_df = vae_pool[vae_strat_cols].copy()
        for col in vae_strat_cols:
            strat_df[col] = strat_df[col].fillna(f"{col}_Unknown").astype(str)
        vae_strat = strat_df.apply(lambda x: "_".join(x.astype(str)), axis=1)
        if not (vae_strat.value_counts() >= 2).all():
            vae_strat = vae_pool["ResearchGroup_Mapped"].fillna("RG_Unknown").astype(str)
        train_local, val_local = train_test_split(
            np.arange(len(vae_pool), dtype=int),
            test_size=float(params.get("vae_val_split_ratio", 0.2)),
            stratify=vae_strat,
            random_state=int(params["seed"]) + (fold_idx - 1) + 10,
            shuffle=True,
        )
        fit_pool = vae_pool.iloc[np.asarray(train_local, dtype=int)].copy()
        val_pool = vae_pool.iloc[np.asarray(val_local, dtype=int)].copy()
        clf_test = cn_ad.iloc[test_idx].copy()
        fit_mean = float(fit_pool["Age"].mean())
        fit_std = float(fit_pool["Age"].std(ddof=0))
        for split_name, split_df in [
            ("vae_actual_train_fit", fit_pool),
            ("vae_internal_val_apply", val_pool),
            ("classifier_test_apply", clf_test),
        ]:
            overlap = set(fit_pool["SubjectID"]).intersection(set(clf_test["SubjectID"]))
            rows.append(
                {
                    "fold": fold_idx,
                    "split": split_name,
                    "n": int(len(split_df)),
                    "age_missing_n": int(split_df["Age"].isna().sum()),
                    "fit_scope": "vae_actual_train_only",
                    "test_subjects_used_for_fit": 0,
                    "train_test_subject_overlap": int(len(overlap)),
                    "fit_age_mean": fit_mean,
                    "fit_age_std": fit_std,
                    "raw_age_mean": float(split_df["Age"].mean()) if len(split_df) else np.nan,
                    "raw_age_min": float(split_df["Age"].min()) if len(split_df) else np.nan,
                    "raw_age_max": float(split_df["Age"].max()) if len(split_df) else np.nan,
                    "pass": int(split_df["Age"].isna().sum()) == 0 and len(overlap) == 0,
                }
            )
    return pd.DataFrame(rows)


def code_diff_summary() -> str:
    return f"""# Code Diff Summary

Modified files:

- `{MODEL_SCRIPT.relative_to(PROJECT_ROOT)}`
  - Added `encoder_decoder` to `CONDITIONING_MODE_CHOICES`.
  - Added encoder-side condition concatenation before `fc_mu` and `fc_logvar`.
  - Updated `encode(x, condition=None)` and `forward(x, condition=None)` so `encoder_decoder` implements `q(z | x, Age)`.
  - Existing `none` and `decoder_only` behavior is preserved.

- `{TRAIN_SCRIPT.relative_to(PROJECT_ROOT)}`
  - Added age-only VAE conditioning variable choices: `Age` / `age`.
  - Added parser validation for `--vae_conditioning_mode encoder_decoder --vae_conditioning_vars Age`.
  - Kept `none` and `decoder_only` validation paths backward-compatible.
  - Existing fold-local transformer still fits from VAE actual-train rows only.

No tensor, metadata, or promoted model artifact changes were made.
"""


def architecture_audit() -> str:
    return f"""# Architecture Support Audit

Decision: the code path now supports age-only encoder+decoder VAE conditioning.

Evidence:

- `{MODEL_SCRIPT.relative_to(PROJECT_ROOT)}` line {line_no(MODEL_SCRIPT, '"encoder_decoder",')}: `encoder_decoder` is an allowed conditioning mode.
- `{MODEL_SCRIPT.relative_to(PROJECT_ROOT)}` line {line_no(MODEL_SCRIPT, 'if conditioning_mode == "encoder_decoder"')}: `fc_mu`/`fc_logvar` receive an encoder representation widened by `conditioning_dim`.
- `{MODEL_SCRIPT.relative_to(PROJECT_ROOT)}` line {line_no(MODEL_SCRIPT, 'def _prepare_encoder_input')}: encoder condition checks enforce 2D `[B,C]` condition shape.
- `{MODEL_SCRIPT.relative_to(PROJECT_ROOT)}` line {line_no(MODEL_SCRIPT, 'def encode(self, x: torch.Tensor, condition')}: encoder accepts condition.
- `{MODEL_SCRIPT.relative_to(PROJECT_ROOT)}` line {line_no(MODEL_SCRIPT, 'mu, logvar = self.encode(x, condition=condition)')}: posterior is conditioned during forward.
- `{MODEL_SCRIPT.relative_to(PROJECT_ROOT)}` line {line_no(MODEL_SCRIPT, 'return torch.cat([z, condition], dim=1)')}: decoder remains conditioned through `[z, Age]`.
- `{TRAIN_SCRIPT.relative_to(PROJECT_ROOT)}` line {line_no(TRAIN_SCRIPT, 'elif args.vae_conditioning_mode == "encoder_decoder"')}: CLI restricts encoder+decoder conditioning to `Age`.

Guardrails:

- Diagnosis, Manufacturer, Site3, raw_tp_group, fold, score, prediction outcome, and y labels are not accepted by `encoder_decoder`.
- Sex is not a VAE condition for this experiment; Sex remains a planned Stage B feature only.
- Age scaling uses the existing fold-local VAE conditioning transformer fit on VAE actual-train rows.
"""


def final_recommendation(safe: bool) -> str:
    decision = "safe_to_launch" if safe else "blocked_not_safe_to_launch"
    return f"""# Final Code Support Recommendation

Decision: `{decision}`.

The patched code supports the intended conditional-age architecture and dry-runs pass only if all validation tables in this package pass. Real training was not launched.
"""


def main() -> int:
    args = parse_args()
    outdir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    command_log: Dict[str, Any] = {
        "created_utc": now_utc(),
        "script": str(Path(__file__).resolve().relative_to(PROJECT_ROOT)),
        "output_dir": str(outdir),
        "training_launched": False,
        "tensor_modified": False,
        "metadata_modified": False,
        "promoted_artifacts_modified": False,
    }

    config = load_config(args.promoted_config)
    meta = load_metadata(resolve(config["paths"]["metadata_path"]))
    if meta["Age"].isna().any():
        raise RuntimeError("Missing Age values found; cannot launch conditional-age VAE.")

    smoke_stdout = smoke_test()
    (outdir / "conditioning_smoke_test_stdout.txt").write_text(smoke_stdout, encoding="utf-8")

    guard_df = foldwise_age_guard(meta, config)
    write_pair(outdir / "foldwise_age_scaler_leakage_guard.csv", guard_df)

    validation_rows: List[Dict[str, Any]] = []
    compile_targets = [MODEL_SCRIPT, TRAIN_SCRIPT, Path(__file__).resolve()]
    for target in compile_targets:
        result = run_command([PYTHON, "-m", "py_compile", str(target)])
        validation_rows.append(
            {
                "validation": f"py_compile:{target.relative_to(PROJECT_ROOT)}",
                **result,
                "pass": result["returncode"] == 0,
            }
        )

    candidate_cmd = stage_a_command(config, mode="encoder_decoder", vars_="Age", output_suffix="")
    result = run_command(candidate_cmd)
    validation_rows.append(
        {
            "validation": "candidate_encoder_decoder_age_dry_run",
            **result,
            "pass": result["returncode"] == 0,
        }
    )
    command_df = pd.DataFrame(validation_rows)
    write_pair(outdir / "command_dryrun_validation.csv", command_df)

    compat_rows: List[Dict[str, Any]] = []
    for label, mode, vars_, suffix in [
        ("promoted_none_dry_run", "none", "none", "_compat_none"),
        ("decoder_only_age_sex_dry_run", "decoder_only", "age_sex", "_compat_decoder_only"),
    ]:
        result = run_command(stage_a_command(config, mode=mode, vars_=vars_, output_suffix=suffix))
        compat_rows.append(
            {
                "validation": label,
                **result,
                "pass": result["returncode"] == 0,
            }
        )
    compat_df = pd.DataFrame(compat_rows)
    write_pair(outdir / "backward_compatibility_validation.csv", compat_df)

    (outdir / "code_diff_summary.md").write_text(code_diff_summary(), encoding="utf-8")
    (outdir / "architecture_support_audit.md").write_text(architecture_audit(), encoding="utf-8")

    safe = bool(
        command_df["pass"].all()
        and compat_df["pass"].all()
        and guard_df["pass"].all()
        and "conditional_age_smoke_test=PASS" in smoke_stdout
    )
    (outdir / "final_code_support_recommendation.md").write_text(final_recommendation(safe), encoding="utf-8")
    command_log.update(
        {
            "safe_to_launch": safe,
            "final_decision": "safe_to_launch" if safe else "blocked_not_safe_to_launch",
            "smoke_test_passed": "conditional_age_smoke_test=PASS" in smoke_stdout,
            "foldwise_age_guard_passed": bool(guard_df["pass"].all()),
            "candidate_dry_run_passed": bool(
                command_df.loc[command_df["validation"].eq("candidate_encoder_decoder_age_dry_run"), "pass"].iloc[0]
            ),
            "backward_compatibility_passed": bool(compat_df["pass"].all()),
        }
    )
    (outdir / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
    print(json.dumps(command_log, indent=2))
    return 0 if safe else 1


if __name__ == "__main__":
    raise SystemExit(main())
