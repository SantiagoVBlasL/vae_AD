#!/usr/bin/env python3
"""
Smoketest for SHAP + IG interpretability on the final promoted model.

Scope: fold 1 only, SHAP on all 80 test subjects, IG on 2 subjects × 8 steps.
Validates shapes, finiteness, and artifact alignment before committing to full run.

Read-only with respect to model artifacts, tensors, and metadata.
All outputs go to /media/diego/Datos/.../smoketest/.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("smoketest_shap_ig")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ── Paths ─────────────────────────────────────────────────────────────────────
PROMOTED_RUN = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
)
TENSOR_PATH = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026"
    "/adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors"
    "/GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)
METADATA_PATH = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/adni_035_metadata_rescue_preflight/patched_metadata_candidate.csv"
)
ROI_ANNOTATION_PATH = PROMOTED_RUN / "roi_info_from_tensor.csv"
INTERP_SCRIPT = PROJECT_ROOT / "scripts/run_interpretability.py"

HEAVY_OUT = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/final_model_shap_ig_20260624/smoketest"
)

# ── Model params ───────────────────────────────────────────────────────────────
CHANNELS = [1, 0, 2]
LATENT_DIM = 384
DROPOUT = 0.15
NUM_CONV = 4
DECODER_TYPE = "convtranspose"
INTER_FC = "quarter"
FINAL_ACT = "tanh"
META_FEATS = ["Age", "Sex"]
SEED = 42

FOLD = 1
SMOKETEST_IG_N = 2
SMOKETEST_IG_STEPS = 8

PYTHON = sys.executable


def _run(cmd: list[str], tag: str) -> None:
    log.info(f"[{tag}] Running: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.time() - t0
    log.info(f"[{tag}] Done in {elapsed:.1f}s (returncode={result.returncode})")
    if result.returncode != 0:
        raise RuntimeError(f"[{tag}] FAILED with returncode={result.returncode}")


def common_args() -> list[str]:
    return [
        "--run_dir", str(PROMOTED_RUN),
        "--fold", str(FOLD),
        "--clf", "logreg",
        "--global_tensor_path", str(TENSOR_PATH),
        "--metadata_path", str(METADATA_PATH),
        "--channels_to_use", *[str(c) for c in CHANNELS],
        "--latent_dim", str(LATENT_DIM),
        "--latent_features_type", "mu",
        "--metadata_features", *META_FEATS,
        "--num_conv_layers_encoder", str(NUM_CONV),
        "--decoder_type", DECODER_TYPE,
        "--dropout_rate_vae", str(DROPOUT),
        "--intermediate_fc_dim_vae", INTER_FC,
        "--vae_final_activation", FINAL_ACT,
        "--seed", str(SEED),
    ]


def run_shap_smoketest(out_dir: Path) -> None:
    fold_out = out_dir / f"fold_{FOLD}"
    fold_out.mkdir(parents=True, exist_ok=True)
    cmd = [PYTHON, str(INTERP_SCRIPT), "shap"] + common_args() + [
        "--bg_mode", "train",
        "--bg_sample_size", "50",
        "--bg_seed", str(SEED),
        "--freeze_meta", "Age", "Sex",
        "--freeze_strategy", "train_stats",
        "--shap_link", "identity",
        "--shap_tag", "smoketest",
    ]
    _run(cmd, "SHAP_smoketest")


def run_ig_smoketest(out_dir: Path) -> None:
    fold_out = out_dir / f"fold_{FOLD}"
    fold_out.mkdir(parents=True, exist_ok=True)
    cmd = [PYTHON, str(INTERP_SCRIPT), "saliency"] + common_args() + [
        "--roi_annotation_path", str(ROI_ANNOTATION_PATH),
        "--saliency_method", "integrated_gradients",
        "--ig_n_steps", str(SMOKETEST_IG_STEPS),
        "--ig_baseline", "cn_median_train",
        "--top_k", "20",
        "--shap_weight_mode", "ad_vs_cn_diff",
        "--shap_tag", "smoketest",
    ]
    _run(cmd, "IG_smoketest")


def verify_outputs(out_dir: Path) -> dict:
    import joblib as _joblib
    fold_dir = PROMOTED_RUN / f"fold_{FOLD}"
    results: dict = {"pass": True, "checks": []}

    def _check(name: str, ok: bool, detail: str = "") -> None:
        status = "PASS" if ok else "FAIL"
        results["checks"].append({"check": name, "status": status, "detail": detail})
        if not ok:
            results["pass"] = False
            log.error(f"  CHECK FAIL: {name} — {detail}")
        else:
            log.info(f"  CHECK PASS: {name}")

    # VAE checkpoint
    ckpt = PROMOTED_RUN / f"fold_{FOLD}/vae_model_fold_{FOLD}.pt"
    _check("vae_checkpoint_exists", ckpt.exists(), str(ckpt))

    # logreg pipeline
    raw_pipe = PROMOTED_RUN / f"fold_{FOLD}/classifier_logreg_raw_pipeline_fold_{FOLD}.joblib"
    _check("logreg_raw_pipeline_exists", raw_pipe.exists(), str(raw_pipe))

    # SHAP output: interpret_fold.py saves a joblib pack in interpretability_shap/
    shap_pack_path = fold_dir / "interpretability_shap" / "shap_pack_logreg_smoketest.joblib"
    _check("shap_output_found", shap_pack_path.exists(), str(shap_pack_path))
    if shap_pack_path.exists():
        pack = _joblib.load(shap_pack_path)
        sv = pack.get("shap_values")
        if sv is not None:
            arr = np.array(sv)
            _check("shap_shape_cols_eq_386", arr.ndim == 2 and arr.shape[1] == 386,
                   f"shape={arr.shape}")
            _check("shap_finite", bool(np.isfinite(arr).all()),
                   f"nan={np.isnan(arr).sum()}")
        else:
            _check("shap_values_key_present", False, "key 'shap_values' not in pack")

    # IG output: saliency_map_diff_signed_integrated_gradients_top20.npy in interpretability_logreg/
    # Shape is (C, R, R) = (3, 131, 131) — interpret_fold.py averages over subjects internally.
    ig_path = (fold_dir / "interpretability_logreg"
               / "saliency_map_diff_signed_integrated_gradients_top20.npy")
    _check("ig_output_found", ig_path.exists(), str(ig_path))
    if ig_path.exists():
        ig = np.load(ig_path)
        _check("ig_shape_channels_3", ig.ndim == 3 and ig.shape[0] == 3,
               f"shape={ig.shape}")
        _check("ig_shape_roi_131", ig.shape[-1] == 131, f"shape={ig.shape}")
        _check("ig_finite", bool(np.isfinite(ig).all()), f"nan={np.isnan(ig).sum()}")
        _check("ig_not_all_zero", not np.allclose(ig, 0.0),
               "all-zero map (gradient vanishing through tanh?)")

    # IG baseline artifact: cn_median_train.npy should be saved alongside the map
    baseline_path = fold_dir / "interpretability_logreg" / "ig_baseline_cn_median_train.npy"
    _check("ig_baseline_file_saved", baseline_path.exists(), str(baseline_path))
    if baseline_path.exists():
        bl = np.load(baseline_path)
        _check("ig_baseline_shape", bl.shape == (3, 131, 131), f"shape={bl.shape}")
        _check("ig_baseline_not_all_zero", not np.allclose(bl, 0.0),
               "baseline is all-zero (cn_median_train should differ from zeros)")

    return results


def main() -> None:
    t_start = time.time()
    log.info("=" * 60)
    log.info("SHAP/IG Smoketest — fold 1, 2 subjects, 8 IG steps")
    log.info("=" * 60)

    # Pre-flight guards
    if not TENSOR_PATH.exists():
        raise FileNotFoundError(f"Tensor not found: {TENSOR_PATH}")
    if not METADATA_PATH.exists():
        raise FileNotFoundError(f"Metadata not found: {METADATA_PATH}")
    if not INTERP_SCRIPT.exists():
        raise FileNotFoundError(f"Interpretability script not found: {INTERP_SCRIPT}")
    if not ROI_ANNOTATION_PATH.exists():
        raise FileNotFoundError(f"ROI annotation not found: {ROI_ANNOTATION_PATH}")

    HEAVY_OUT.mkdir(parents=True, exist_ok=True)

    run_shap_smoketest(HEAVY_OUT)
    run_ig_smoketest(HEAVY_OUT)

    log.info("Verifying outputs...")
    verdict = verify_outputs(HEAVY_OUT)

    elapsed = time.time() - t_start
    verdict["elapsed_seconds"] = round(elapsed, 1)
    verdict["fold"] = FOLD
    verdict["ig_subjects"] = SMOKETEST_IG_N
    verdict["ig_steps"] = SMOKETEST_IG_STEPS
    verdict["generated_utc"] = datetime.now(timezone.utc).isoformat()

    out_json = HEAVY_OUT / "smoketest_verification.json"
    out_json.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    log.info(f"Verification written: {out_json}")

    if verdict["pass"]:
        log.info(f"SMOKETEST PASSED in {elapsed:.1f}s — safe to proceed to full run.")
    else:
        log.error(f"SMOKETEST FAILED in {elapsed:.1f}s — DO NOT launch full run.")
        sys.exit(1)


if __name__ == "__main__":
    main()
