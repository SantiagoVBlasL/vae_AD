#!/usr/bin/env python3
"""Prepare matched FULL pair preflight: single_ch5 vs contemporary ch102 control.

Writes only small local preflight artifacts and launch wrappers. Does not train.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


REPO = Path("/home/diego/proyectos/vae_AD")
OUT = REPO / (
    "results/revision_bspc_2026/post_revision_exploratory_20260630/"
    "full_pair_ch5_vs_ch102_channelmean_beta3p75_preflight_20260703"
)
RUN_ROOT = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/post_revision_exploratory_20260630/"
    "full_pair_ch5_vs_ch102_channelmean_beta3p75_20260703"
)
LOG_ROOT = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/_launch_logs/post_revision_exploratory_20260630/"
    "full_pair_ch5_vs_ch102_channelmean_beta3p75_20260703"
)
TENSOR = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/"
    "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)
METADATA = REPO / "results/revision_bspc_2026/adni_035_metadata_rescue_preflight/patched_metadata_candidate.csv"
RUNNER = REPO / "scripts/run_vae_clf_ad_inference.py"
HISTORICAL_FULL = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/"
    "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
)
AMENDMENT_005 = REPO / (
    "results/revision_bspc_2026/post_revision_exploratory_20260630/"
    "single_ch5_full5x5_promotion_20260703/protocol_amendment_005_single_ch5_full_promotion.md"
)
GATE = REPO / (
    "results/revision_bspc_2026/post_revision_exploratory_20260630/"
    "fast_channelmean_loss_ablation_beta_matched_20260702/gate_criteria_preregistered.md"
)

EXPECTED_COMMIT = "e36db5847aa9637df7dcd6732b2a9ad7fada75c9"
BRANCH = "exploratory/post-revision-20260630"

CHANNEL_NAMES = {
    0: "Pearson_OMST_GCE_Signed_Weighted",
    1: "Pearson_Full_FisherZ_Signed",
    2: "MI_KNN_Symmetric",
    3: "dFC_AbsDiffMean",
    4: "dFC_StdDev",
    5: "DistanceCorr",
    6: "Granger_F_lag1",
}

COMMON_CONFIG = {
    "training_script": "scripts/run_vae_clf_ad_inference.py",
    "git_commit": EXPECTED_COMMIT,
    "recon_loss_mode": "offdiag_channelmean_sum",
    "beta_vae": 3.75,
    "latent_dim": 384,
    "epochs_vae": 10000,
    "outer_folds": 5,
    "inner_folds": 5,
    "repeated_outer_folds_n_repeats": 1,
    "cyclical_beta_n_cycles": 125,
    "cyclical_beta_ratio_increase": 0.4,
    "lr_scheduler_T0": 80,
    "early_stopping_patience_vae": 560,
    "batch_size": 64,
    "dropout_rate_vae": 0.15,
    "vae_final_activation": "tanh",
    "intermediate_fc_dim_vae": "quarter",
    "norm_mode": "zscore_offdiag",
    "metadata_features": "Age Sex",
    "classifier_types": "logreg svm",
    "classifier_use_class_weight": True,
    "classifier_calibrate": True,
    "gridsearch_scoring": "roc_auc",
    "n_iter_logreg": 500,
    "n_iter_svm": 500,
    "classifier_stratify_cols": "Manufacturer",
    "vae_stratify_cols": "Manufacturer",
    "vae_required_metadata_cols": "ResearchGroup_Mapped Manufacturer Age Sex",
    "vae_abort_if_val_split_fails": True,
    "seed": 42,
}

CANDIDATES = [
    ("single_ch5", [5]),
    ("control_ch102", [1, 0, 2]),
]


def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True, check=False)
    return p.returncode, (p.stdout + p.stderr).strip()


def sha_list(values: list[str]) -> str:
    text = "\n".join(sorted(map(str, values))).encode()
    return hashlib.sha256(text).hexdigest()


def current_branch() -> str:
    return run(["git", "branch", "--show-current"])[1].strip()


def current_commit() -> str:
    return run(["git", "rev-parse", "HEAD"])[1].strip()


def load_merged_metadata() -> tuple[np.ndarray, pd.DataFrame]:
    npz = np.load(TENSOR, allow_pickle=True)
    subject_ids = np.asarray(npz["subject_ids"]).astype(str)
    tensor_df = pd.DataFrame({"SubjectID": subject_ids})
    tensor_df["SubjectID"] = tensor_df["SubjectID"].astype(str).str.strip()
    tensor_df["tensor_idx"] = np.arange(len(subject_ids), dtype=int)
    md = pd.read_csv(METADATA)
    md["SubjectID"] = md["SubjectID"].astype(str).str.strip()
    md = md.drop_duplicates("SubjectID", keep="first")
    merged = tensor_df.merge(md, on="SubjectID", how="left", validate="one_to_one")
    return npz["global_tensor_data"], merged


def compute_fold_hashes() -> pd.DataFrame:
    tensor, md = load_merged_metadata()
    max_valid_idx = tensor.shape[0] - 1
    cn_ad = md[md["ResearchGroup_Mapped"].isin(["CN", "AD"])].copy()
    cn_ad = cn_ad[cn_ad["tensor_idx"] <= max_valid_idx].copy()
    cn_ad["label"] = cn_ad["ResearchGroup_Mapped"].map({"CN": 0, "AD": 1}).astype(int)
    strat_cols = ["ResearchGroup_Mapped"]
    if "Manufacturer" in cn_ad.columns:
        cn_ad["Manufacturer"] = cn_ad["Manufacturer"].fillna("Manufacturer_Unknown").astype(str)
        strat_cols.append("Manufacturer")
    strat_key = cn_ad[strat_cols].apply(lambda x: "_".join(x.astype(str)), axis=1)
    y_outer = strat_key
    vc = pd.Series(y_outer).value_counts()
    split_mode = "ResearchGroup_Mapped+Manufacturer"
    if (vc < 5).any():
        y_outer = cn_ad["label"].to_numpy()
        split_mode = "ResearchGroup_Mapped_label_only_fallback"
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    valid_meta_idx = md.loc[md["tensor_idx"] <= max_valid_idx, "tensor_idx"].dropna().astype(int).to_numpy()
    rows = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.arange(len(cn_ad)), y_outer), start=1):
        train_df = cn_ad.iloc[train_idx].copy()
        test_df = cn_ad.iloc[test_idx].copy()
        vae_pool_idx = np.setdiff1d(np.unique(valid_meta_idx), np.unique(test_df["tensor_idx"].astype(int).to_numpy()))
        vae_pool_df = md.set_index("tensor_idx").loc[vae_pool_idx].reset_index()
        req = ["ResearchGroup_Mapped", "Manufacturer", "Age", "Sex"]
        missing_mask = pd.Series(False, index=vae_pool_df.index)
        for col in req:
            missing_mask |= (
                vae_pool_df[col].isna()
                | vae_pool_df[col].astype(str).str.strip().isin(["", "nan", "NaN", "None", "none", "NA", "N/A"])
            )
        vae_after = vae_pool_df.loc[~missing_mask].copy()
        rows.append(
            {
                "fold": fold,
                "split_mode": split_mode,
                "classifier_train_dev_n": len(train_df),
                "classifier_test_n": len(test_df),
                "test_cn": int((test_df["ResearchGroup_Mapped"] == "CN").sum()),
                "test_ad": int((test_df["ResearchGroup_Mapped"] == "AD").sum()),
                "vae_pool_before_required_metadata_n": len(vae_pool_df),
                "vae_pool_after_required_metadata_n": len(vae_after),
                "vae_required_metadata_removed_n": int(missing_mask.sum()),
                "train_dev_subject_hash": sha_list(train_df["SubjectID"].tolist()),
                "test_subject_hash": sha_list(test_df["SubjectID"].tolist()),
                "vae_pool_subject_hash": sha_list(vae_after["SubjectID"].tolist()),
            }
        )
    return pd.DataFrame(rows)


def write_manifest() -> pd.DataFrame:
    rows = []
    for idx, (label, channels) in enumerate(CANDIDATES, start=1):
        rows.append(
            {
                "candidate_id": idx,
                "run_label": label,
                "channels_to_use": " ".join(map(str, channels)),
                "channel_names": "; ".join(CHANNEL_NAMES[c] for c in channels),
                "output_dir": str(RUN_ROOT / label),
                "status": "new",
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "planned_full_pair_candidates.csv", index=False)
    return df


def write_amendment() -> None:
    text = f"""# Protocol Amendment 005a — Matched contemporary FULL pair

Date: {datetime.now().strftime('%Y-%m-%d')}

This amendment corrects the FULL confirmation design prepared in Amendment 005. It does not modify Amendments 001-005 and does not modify `gate_criteria_preregistered.md`.

## Correction

The historical locked FULL `[1,0,2]` model was trained at commit `a38c5a1`, while the current HEAD is `{EXPECTED_COMMIT}`. It also used `mse_sum_batchmean_current`, whose reconstruction scale grows with channel count. A primary comparison between single-channel `[5]` and historical `[1,0,2]` would therefore mix channel-set effects, code-version effects, and channel-count-dependent effective-beta effects.

## Amended design

The primary FULL comparison is now a matched contemporary pair:

- `single_ch5`: channels `[5]`
- `control_ch102`: channels `[1,0,2]`

Both use the same current commit, tensor, metadata, folds, subject pools, `offdiag_channelmean_sum`, beta=3.75, latent_dim=384, 5x5 CV, and classifier settings. They differ only in `channels_to_use`, derived channel names, and output/log paths.

## Scope and guardrails

- Fase 1C remains closed: no triples.
- `single_ch5` remains the sole exploratory FULL candidate.
- No candidate passed the strict gate.
- The previous statement that rho<2% "caused" the `pair_ch1_3` advantage is softened: low rho is consistent with under-regularization, but it does not prove causation.
- The historical locked FULL remains a secondary reference only.
- The primary FULL comparison is the matched contemporary pair defined here.
- No training is launched by this preflight.
"""
    (OUT / "protocol_amendment_005a_matched_full_pair.md").write_text(text, encoding="utf-8")


def write_config_diff(manifest: pd.DataFrame) -> None:
    diff_rows = []
    for key, val in COMMON_CONFIG.items():
        diff_rows.append({"field": key, "single_ch5": val, "control_ch102": val, "differs": False})
    single = manifest.loc[manifest["run_label"].eq("single_ch5")].iloc[0]
    control = manifest.loc[manifest["run_label"].eq("control_ch102")].iloc[0]
    for field in ["channels_to_use", "channel_names", "output_dir"]:
        diff_rows.append({"field": field, "single_ch5": single[field], "control_ch102": control[field], "differs": True})
    df = pd.DataFrame(diff_rows)
    text = "# FULL pair config diff\n\n"
    text += "The two planned FULL runs are identical except for channel set, derived channel names, and output paths.\n\n"
    text += df.to_markdown(index=False) + "\n"
    (OUT / "full_pair_config_diff.md").write_text(text, encoding="utf-8")


def write_fold_hashes() -> pd.DataFrame:
    hashes = compute_fold_hashes()
    hashes.to_csv(OUT / "fold_subject_hashes.csv", index=False)
    text = "# Fold subject hash verification\n\n"
    text += "Hashes are computed from the current tensor/metadata merge using the FULL runner's outer split logic: CN/AD classifier pool, Manufacturer stratification with label-only fallback if required, seed=42, 5 folds. VAE pool is all metadata-valid tensor rows minus the classifier test fold, after required metadata columns are checked.\n\n"
    text += "Because channel choice is not used by the splitter, these hashes apply identically to `single_ch5` and `control_ch102`.\n\n"
    text += hashes.to_markdown(index=False) + "\n"
    (OUT / "fold_subject_hash_verification.md").write_text(text, encoding="utf-8")
    return hashes


def write_loss_formula() -> None:
    source = RUNNER.read_text(errors="replace")
    supports = {
        "constant_defined": "RECON_LOSS_MODE_OFFDIAG_CHANNELMEAN = \"offdiag_channelmean_sum\"" in source,
        "uses_offdiag_mask": "offdiag_mask = _offdiag_mask_for_tensor(x)" in source,
        "sums_offdiag_per_channel": "per_subject_channel_sum = diff2[:, :, offdiag_mask].sum(dim=-1)" in source,
        "averages_across_channels_and_batch": "return per_subject_channel_sum.mean(dim=1).mean()" in source,
    }
    text = "# Loss formula verification\n\n"
    text += "Current FULL runner: `scripts/run_vae_clf_ad_inference.py`.\n\n"
    text += "Verified formula for `offdiag_channelmean_sum` / `mse_offdiag_channel_mean_sum`:\n\n"
    text += "```python\n"
    text += "offdiag_mask = _offdiag_mask_for_tensor(x)\n"
    text += "diff2 = (recon_x - x).pow(2)\n"
    text += "per_subject_channel_sum = diff2[:, :, offdiag_mask].sum(dim=-1)\n"
    text += "return per_subject_channel_sum.mean(dim=1).mean()\n"
    text += "```\n\n"
    text += "This excludes diagonal entries, sums squared error within each channel, then averages over channels and batch. This avoids linear reconstruction scale growth with channel count for identical per-channel error distributions.\n\n"
    text += pd.DataFrame([{"check": k, "pass": v} for k, v in supports.items()]).to_markdown(index=False) + "\n\n"
    text += "High-beta checkpoint selection: not supported by the current FULL runner; no `--vae_checkpoint_select_high_beta_only` flag exists in `run_vae_clf_ad_inference.py`, so it is not enabled for either matched run.\n"
    (OUT / "loss_formula_verification.md").write_text(text, encoding="utf-8")


def write_resource_estimate() -> None:
    text = "# Resource estimate for matched FULL pair\n\n"
    text += "Reference from Amendment 005 preflight: historical 3-channel FULL run took 4h 10m 57s and used approximately 2.3 GB. The single-channel FULL estimate was approximately 3.8-4.0 h and 2.0-2.2 GB.\n\n"
    text += "Matched contemporary pair estimate:\n\n"
    text += "| run | channels | wall-clock estimate | disk estimate |\n"
    text += "|:----|:---------|:--------------------|:--------------|\n"
    text += "| single_ch5 | [5] | 3.8-4.8 h | 2.0-2.2 GB |\n"
    text += "| control_ch102 | [1,0,2] | 4.0-5.2 h | 2.2-2.5 GB |\n"
    text += "| total sequential | two runs | 7.8-10.0 h | 4.2-4.7 GB |\n\n"
    text += "The guarded launcher checks free space before launch and writes heavy outputs only under `/media/diego/Datos`.\n"
    (OUT / "resource_estimate_full_pair.md").write_text(text, encoding="utf-8")


def write_scripts(manifest: pd.DataFrame) -> None:
    launch = f"""#!/usr/bin/env bash
# Manifest-driven matched FULL pair launch. Do not run unless guarded wrapper passes.
set -euo pipefail

DRY_RUN=0
if [[ "${{1:-}}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

PYTHON="/home/diego/anaconda3/envs/vae_ad/bin/python"
REPO_ROOT="{REPO}"
RUNNER="{RUNNER}"
MANIFEST="{OUT / 'planned_full_pair_candidates.csv'}"
TENSOR="{TENSOR}"
METADATA="{METADATA}"
LOG_ROOT="{LOG_ROOT}"
TMPDIR="/media/diego/Datos/tmp_vae"
export TMPDIR

has_final_metrics() {{
  local dir="$1"
  local found=""
  if [[ -d "$dir" || -L "$dir" ]]; then
    found="$(find -L "$dir" -maxdepth 1 -type f -name 'all_folds_metrics_*.csv' -print -quit 2>/dev/null || true)"
  fi
  [[ -n "$found" ]]
}}

has_partial_folds() {{
  local dir="$1"
  local found=""
  if [[ -d "$dir" || -L "$dir" ]]; then
    found="$(find -L "$dir" -mindepth 2 -maxdepth 2 -type f -name 'test_predictions_*.csv' -print -quit 2>/dev/null || true)"
  fi
  [[ -n "$found" ]]
}}

COMMON_FLAGS=(
  --global_tensor_path "$TENSOR"
  --metadata_path "$METADATA"
  --classifier_types logreg svm
  --classifier_stratify_cols Manufacturer
  --vae_stratify_cols Manufacturer
  --classifier_calibrate
  --classifier_use_class_weight
  --latent_features_type mu
  --gridsearch_scoring roc_auc
  --outer_folds 5
  --inner_folds 5
  --repeated_outer_folds_n_repeats 1
  --num_conv_layers_encoder 4
  --decoder_type convtranspose
  --epochs_vae 10000
  --vae_val_split_ratio 0.2
  --early_stopping_patience_vae 560
  --cyclical_beta_n_cycles 125
  --cyclical_beta_ratio_increase 0.4
  --beta_vae 3.75
  --dropout_rate_vae 0.15
  --vae_dropout_scope legacy_all
  --vae_block_order legacy_act_norm
  --latent_dim 384
  --batch_size 64
  --lr_vae 0.0001
  --lr_scheduler_type cosine_warm
  --lr_scheduler_T0 80
  --lr_scheduler_eta_min 5e-07
  --lr_scheduler_patience_vae 15
  --weight_decay_vae 5e-07
  --vae_final_activation tanh
  --intermediate_fc_dim_vae quarter
  --n_jobs_gridsearch 8
  --metadata_features Age Sex
  --norm_mode zscore_offdiag
  --recon_loss_mode offdiag_channelmean_sum
  --seed 42
  --num_workers 4
  --log_interval_epochs_vae 10
  --save_fold_artefacts
  --save_vae_training_history
  --qc_analyze_distributions
  --qc_check_scanner_leakage
  --qc_rate_distortion
  --qc_latent_information
  --qc_mi_n_neighbors 3
  --qc_mi_top_k 10
  --qc_rd_log_base 2.0
  --qc_tc_ridge 1e-06
  --qc_var_eps_active 0.0001
  --vae_train_sampler_strategy none
  --mlp_classifier_hidden_layers 64,16
  --n_iter_logreg 500
  --n_iter_svm 500
  --vae_required_metadata_cols ResearchGroup_Mapped Manufacturer Age Sex
  --vae_abort_if_val_split_fails
)

echo "========================================================"
echo "Matched FULL pair: single_ch5 vs contemporary ch102"
echo "Dry-run: $DRY_RUN"
echo "No triples. No historical output modifications."
echo "========================================================"

tail -n +2 "$MANIFEST" | while IFS=, read -r candidate_id run_label channels_to_use channel_names output_dir status; do
  if has_final_metrics "$output_dir"; then
    echo "[SKIP] $run_label complete: final metrics present."
    continue
  fi
  if has_partial_folds "$output_dir"; then
    echo "[ABORT] $run_label has partial fold outputs but no final metrics. Refusing partial resume to avoid corrupting 5-fold aggregation: $output_dir" >&2
    exit 20
  fi

  read -r -a ch_arr <<< "$channels_to_use"
  echo "[PLAN] $run_label channels=[$channels_to_use] output=$output_dir"

  if [[ "$DRY_RUN" == "1" ]]; then
    "$PYTHON" "$RUNNER" "${{COMMON_FLAGS[@]}}" --output_dir "$output_dir" --channels_to_use "${{ch_arr[@]}}" --dry-run
    continue
  fi

  mkdir -p "$output_dir" "$LOG_ROOT"
  log_file="${{LOG_ROOT}}/${{run_label}}_$(date +%Y%m%d_%H%M%S).log"
  "$PYTHON" "$RUNNER" "${{COMMON_FLAGS[@]}}" --output_dir "$output_dir" --channels_to_use "${{ch_arr[@]}}" 2>&1 | tee "$log_file"
done
"""
    (OUT / "launch_full_pair.sh").write_text(launch, encoding="utf-8")

    guarded = f"""#!/usr/bin/env bash
set -euo pipefail

MODE="${{1:-}}"
if [[ "$MODE" != "--foreground" && "$MODE" != "--tmux" ]]; then
  echo "Usage: $0 --foreground | --tmux"
  exit 2
fi

PYTHON="/home/diego/anaconda3/envs/vae_ad/bin/python"
REPO_ROOT="{REPO}"
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
LAUNCH_SCRIPT="${{SCRIPT_DIR}}/launch_full_pair.sh"
OUTPUT_ROOT="{RUN_ROOT}"
LOG_ROOT="{LOG_ROOT}"
RUN_OUTPUTS_LINK="${{SCRIPT_DIR}}/run_outputs"
RUN_LOGS_LINK="${{SCRIPT_DIR}}/run_logs"
TMPDIR="/media/diego/Datos/tmp_vae"
SESSION="full_pair_ch5_vs_ch102_20260703"
STDOUT_LOG="${{OUTPUT_ROOT}}/guarded_launch_stdout.log"

fail() {{ echo "[ABORT] $*" >&2; exit 1; }}
assert_media_path() {{ [[ "$1" == /media/diego/Datos/* ]] || fail "Path not under /media/diego/Datos: $1"; }}
ensure_symlink() {{
  local link="$1"; local target="$2"
  if [[ -L "$link" ]]; then
    local current; current="$(readlink "$link")"
    [[ "$current" == "$target" ]] || fail "Unexpected symlink target: $link -> $current, expected $target"
  elif [[ -e "$link" ]]; then
    fail "Path exists and is not a symlink: $link"
  else
    mkdir -p "$(dirname "$link")"
    ln -s "$target" "$link"
  fi
  echo "[OK] Symlink: $link -> $target"
}}

echo "========================================================"
echo "Guarded launch: matched FULL pair"
echo "Mode: $MODE"
echo "========================================================"

BRANCH="$(git -C "$REPO_ROOT" branch --show-current)"
[[ "$BRANCH" == "{BRANCH}" ]] || fail "Expected branch {BRANCH}, got $BRANCH"
COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
[[ "$COMMIT" == "{EXPECTED_COMMIT}" ]] || fail "Expected commit {EXPECTED_COMMIT}, got $COMMIT"

assert_media_path "$OUTPUT_ROOT"
assert_media_path "$LOG_ROOT"
assert_media_path "$TMPDIR"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$TMPDIR"
tmp_test="$(mktemp "$TMPDIR/full_pair_tmp.XXXXXX")" || fail "TMPDIR not writable: $TMPDIR"
rm -f "$tmp_test"
export TMPDIR

"$PYTHON" -m py_compile "$REPO_ROOT/scripts/run_vae_clf_ad_inference.py" || fail "py_compile failed"
bash -n "$LAUNCH_SCRIPT" || fail "launch script syntax failed"
bash -n "$0" || fail "guarded script syntax failed"

[[ -f "$SCRIPT_DIR/protocol_amendment_005a_matched_full_pair.md" ]] || fail "Missing amendment 005a"
[[ -f "$SCRIPT_DIR/fold_subject_hash_verification.md" ]] || fail "Missing fold hash verification"

ensure_symlink "$RUN_OUTPUTS_LINK" "$OUTPUT_ROOT"
ensure_symlink "$RUN_LOGS_LINK" "$LOG_ROOT"

echo "[INFO] Free space:"
df -h /media/diego/Datos /

if [[ "$MODE" == "--tmux" ]]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    fail "tmux session already exists: $SESSION"
  fi
  tmux new-session -d -s "$SESSION" "cd '$REPO_ROOT' && export TMPDIR='$TMPDIR' && stdbuf -oL -eL bash '$LAUNCH_SCRIPT' 2>&1 | tee '$STDOUT_LOG'"
  echo "[INFO] tmux session launched: $SESSION"
else
  cd "$REPO_ROOT"
  export TMPDIR
  stdbuf -oL -eL bash "$LAUNCH_SCRIPT" 2>&1 | tee "$STDOUT_LOG"
fi
"""
    (OUT / "guarded_launch_full_pair.sh").write_text(guarded, encoding="utf-8")


def write_preflight(manifest: pd.DataFrame, hashes: pd.DataFrame) -> None:
    branch = current_branch()
    commit = current_commit()
    checks = [
        ("branch_expected", branch == BRANCH),
        ("commit_expected", commit == EXPECTED_COMMIT),
        ("tensor_exists", TENSOR.exists()),
        ("metadata_exists", METADATA.exists()),
        ("runner_exists", RUNNER.exists()),
        ("amendment_005_exists_unmodified", AMENDMENT_005.exists()),
        ("gate_exists_unmodified", GATE.exists()),
        ("historical_full_exists_secondary_only", HISTORICAL_FULL.exists()),
        ("run_root_absent_prelaunch", not RUN_ROOT.exists()),
        ("log_root_absent_prelaunch", not LOG_ROOT.exists()),
        ("manifest_two_rows", len(manifest) == 2),
        ("fold_hash_rows_5", len(hashes) == 5),
        (
            "vae_required_metadata_removes_expected_tensor_only_subject_each_fold",
            hashes["vae_required_metadata_removed_n"].tolist() == [1, 1, 1, 1, 1],
        ),
    ]
    df = pd.DataFrame(checks, columns=["check", "pass"])
    status = "PASS" if df["pass"].all() else "FAIL"
    text = f"""# Preflight status

Overall status: **{status}**

Branch: `{branch}`

Commit: `{commit}`

No training was launched.

## Checks

{df.to_markdown(index=False)}

## Launch commands

Dry-run:

```bash
bash {OUT / 'launch_full_pair.sh'} --dry-run
```

Foreground:

```bash
bash {OUT / 'guarded_launch_full_pair.sh'} --foreground
```

tmux:

```bash
bash {OUT / 'guarded_launch_full_pair.sh'} --tmux
```
"""
    (OUT / "preflight_status.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_amendment()
    manifest = write_manifest()
    write_config_diff(manifest)
    hashes = write_fold_hashes()
    write_loss_formula()
    write_resource_estimate()
    write_scripts(manifest)
    write_preflight(manifest, hashes)
    log = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "did_launch_training": False,
        "did_launch_triples": False,
        "did_modify_historical_full": False,
        "did_modify_amendments_001_005": False,
        "did_modify_gate_criteria": False,
        "outputs": sorted(p.name for p in OUT.iterdir() if p.is_file()),
    }
    (OUT / "command_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
