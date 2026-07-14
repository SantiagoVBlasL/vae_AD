#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/diego/proyectos/vae_AD"
RUN_NAME="recover035_latent384_beta3p75_foldcombat_mfr_age_sex_cleanrepro_20260706"
RUN_DIR="${REPO_ROOT}/results/revision_bspc_2026/${RUN_NAME}"
BIG_DISK_RUN_DIR="/media/diego/Datos/vae_AD_results/revision_bspc_2026/${RUN_NAME}"
LOCKED_RUN_DIR="${REPO_ROOT}/results/revision_bspc_2026/recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
CONFIG="configs/runs/adni_v5_1c_recover035_latent384_beta3p75_foldcombat_mfr_age_sex_cleanrepro_20260706.json"
REFERENCE_CONFIG="configs/runs/adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5.json"
PYTHON="/home/diego/anaconda3/envs/vae_ad/bin/python"
PY_LAUNCHER="scripts/revision_bspc_2026/run_adni_v5_1c_foldcombat_cleanrepro_20260706.py"
DOWNSTREAM_HELPER="scripts/revision_bspc_2026/run_foldcombat_cleanrepro_downstream_classifier.py"

usage() {
  echo "Usage:"
  echo "  $0 --preflight-only"
  echo "  $0 --confirm-training"
  echo
  echo "No training starts unless --confirm-training is supplied exactly."
}

if [[ $# -ne 1 ]]; then
  usage
  exit 2
fi

MODE="$1"
if [[ "${MODE}" != "--preflight-only" && "${MODE}" != "--confirm-training" ]]; then
  usage
  exit 2
fi

cd "${REPO_ROOT}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: Python executable is unavailable: ${PYTHON}" >&2
  exit 1
fi

REQUIRED_FILES=(
  "${CONFIG}"
  "${REFERENCE_CONFIG}"
  "${PY_LAUNCHER}"
  "${DOWNSTREAM_HELPER}"
  "scripts/run_vae_clf_ad_inference.py"
  "scripts/revision_bspc_2026/foldwise_combat_input_harmonization.py"
  "scripts/revision_bspc_2026/run_recover035_latent384_beta3p75_stageB_oof_score_calibration.py"
)
for path in "${REQUIRED_FILES[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: required file missing: ${path}" >&2
    exit 1
  fi
done

CURRENT_BRANCH="$(git branch --show-current)"
if [[ "${CURRENT_BRANCH}" != "exploratory/post-revision-20260630" ]]; then
  echo "ERROR: expected branch exploratory/post-revision-20260630; found ${CURRENT_BRANCH}" >&2
  exit 1
fi

if [[ "${MODE}" == "--preflight-only" ]]; then
  "${PYTHON}" "${PY_LAUNCHER}" \
    --config "${CONFIG}" \
    --reference-config "${REFERENCE_CONFIG}" \
    --dry-run
  echo "PREFLIGHT_ONLY_OK: no training was launched and no run directory was created."
  exit 0
fi

COLLISIONS=(
  "${RUN_DIR}"
  "${BIG_DISK_RUN_DIR}"
  "${REPO_ROOT}/results/revision_bspc_2026/${RUN_NAME}_split_preview.csv"
  "${REPO_ROOT}/results/revision_bspc_2026/${RUN_NAME}_split_preview_summary.csv"
)
for path in "${COLLISIONS[@]}"; do
  if [[ -e "${path}" || -L "${path}" ]]; then
    echo "ERROR: clean output collision; refusing overwrite or reuse: ${path}" >&2
    exit 1
  fi
done

GLOBAL_TENSOR="$(
  "${PYTHON}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["global_tensor_path"])' \
    "${CONFIG}"
)"
METADATA="$(
  "${PYTHON}" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["metadata_path"])' \
    "${CONFIG}"
)"
for path in "${GLOBAL_TENSOR}" "${METADATA}"; do
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: immutable experiment input missing: ${path}" >&2
    exit 1
  fi
done
for fold in 1 2 3 4 5; do
  for split_file in train_dev_subjects_fold.csv test_subjects_fold.csv; do
    path="${LOCKED_RUN_DIR}/fold_${fold}/${split_file}"
    if [[ ! -f "${path}" ]]; then
      echo "ERROR: locked split reference missing: ${path}" >&2
      exit 1
    fi
  done
done

mkdir -p "${RUN_DIR}/provenance/source_snapshot"
PROVENANCE_DIR="${RUN_DIR}/provenance"
SOURCE_SNAPSHOT="${PROVENANCE_DIR}/source_snapshot"

mapfile -t PACKAGE_SOURCES < <(find src/betavae_xai -type f -name '*.py' -print | sort)
SOURCE_FILES=(
  "${CONFIG}"
  "${REFERENCE_CONFIG}"
  "${PY_LAUNCHER}"
  "${DOWNSTREAM_HELPER}"
  "scripts/revision_bspc_2026/guarded_launch_foldcombat_cleanrepro_20260706.sh"
  "scripts/revision_bspc_2026/monitor_foldcombat_cleanrepro_20260706.sh"
  "scripts/run_vae_clf_ad_inference.py"
  "scripts/revision_bspc_2026/foldwise_combat_input_harmonization.py"
  "scripts/revision_bspc_2026/run_recover035_latent384_beta3p75_stageB_oof_score_calibration.py"
  "${PACKAGE_SOURCES[@]}"
)

cp --parents "${SOURCE_FILES[@]}" "${SOURCE_SNAPSHOT}"
sha256sum "${SOURCE_FILES[@]}" > "${PROVENANCE_DIR}/live_source_sha256.txt"
find "${SOURCE_SNAPSHOT}" -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > "${PROVENANCE_DIR}/source_snapshot_sha256.txt"

git rev-parse HEAD > "${PROVENANCE_DIR}/git_head.txt"
git branch --show-current > "${PROVENANCE_DIR}/git_branch.txt"
git status --porcelain=v1 -uall > "${PROVENANCE_DIR}/git_status_porcelain.txt"
git diff --binary > "${PROVENANCE_DIR}/git_worktree.diff"
git diff --cached --binary > "${PROVENANCE_DIR}/git_index.diff"
git submodule status --recursive > "${PROVENANCE_DIR}/git_submodule_status.txt"

sha256sum "${GLOBAL_TENSOR}" > "${PROVENANCE_DIR}/input_tensor_sha256.txt"
sha256sum "${METADATA}" > "${PROVENANCE_DIR}/metadata_sha256.txt"
stat --printf='%n\t%s\t%y\n' "${GLOBAL_TENSOR}" "${METADATA}" \
  > "${PROVENANCE_DIR}/input_file_stat.tsv"

"${PYTHON}" -c \
  'import json,sys; from pathlib import Path; cfg=json.load(open(sys.argv[1])); root=Path(sys.argv[2]); cfg["resolved_paths"]={k:str((Path(v) if Path(v).is_absolute() else root/Path(v)).resolve()) for k,v in cfg["paths"].items()}; print(json.dumps(cfg,indent=2,sort_keys=True))' \
  "${CONFIG}" "${REPO_ROOT}" > "${PROVENANCE_DIR}/resolved_config.json"

date --utc --iso-8601=seconds > "${PROVENANCE_DIR}/prepared_utc.txt"
touch "${RUN_DIR}/.cleanrepro_prepared"

PREFLIGHT_LOG="${PROVENANCE_DIR}/trainer_dry_run.log"
"${PYTHON}" "${PY_LAUNCHER}" \
  --config "${CONFIG}" \
  --reference-config "${REFERENCE_CONFIG}" \
  --dry-run > "${PREFLIGHT_LOG}" 2>&1

sha256sum -c "${PROVENANCE_DIR}/live_source_sha256.txt"

LOG_DIR="${RUN_DIR}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date --utc +%Y%m%dT%H%M%SZ)"
TRAIN_LOG="${LOG_DIR}/training_${TIMESTAMP}.log"
TRAIN_COMMAND=(
  "${PYTHON}"
  "${PY_LAUNCHER}"
  "--config"
  "${CONFIG}"
  "--reference-config"
  "${REFERENCE_CONFIG}"
  "--confirm-training"
)
printf '%q ' "${TRAIN_COMMAND[@]}" > "${PROVENANCE_DIR}/executed_training_command.txt"
printf '\n' >> "${PROVENANCE_DIR}/executed_training_command.txt"

echo "$$" > "${RUN_DIR}/guarded_launcher_pid.txt"
date --utc --iso-8601=seconds > "${RUN_DIR}/training_started_utc.txt"
set +e
"${TRAIN_COMMAND[@]}" 2>&1 | tee "${TRAIN_LOG}"
TRAIN_RC="${PIPESTATUS[0]}"
set -e
echo "${TRAIN_RC}" > "${RUN_DIR}/training_exit_code.txt"
if [[ "${TRAIN_RC}" -ne 0 ]]; then
  echo "ERROR: trainer failed with exit code ${TRAIN_RC}; downstream classifier was not run." >&2
  exit "${TRAIN_RC}"
fi

printf 'fold\ttrain_order_sha256\ttest_order_sha256\torder_matches_locked\n' \
  > "${PROVENANCE_DIR}/fold_subject_split_parity.tsv"
for fold in 1 2 3 4 5; do
  FOLD_DIR="${RUN_DIR}/fold_${fold}"
  REQUIRED_FOLD_OUTPUTS=(
    "${FOLD_DIR}/vae_model_fold_${fold}.pt"
    "${FOLD_DIR}/vae_train_history_fold_${fold}.joblib"
    "${FOLD_DIR}/vae_norm_params.joblib"
    "${FOLD_DIR}/train_dev_subjects_fold.csv"
    "${FOLD_DIR}/test_subjects_fold.csv"
    "${FOLD_DIR}/input_harmonization_leakage_guard.csv"
    "${FOLD_DIR}/input_harmonization_fit_audit.csv"
    "${FOLD_DIR}/input_harmonization_channel_shift_summary.csv"
    "${FOLD_DIR}/input_harmonization_tensor_sample_manifest.csv"
    "${FOLD_DIR}/input_harmonization_fitted_objects/foldwise_combat_tensor.joblib"
    "${RUN_DIR}/downstream_diagnostic_classifier/latent_cache/fold_${fold}_trainDev_latent_mu.csv"
    "${RUN_DIR}/downstream_diagnostic_classifier/latent_cache/fold_${fold}_test_latent_mu.csv"
  )
  for path in "${REQUIRED_FOLD_OUTPUTS[@]}"; do
    if [[ ! -f "${path}" ]]; then
      echo "ERROR: required fold output missing; downstream classifier blocked: ${path}" >&2
      exit 1
    fi
  done
  "${PYTHON}" -c \
    'import hashlib,pandas as pd,sys; f=sys.argv[1]; newtr=pd.read_csv(sys.argv[2])["SubjectID"].astype(str).tolist(); oldtr=pd.read_csv(sys.argv[3])["SubjectID"].astype(str).tolist(); newte=pd.read_csv(sys.argv[4])["SubjectID"].astype(str).tolist(); oldte=pd.read_csv(sys.argv[5])["SubjectID"].astype(str).tolist(); assert newtr==oldtr, f"fold {f} train/dev subject order mismatch"; assert newte==oldte, f"fold {f} test subject order mismatch"; h=lambda x:hashlib.sha256("\\n".join(x).encode()).hexdigest(); print(f"{f}\\t{h(newtr)}\\t{h(newte)}\\tTrue")' \
    "${fold}" \
    "${FOLD_DIR}/train_dev_subjects_fold.csv" \
    "${LOCKED_RUN_DIR}/fold_${fold}/train_dev_subjects_fold.csv" \
    "${FOLD_DIR}/test_subjects_fold.csv" \
    "${LOCKED_RUN_DIR}/fold_${fold}/test_subjects_fold.csv" \
    >> "${PROVENANCE_DIR}/fold_subject_split_parity.tsv"
done

DOWNSTREAM_LOG="${LOG_DIR}/downstream_diagnostic_classifier_${TIMESTAMP}.log"
DOWNSTREAM_COMMAND=(
  "${PYTHON}"
  "${DOWNSTREAM_HELPER}"
  "--run-dir"
  "${RUN_DIR}"
  "--n-jobs"
  "8"
)
printf '%q ' "${DOWNSTREAM_COMMAND[@]}" > "${PROVENANCE_DIR}/executed_downstream_command.txt"
printf '\n' >> "${PROVENANCE_DIR}/executed_downstream_command.txt"
"${DOWNSTREAM_COMMAND[@]}" 2>&1 | tee "${DOWNSTREAM_LOG}"

FINAL_REQUIRED=(
  "${RUN_DIR}/downstream_diagnostic_classifier/downstream_classifier_oof_predictions.csv"
  "${RUN_DIR}/downstream_diagnostic_classifier/downstream_classifier_foldwise_metrics.csv"
  "${RUN_DIR}/downstream_diagnostic_classifier/downstream_classifier_primary_metrics.csv"
  "${RUN_DIR}/downstream_diagnostic_classifier/frozen_classifier_manifest.csv"
)
for path in "${FINAL_REQUIRED[@]}"; do
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: required downstream output missing: ${path}" >&2
    exit 1
  fi
done

date --utc --iso-8601=seconds > "${RUN_DIR}/cleanrepro_completed_utc.txt"
touch "${RUN_DIR}/.cleanrepro_complete"
echo "CLEAN_REPRO_COMPLETE: ${RUN_DIR}"
