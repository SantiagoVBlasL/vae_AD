#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/diego/proyectos/vae_AD"
RUN_NAME="recover035_latent384_beta3p75_foldcombat_mfr_age_sex_cleanrepro_20260706"
RUN_DIR="${REPO_ROOT}/results/revision_bspc_2026/${RUN_NAME}"

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [run_directory]" >&2
  exit 2
fi
if [[ $# -eq 1 ]]; then
  RUN_DIR="$1"
fi

echo "run_dir=${RUN_DIR}"
if [[ ! -d "${RUN_DIR}" ]]; then
  echo "status=NOT_CREATED"
  echo "No training output exists."
  exit 0
fi

if [[ -f "${RUN_DIR}/.cleanrepro_complete" ]]; then
  echo "status=COMPLETE"
elif [[ -f "${RUN_DIR}/training_exit_code.txt" ]]; then
  EXIT_CODE="$(cat "${RUN_DIR}/training_exit_code.txt")"
  if [[ "${EXIT_CODE}" == "0" ]]; then
    echo "status=VAE_COMPLETE_DOWNSTREAM_PENDING_OR_RUNNING"
  else
    echo "status=TRAINING_FAILED exit_code=${EXIT_CODE}"
  fi
elif [[ -f "${RUN_DIR}/training_started_utc.txt" ]]; then
  echo "status=TRAINING_OR_INTERRUPTED"
else
  echo "status=PREPARED_NOT_STARTED"
fi

echo
echo "matching_processes:"
pgrep -af "${RUN_NAME}|run_vae_clf_ad_inference.py|run_foldcombat_cleanrepro_downstream_classifier.py" || true

echo
echo "fold_artifacts:"
for fold in 1 2 3 4 5; do
  FOLD_DIR="${RUN_DIR}/fold_${fold}"
  checkpoint="missing"
  history="missing"
  combat="missing"
  cache_train="missing"
  cache_test="missing"
  [[ -f "${FOLD_DIR}/vae_model_fold_${fold}.pt" ]] && checkpoint="present"
  [[ -f "${FOLD_DIR}/vae_train_history_fold_${fold}.joblib" ]] && history="present"
  [[ -f "${FOLD_DIR}/input_harmonization_fitted_objects/foldwise_combat_tensor.joblib" ]] && combat="present"
  [[ -f "${RUN_DIR}/downstream_diagnostic_classifier/latent_cache/fold_${fold}_trainDev_latent_mu.csv" ]] && cache_train="present"
  [[ -f "${RUN_DIR}/downstream_diagnostic_classifier/latent_cache/fold_${fold}_test_latent_mu.csv" ]] && cache_test="present"
  echo "fold=${fold} checkpoint=${checkpoint} history=${history} combat=${combat} train_cache=${cache_train} test_cache=${cache_test}"
done

echo
echo "downstream_artifacts:"
for name in \
  downstream_classifier_oof_predictions.csv \
  downstream_classifier_foldwise_metrics.csv \
  downstream_classifier_primary_metrics.csv \
  frozen_classifier_manifest.csv; do
  path="${RUN_DIR}/downstream_diagnostic_classifier/${name}"
  if [[ -f "${path}" ]]; then
    echo "${name}=present"
  else
    echo "${name}=missing"
  fi
done

LATEST_LOG="$(find "${RUN_DIR}/logs" -maxdepth 1 -type f -name '*.log' -print 2>/dev/null | sort | tail -n 1 || true)"
if [[ -n "${LATEST_LOG}" ]]; then
  echo
  echo "latest_log=${LATEST_LOG}"
  tail -n 40 "${LATEST_LOG}"
fi
