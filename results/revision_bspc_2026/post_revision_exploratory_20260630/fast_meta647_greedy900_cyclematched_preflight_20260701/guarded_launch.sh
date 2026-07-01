#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" != "--confirm-training" ]]; then
  echo "Refusing to launch: pass --confirm-training explicitly."
  exit 2
fi

MODE="${2:-full}"
RUN_ROOT="/media/diego/Datos/vae_AD_results/revision_bspc_2026/post_revision_exploratory_20260630/fast_meta647_greedy900_cyclematched_20260701"
LOG_ROOT="/media/diego/Datos/vae_AD_results/revision_bspc_2026/_launch_logs/post_revision_exploratory_20260630/fast_meta647_greedy900_cyclematched_20260701"
mkdir -p "$LOG_ROOT"

if pgrep -af "run_vae_clf_ad_(ablation|inference)\.py|ablation_canales\.py" >/tmp/fast_greedy900_active_train.txt; then
  echo "Refusing to launch because an active VAE/FAST process was detected:"
  cat /tmp/fast_greedy900_active_train.txt
  exit 3
fi

case "$MODE" in
  full)
    exec bash "/home/diego/proyectos/vae_AD/results/revision_bspc_2026/post_revision_exploratory_20260630/fast_meta647_greedy900_cyclematched_preflight_20260701/launch_greedy900_full.sh"
    ;;
  sentinel)
    exec bash "/home/diego/proyectos/vae_AD/results/revision_bspc_2026/post_revision_exploratory_20260630/fast_meta647_greedy900_cyclematched_preflight_20260701/launch_greedy900_sentinel_only.sh"
    ;;
  *)
    echo "Unknown mode: $MODE. Use full or sentinel."
    exit 4
    ;;
esac
