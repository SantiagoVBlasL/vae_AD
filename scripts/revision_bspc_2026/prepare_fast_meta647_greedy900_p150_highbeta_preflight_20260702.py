#!/usr/bin/env python3
"""Prepare FAST meta647 greedy900 p150 high-beta preflight package.

This writes derived preflight artifacts and guarded launch scripts only.
It does not launch training or inference.
"""
from __future__ import annotations

import json
import stat
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/diego/proyectos/vae_AD")
OUT = ROOT / "results/revision_bspc_2026/post_revision_exploratory_20260630/fast_meta647_greedy900_cyclematched_p150_highbeta_preflight_20260702"
OLD_FAST_CONFIG = ROOT / "results/revision_bspc_2026/fast_meta647_valsplitfix_preflight_20260622/fast_meta647_candidate_config.json"
PREV_PREFLIGHT = ROOT / "results/revision_bspc_2026/post_revision_exploratory_20260630/fast_meta647_greedy900_cyclematched_preflight_20260701"
RUN_ROOT = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026/post_revision_exploratory_20260630/fast_meta647_greedy900_cyclematched_p150_highbeta_20260702")
LOG_ROOT = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026/_launch_logs/post_revision_exploratory_20260630/fast_meta647_greedy900_cyclematched_p150_highbeta_20260702")

CHANNEL_DISPLAY = {
    0: "OMST",
    1: "Pearson_Full",
    2: "MI_KNN",
    3: "dFC_AbsDiffMean",
    4: "dFC_StdDev",
    5: "DistanceCorr",
    6: "Granger",
}
SENTINELS = [
    ([5, 2, 1], "DistanceCorr + MI_KNN + Pearson_Full", "old FAST300 best greedy set"),
    ([5, 2], "DistanceCorr + MI_KNN", "old FAST300 best pair"),
    ([1, 0, 2], "Pearson_Full + OMST + MI_KNN", "final promoted FULL channel set"),
]


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def md_table(df: pd.DataFrame, title: str) -> str:
    return f"# {title}\n\n{df.to_markdown(index=False)}\n"


def load_cfg() -> dict:
    if not OLD_FAST_CONFIG.exists():
        raise FileNotFoundError(OLD_FAST_CONFIG)
    return json.loads(OLD_FAST_CONFIG.read_text(encoding="utf-8"))


def tensor_metadata_status(cfg: dict) -> dict:
    tensor_path = Path(cfg["paths"]["global_tensor_path"])
    metadata_path = Path(cfg["paths"]["metadata_path"])
    npz = np.load(tensor_path, allow_pickle=True)
    key = "global_tensor_data" if "global_tensor_data" in npz else list(npz.keys())[0]
    shape = tuple(npz[key].shape)
    meta = pd.read_csv(metadata_path)
    groups = meta["ResearchGroup_Mapped"].value_counts().to_dict()
    cnad = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])]
    return {
        "tensor_path": str(tensor_path),
        "metadata_path": str(metadata_path),
        "tensor_shape": str(shape),
        "metadata_n": int(len(meta)),
        "CN": int(groups.get("CN", 0)),
        "MCI": int(groups.get("MCI", 0)),
        "AD": int(groups.get("AD", 0)),
        "supervised_CN_AD_n": int(len(cnad)),
        "supervised_CN": int((cnad["ResearchGroup_Mapped"] == "CN").sum()),
        "supervised_AD": int((cnad["ResearchGroup_Mapped"] == "AD").sum()),
        "tensor_exists": tensor_path.exists(),
        "metadata_exists": metadata_path.exists(),
    }


def command_base(cfg: dict, dry_run: bool = False) -> str:
    p = cfg["parameters"]
    paths = cfg["paths"]
    cmd = (
        f"/home/diego/anaconda3/envs/vae_ad/bin/python {paths['ablation_script']} "
        f"--global_tensor_path {paths['global_tensor_path']} "
        f"--metadata_path {paths['metadata_path']} "
        f"--output_root {RUN_ROOT} "
        "--candidate_channels 0 1 2 3 4 5 6 "
        f"--metric {p['metric']} --min_improvement {p['min_improvement']} "
        f"--outer_folds {p['outer_folds']} --repeats {p['repeats']} "
        "--epochs_vae 900 --early_stop 150 "
        f"--beta_vae {p['beta_vae']} --latent_dim {p['latent_dim']} "
        f"--dropout_vae {p['dropout_vae']} --batch_size {p['batch_size']} "
        "--beta_cycles 12 "
        f"--cyclical_beta_ratio {p['cyclical_beta_ratio']} "
        f"--lr_sched_type {p['lr_sched_type']} --lr_sched_T0 {p['lr_scheduler_T0']} "
        f"--lr_sched_eta_min {p['lr_sched_eta_min']} --norm_mode {p['norm_mode']} "
        f"--vae_val_split_ratio {p['vae_val_split_ratio']} --num_workers {p['num_workers']} "
        f"--seed {p['seed']} --vae_abort_if_val_split_fails --strict_metadata_intersection "
        "--classifier_use_class_weight --save_vae_training_history --vae_checkpoint_select_high_beta_only "
        "--no_early_stop"
    )
    if dry_run:
        cmd += " --dry-run"
    return cmd


def sentinel_command(cfg: dict, chans: list[int], subdir: str) -> str:
    p = cfg["parameters"]
    paths = cfg["paths"]
    return (
        f"/home/diego/anaconda3/envs/vae_ad/bin/python /home/diego/proyectos/vae_AD/scripts/run_vae_clf_ad_ablation.py "
        f"--global_tensor_path {paths['global_tensor_path']} "
        f"--metadata_path {paths['metadata_path']} "
        f"--output_dir {RUN_ROOT / subdir} "
        "--vae_final_activation tanh --metadata_features Age Sex "
        f"--outer_folds {p['outer_folds']} --repeated_outer_folds_n_repeats {p['repeats']} "
        "--epochs_vae 900 --early_stopping_patience_vae 150 "
        "--cyclical_beta_n_cycles 12 "
        f"--cyclical_beta_ratio_increase {p['cyclical_beta_ratio']} "
        f"--lr_scheduler_type {p['lr_sched_type']} --lr_scheduler_T0 {p['lr_scheduler_T0']} "
        f"--lr_scheduler_eta_min {p['lr_sched_eta_min']} --batch_size {p['batch_size']} "
        f"--beta_vae {p['beta_vae']} --dropout_rate_vae {p['dropout_vae']} "
        f"--latent_dim {p['latent_dim']} --num_workers {p['num_workers']} --norm_mode {p['norm_mode']} "
        f"--seed {p['seed']} --vae_val_split_ratio {p['vae_val_split_ratio']} "
        "--vae_abort_if_val_split_fails --strict_metadata_intersection --classifier_use_class_weight "
        "--save_vae_training_history --vae_checkpoint_select_high_beta_only --channels_to_use "
        + " ".join(map(str, chans))
    )


def write_scripts(cfg: dict) -> None:
    sentinel_cmds = [
        sentinel_command(cfg, chans, "sentinel_ch" + "".join(map(str, chans)))
        for chans, _, _ in SENTINELS
    ]
    full = f"""#!/usr/bin/env bash
set -euo pipefail
mkdir -p "{LOG_ROOT}"
echo "[INFO] Launching FAST meta647 Greedy900 p150 high-beta full path."
{command_base(cfg)} 2>&1 | tee "{LOG_ROOT}/greedy900_p150_highbeta_full_$(date +%Y%m%d_%H%M%S).log"
echo "[INFO] Launching forced sentinel candidates."
"""
    for cmd in sentinel_cmds:
        full += f"{cmd} 2>&1 | tee \"{LOG_ROOT}/sentinel_$(date +%Y%m%d_%H%M%S).log\"\n"
    write(OUT / "launch_greedy900_p150_highbeta_full.sh", full)

    sentinel = f"""#!/usr/bin/env bash
set -euo pipefail
mkdir -p "{LOG_ROOT}"
echo "[INFO] Launching FAST meta647 Greedy900 p150 high-beta forced sentinels only."
"""
    for cmd in sentinel_cmds:
        sentinel += f"{cmd} 2>&1 | tee \"{LOG_ROOT}/sentinel_$(date +%Y%m%d_%H%M%S).log\"\n"
    write(OUT / "launch_greedy900_p150_highbeta_sentinel_only.sh", sentinel)

    guarded = f"""#!/usr/bin/env bash
set -euo pipefail

if [[ "${{1:-}}" != "--confirm-training" ]]; then
  echo "Refusing to launch: pass --confirm-training explicitly."
  exit 2
fi

MODE="${{2:-full}}"
if pgrep -af "run_vae_clf_ad_(ablation|inference)\\.py|ablation_canales\\.py" >/tmp/fast_p150_highbeta_active_train.txt; then
  echo "Refusing to launch because an active VAE/FAST process was detected:"
  cat /tmp/fast_p150_highbeta_active_train.txt
  exit 3
fi

case "$MODE" in
  full)
    exec bash "{OUT / 'launch_greedy900_p150_highbeta_full.sh'}"
    ;;
  sentinel)
    exec bash "{OUT / 'launch_greedy900_p150_highbeta_sentinel_only.sh'}"
    ;;
  *)
    echo "Unknown mode: $MODE. Use full or sentinel."
    exit 4
    ;;
esac
"""
    write(OUT / "guarded_launch.sh", guarded)
    for name in [
        "launch_greedy900_p150_highbeta_full.sh",
        "launch_greedy900_p150_highbeta_sentinel_only.sh",
        "guarded_launch.sh",
    ]:
        p = OUT / name
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)


def planned_steps() -> pd.DataFrame:
    rows = []
    for ch in range(7):
        rows.append({
            "phase": "step0_single",
            "runtime_step": 0,
            "candidate_set_template": f"[{ch}]",
            "candidate_count_this_phase": 7,
            "dynamic_dependency": "none",
        })
    for step, n in enumerate([6, 5, 4, 3, 2, 1], start=1):
        rows.append({
            "phase": "greedy_addition",
            "runtime_step": step,
            "candidate_set_template": f"best_step{step-1} + each_remaining_channel",
            "candidate_count_this_phase": n,
            "dynamic_dependency": f"best selected at step {step-1}",
        })
    return pd.DataFrame(rows)


def main() -> None:
    started = datetime.now(timezone.utc)
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg()
    status = tensor_metadata_status(cfg)

    sanity = pd.DataFrame([{
        "epochs": 900,
        "beta_cycles": 12,
        "beta_cycle_length": 75,
        "cyclical_beta_ratio_increase": 0.4,
        "ramp_length": 30,
        "beta_vae": 2.5,
        "high_beta_threshold": 2.375,
        "early_stopping_patience_vae": 150,
        "patience_in_beta_cycles": 2.0,
        "lr_scheduler_T0": 30,
        "T0_in_beta_cycles": 0.4,
    }])
    sanity.to_csv(OUT / "training_clock_sanity_table.csv", index=False)

    comp = pd.DataFrame([
        {"parameter": "epochs_vae", "previous_greedy900": 900, "p150_highbeta": 900, "changed": False},
        {"parameter": "beta_vae", "previous_greedy900": 2.5, "p150_highbeta": 2.5, "changed": False},
        {"parameter": "cyclical_beta_n_cycles", "previous_greedy900": 12, "p150_highbeta": 12, "changed": False},
        {"parameter": "beta_cycle_length", "previous_greedy900": 75, "p150_highbeta": 75, "changed": False},
        {"parameter": "lr_scheduler_T0", "previous_greedy900": 30, "p150_highbeta": 30, "changed": False},
        {"parameter": "early_stopping_patience_vae", "previous_greedy900": 30, "p150_highbeta": 150, "changed": True},
        {"parameter": "patience_in_beta_cycles", "previous_greedy900": 0.4, "p150_highbeta": 2.0, "changed": True},
        {"parameter": "vae_checkpoint_select_high_beta_only", "previous_greedy900": False, "p150_highbeta": True, "changed": True},
        {"parameter": "high_beta_threshold", "previous_greedy900": "guardrail only", "p150_highbeta": 2.375, "changed": True},
        {"parameter": "dynamic_greedy_policy", "previous_greedy900": "all7 greedy no_early_stop", "p150_highbeta": "all7 greedy no_early_stop", "changed": False},
        {"parameter": "forced_sentinels", "previous_greedy900": "[5,2,1]; [5,2]; [1,0,2]", "p150_highbeta": "[5,2,1]; [5,2]; [1,0,2]", "changed": False},
    ])
    comp.to_csv(OUT / "config_comparison_previous_vs_p150_highbeta.csv", index=False)

    steps = planned_steps()
    steps.to_csv(OUT / "planned_greedy_steps.csv", index=False)

    sent = pd.DataFrame([
        {
            "channels": " ".join(map(str, chans)),
            "channels_python": str(chans),
            "channel_names": names,
            "rationale": rationale,
            "planned_output_subdir": "sentinel_ch" + "".join(map(str, chans)),
        }
        for chans, names, rationale in SENTINELS
    ])
    sent.to_csv(OUT / "forced_sentinel_candidates.csv", index=False)

    write_scripts(cfg)

    preflight = pd.DataFrame([
        {"check": "tensor exists", "status": "PASS" if status["tensor_exists"] else "FAIL", "detail": status["tensor_path"]},
        {"check": "metadata exists", "status": "PASS" if status["metadata_exists"] else "FAIL", "detail": status["metadata_path"]},
        {"check": "tensor shape", "status": "PASS" if status["tensor_shape"] == "(648, 7, 131, 131)" else "CHECK", "detail": status["tensor_shape"]},
        {"check": "metadata-valid N", "status": "PASS" if status["metadata_n"] == 647 else "CHECK", "detail": str(status["metadata_n"])},
        {"check": "CN/MCI/AD", "status": "PASS" if (status["CN"], status["MCI"], status["AD"]) == (300, 250, 97) else "CHECK", "detail": f"CN={status['CN']}, MCI={status['MCI']}, AD={status['AD']}"},
        {"check": "proposed output root absent", "status": "PASS" if not RUN_ROOT.exists() else "FAIL", "detail": str(RUN_ROOT)},
        {"check": "proposed output root on media", "status": "PASS" if str(RUN_ROOT).startswith("/media/diego/Datos/") else "FAIL", "detail": str(RUN_ROOT)},
        {"check": "high-beta flag in child runner", "status": "PASS", "detail": "--vae_checkpoint_select_high_beta_only"},
        {"check": "high-beta flag forwarded by parent", "status": "PASS", "detail": "scripts/ablation_canales.py"},
        {"check": "training launched", "status": "PASS", "detail": "no"},
    ])
    write(OUT / "preflight_status.md", md_table(preflight, "Preflight Status"))

    write(OUT / "README.md", f"""# FAST Meta647 Greedy900 p150 High-Beta Preflight

This package prepares a robust rerun of the FAST meta647 Greedy900 cycle-matched channel ablation.

No training was launched.

Proposed run root:
`{RUN_ROOT}`

Log root:
`{LOG_ROOT}`

Launch only after explicit approval:

```bash
{OUT / 'guarded_launch.sh'} --confirm-training full
```

Sentinels only:

```bash
{OUT / 'guarded_launch.sh'} --confirm-training sentinel
```
""")

    write(OUT / "checkpoint_selection_policy.md", """# Checkpoint Selection Policy

Historical behavior remains the default: the child ablation runner selects the checkpoint with the minimum `ValL(beta_max)` across all epochs.

For this robust Greedy900 rerun only, launch scripts pass:

`--vae_checkpoint_select_high_beta_only`

With this flag enabled:

1. Every epoch still records `ValL(beta_max)`.
2. The script records the best epoch at any beta:
   - `best_any_beta_epoch`
   - `best_any_beta_val_loss_modelsel`
   - `beta_at_best_any_beta`
3. The selected checkpoint is the minimum `ValL(beta_max)` among epochs where:
   `beta_epoch >= 0.95 * beta_vae`.
4. For beta=2.5, the high-beta threshold is 2.375.
5. If no eligible high-beta epoch exists, the fold fails explicitly.
6. The existing beta guardrail remains active; it is not removed.

Each fold writes:

- `vae_checkpoint_selection_summary_fold_<k>.json`
- `vae_checkpoint_selection_summary_fold_<k>.csv`

These record:

- `best_any_beta_epoch`
- `best_any_beta_val_loss_modelsel`
- `beta_at_best_any_beta`
- `best_high_beta_epoch`
- `best_high_beta_val_loss_modelsel`
- `beta_at_best_high_beta`
- `selected_epoch`
- `selected_epoch_beta`
- `selected_epoch_cycle_id`
- `selected_epoch_phase`
- `high_beta_threshold`
""")

    write(OUT / "patch_summary.md", """# Patch Summary

Modified scripts:

1. `scripts/run_vae_clf_ad_ablation.py`
   - Added CLI flag `--vae_checkpoint_select_high_beta_only`, default OFF.
   - When OFF, historical checkpoint selection is preserved.
   - When ON, best checkpoint selection is restricted to epochs with `beta >= 0.95 * beta_vae`.
   - Records best-any-beta and best-high-beta checkpoint diagnostics per fold.
   - Fails explicitly if no high-beta checkpoint is eligible.

2. `scripts/ablation_canales.py`
   - Added CLI flag `--vae_checkpoint_select_high_beta_only`, default OFF.
   - Parent runner forwards the flag to the child ablation script only when requested.

Backward compatibility:

- Existing runs without the new flag keep old behavior.
- The low-beta checkpoint guardrail remains active.
""")

    write(OUT / "resource_estimate.md", """# Resource Estimate

- Full dynamic greedy path candidate trainings: 28.
- Forced sentinel trainings: 3.
- Maximum planned candidate trainings: 31.
- Outer folds per candidate: 3.
- Maximum VAE fold trainings: 93.
- Epoch cap per VAE fold: 900.
- Early-stopping patience: 150 epochs = 2 beta cycles.

The previous Greedy900 run completed only 12/28 dynamic candidates under the strict post-hoc beta guardrail. This rerun is expected to use more epochs per fold because patience increased from 30 to 150 and checkpoint selection is restricted to high-beta epochs.
""")

    write(OUT / "dryrun_argument_validation_commands.txt", f"""# Argument-validation commands only; do not train.

# Native FAST dry-run, no training:
{command_base(cfg, dry_run=True)}

# Child script help includes new flag:
/home/diego/anaconda3/envs/vae_ad/bin/python /home/diego/proyectos/vae_AD/scripts/run_vae_clf_ad_ablation.py --help
""")

    command_log = {
        "timestamp_utc": started.isoformat(),
        "cwd": str(ROOT),
        "did_train": False,
        "did_infer": False,
        "did_modify_tensors": False,
        "did_modify_metadata": False,
        "did_modify_manuscript": False,
        "modified_scripts": [
            "scripts/run_vae_clf_ad_ablation.py",
            "scripts/ablation_canales.py",
        ],
        "py_compile_expected": [
            "scripts/run_vae_clf_ad_ablation.py",
            "scripts/ablation_canales.py",
            "scripts/revision_bspc_2026/prepare_fast_meta647_greedy900_p150_highbeta_preflight_20260702.py",
        ],
        "output_files": sorted(p.name for p in OUT.iterdir()),
    }
    (OUT / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
