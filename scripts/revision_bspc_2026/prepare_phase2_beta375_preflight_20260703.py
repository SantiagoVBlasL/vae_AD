#!/usr/bin/env python3
"""Prepare Phase 2 beta=3.75 robustness preflight for FAST finalists.

This script writes only small local preflight artifacts and launch wrappers.
It does not run training.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd


REPO = Path("/home/diego/proyectos/vae_AD")
PACKAGE = REPO / (
    "results/revision_bspc_2026/post_revision_exploratory_20260630/"
    "fast_phase2_beta375_preflight_20260703"
)
RUN_ROOT = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/post_revision_exploratory_20260630/"
    "fast_phase2_beta375_finalists_20260703"
)
LOG_ROOT = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/_launch_logs/"
    "post_revision_exploratory_20260630/fast_phase2_beta375_finalists_20260703"
)
FASE_ROOT = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/post_revision_exploratory_20260630/"
    "fast_exhaustive_singles_pairs_20260702"
)
FASE_LOG_ROOT = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026/_launch_logs/"
    "post_revision_exploratory_20260630/fast_exhaustive_singles_pairs_20260702"
)
TENSOR = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026/"
    "adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/"
    "GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)
METADATA = REPO / "results/revision_bspc_2026/adni_035_metadata_rescue_preflight/patched_metadata_candidate.csv"
GATE = REPO / (
    "results/revision_bspc_2026/post_revision_exploratory_20260630/"
    "fast_channelmean_loss_ablation_beta_matched_20260702/gate_criteria_preregistered.md"
)
SHORTLIST = REPO / (
    "results/revision_bspc_2026/post_revision_exploratory_20260630/"
    "fast_exhaustive_singles_pairs_gate_completion_20260703/phase2_shortlist_recommendation.md"
)
RUNNER = REPO / "scripts/run_vae_clf_ad_ablation.py"

CHANNEL_NAMES = {
    0: "Pearson_OMST_GCE_Signed_Weighted",
    1: "Pearson_Full_FisherZ_Signed",
    2: "MI_KNN_Symmetric",
    3: "dFC_AbsDiffMean",
    4: "dFC_StdDev",
    5: "DistanceCorr",
    6: "Granger_F_lag1",
}

CANDIDATES = [
    ("single_ch5", [5], "finalist"),
    ("pair_ch1_3", [1, 3], "finalist"),
    ("pair_ch2_5", [2, 5], "finalist"),
    ("control_ch102", [1, 0, 2], "locked_control_rerun"),
]


def run_capture(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True, check=False)
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def current_branch() -> str:
    rc, out = run_capture(["git", "branch", "--show-current"])
    return out.strip() if rc == 0 else f"ERROR:{out}"


def script_contains_required_patches() -> dict[str, bool]:
    text = RUNNER.read_text(errors="replace")
    return {
        "supports_recon_loss_mode": "--recon_loss_mode" in text,
        "supports_offdiag_channelmean_sum": "offdiag_channelmean_sum" in text,
        "supports_high_beta_checkpoint_flag": "--vae_checkpoint_select_high_beta_only" in text,
        "supports_strict_metadata_intersection": "--strict_metadata_intersection" in text,
        "supports_abort_if_val_split_fails": "--vae_abort_if_val_split_fails" in text,
    }


def parse_log_seconds() -> pd.DataFrame:
    rows = []
    pattern = re.compile(r"Pipeline completed in\s+([0-9.]+)s")
    for log_file in sorted(FASE_LOG_ROOT.glob("*.log")):
        text = log_file.read_text(errors="replace")
        match = pattern.search(text)
        if not match:
            continue
        name = log_file.name
        candidate = re.sub(r"_\d{8}_\d{6}\.log$", "", name)
        rows.append(
            {
                "source_log": str(log_file),
                "candidate": candidate,
                "seconds": float(match.group(1)),
                "minutes": float(match.group(1)) / 60.0,
            }
        )
    return pd.DataFrame(rows)


def write_manifest() -> pd.DataFrame:
    rows = []
    for idx, (label, channels, role) in enumerate(CANDIDATES, start=1):
        rows.append(
            {
                "candidate_id": idx,
                "run_label": label,
                "channels": " ".join(str(c) for c in channels),
                "channel_names": "; ".join(CHANNEL_NAMES[c] for c in channels),
                "cardinality": len(channels),
                "role": role,
                "beta_vae": 3.75,
                "status": "new",
                "output_dir": str(RUN_ROOT / label),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(PACKAGE / "planned_phase2_candidates.csv", index=False)
    return df


def write_protocol_amendment() -> None:
    text = f"""# Protocol Amendment 003 — Phase 2 beta-robustness re-test

Date: {datetime.now().strftime('%Y-%m-%d')}

This amendment stages Phase 2 of the post-revision exploratory channel-ablation branch. It is written before launching any Phase 2 training.

## Rationale

The Fase 1A/1B gate-completion audit found that all three shortlisted finalists passed the **LOOSE** AUC-superiority gate, defined as point-estimate ROC-AUC delta >= 0.015 versus the locked `[1,0,2]` control at beta=2.50:

- `single_ch5` `[5]`
- `pair_ch1_3` `[1,3]`
- `pair_ch2_5` `[2,5]`

None of the three passed the **STRICT** gate, defined as 95% CI lower bound >= 0.015.

Per `gate_criteria_preregistered.md`, this beta=3.75 robustness re-test is the pre-specified next step: "An advancing candidate must hold its lead (or tie within noise) when re-tested at beta=3.75 ...".

## Hold-lead criterion for Phase 2

A candidate is considered to hold its lead if it remains loose-gate-eligible at beta=3.75 and its rank order relative to the other finalists and the beta=3.75 `[1,0,2]` control does not reverse.

This task does **not** decide a final winner. It only prepares the beta=3.75 re-test.

## Explicit deferrals and invariants

- Fase 1C triples remain explicitly deferred pending the Phase 2 result.
- `gate_criteria_preregistered.md` remains unmodified.
- Prior amendments remain unmodified.
- Fase 1A/1B outputs, beta pilot arms, and the beta=2.50 locked control directory remain unmodified.
- The `[1,0,2]` control is re-run at beta=3.75 for like-for-like comparison.
"""
    (PACKAGE / "protocol_amendment_003_phase2_beta_robustness.md").write_text(text, encoding="utf-8")


def write_launch_scripts() -> None:
    launch = f"""#!/usr/bin/env bash
# Phase 2 beta=3.75 robustness re-test launcher.
# Do not run directly unless guarded_launch_phase2_beta375.sh has passed checks.
set -euo pipefail

DRY_RUN=0
if [[ "${{1:-}}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

PYTHON="/home/diego/anaconda3/envs/vae_ad/bin/python"
ABLATION_SCRIPT="{RUNNER}"
MANIFEST="{PACKAGE / 'planned_phase2_candidates.csv'}"
TENSOR="{TENSOR}"
METADATA="{METADATA}"
OUTPUT_ROOT="{RUN_ROOT}"
LOG_ROOT="{LOG_ROOT}"
TMPDIR="/media/diego/Datos/tmp_vae"
export TMPDIR

if [[ "$DRY_RUN" != "1" ]]; then
  mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT"
fi

has_metrics() {{
  local dir="$1"
  local found=""
  if [[ -d "$dir" || -L "$dir" ]]; then
    found="$(find -L "$dir" -maxdepth 1 -type f -name 'all_folds_metrics_*.csv' -print -quit 2>/dev/null || true)"
  fi
  [[ -n "$found" ]]
}}

COMMON_FLAGS=(
  --global_tensor_path "$TENSOR"
  --metadata_path "$METADATA"
  --vae_final_activation tanh
  --metadata_features Age Sex
  --outer_folds 3
  --repeated_outer_folds_n_repeats 1
  --epochs_vae 800
  --early_stopping_patience_vae 150
  --cyclical_beta_n_cycles 10
  --cyclical_beta_ratio_increase 0.4
  --lr_scheduler_type cosine_warm
  --lr_scheduler_T0 80
  --lr_scheduler_eta_min 5e-07
  --batch_size 64
  --latent_dim 128
  --dropout_rate_vae 0.15
  --num_workers 4
  --norm_mode zscore_offdiag
  --seed 42
  --vae_val_split_ratio 0.2
  --recon_loss_mode offdiag_channelmean_sum
  --beta_vae 3.75
  --vae_abort_if_val_split_fails
  --strict_metadata_intersection
  --classifier_use_class_weight
  --save_vae_training_history
  --vae_checkpoint_select_high_beta_only
)

echo "========================================================"
echo "Phase 2 beta=3.75 robustness re-test"
echo "Candidates: 3 finalists + beta=3.75 [1,0,2] control"
echo "Output root: $OUTPUT_ROOT"
echo "Log root:    $LOG_ROOT"
echo "Dry-run:     $DRY_RUN"
echo "========================================================"

tail -n +2 "$MANIFEST" | while IFS=, read -r candidate_id run_label channels channel_names cardinality role beta_vae status output_dir; do
  if [[ "$status" != "new" ]]; then
    echo "[ABORT] Unexpected non-new status for $run_label: $status"
    exit 10
  fi
  if has_metrics "$output_dir"; then
    echo "[SKIP] $run_label channels=[$channels] metrics already present."
    continue
  fi

  read -r -a ch_arr <<< "$channels"
  log_file="${{LOG_ROOT}}/${{run_label}}_$(date +%Y%m%d_%H%M%S).log"
  echo "[PLAN] $run_label role=$role beta=$beta_vae channels=[$channels] output=$output_dir"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[DRY-RUN] would run Python for $run_label"
    continue
  fi

  mkdir -p "$output_dir"
  "$PYTHON" "$ABLATION_SCRIPT" \\
    "${{COMMON_FLAGS[@]}}" \\
    --output_dir "$output_dir" \\
    --channels_to_use "${{ch_arr[@]}}" \\
    2>&1 | tee "$log_file"
done

echo "========================================================"
echo "[DONE] Phase 2 manifest traversal complete."
echo "Fase 1C triples remain deferred."
echo "========================================================"
"""
    (PACKAGE / "launch_phase2_beta375.sh").write_text(launch, encoding="utf-8")

    guarded = f"""#!/usr/bin/env bash
# Guarded wrapper for Phase 2 beta=3.75 robustness re-test.
set -euo pipefail

MODE="${{1:-}}"
if [[ "$MODE" != "--foreground" && "$MODE" != "--tmux" ]]; then
  echo "Usage: $0 --foreground | --tmux"
  exit 2
fi

PYTHON="/home/diego/anaconda3/envs/vae_ad/bin/python"
REPO_ROOT="{REPO}"
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
LAUNCH_SCRIPT="${{SCRIPT_DIR}}/launch_phase2_beta375.sh"
MANIFEST="${{SCRIPT_DIR}}/planned_phase2_candidates.csv"
OUTPUT_ROOT="{RUN_ROOT}"
LOG_ROOT="{LOG_ROOT}"
PACKAGE_RUN_OUTPUTS="${{SCRIPT_DIR}}/run_outputs"
PACKAGE_RUN_LOGS="${{SCRIPT_DIR}}/run_logs"
TMPDIR="/media/diego/Datos/tmp_vae"
SESSION="phase2_beta375_20260703"
STDOUT_LOG="${{OUTPUT_ROOT}}/guarded_launch_stdout.log"

fail() {{
  echo "[ABORT] $*" >&2
  exit 1
}}

assert_media_path() {{
  local p="$1"
  [[ "$p" == /media/diego/Datos/* ]] || fail "Path is not under /media/diego/Datos: $p"
}}

ensure_symlink() {{
  local link="$1"
  local target="$2"
  if [[ -L "$link" ]]; then
    local current
    current="$(readlink "$link")"
    [[ "$current" == "$target" ]] || fail "Unexpected symlink target for $link: $current != $target"
    echo "[OK] Symlink already correct: $link -> $target"
  elif [[ -e "$link" ]]; then
    fail "Path exists and is not a symlink: $link"
  else
    mkdir -p "$(dirname "$link")"
    ln -s "$target" "$link"
    echo "[OK] Created symlink: $link -> $target"
  fi
}}

echo "========================================================"
echo "Guarded launch: Phase 2 beta=3.75 robustness re-test"
echo "Mode: $MODE"
echo "No triples will be launched."
echo "========================================================"

BRANCH="$(git -C "$REPO_ROOT" branch --show-current)"
[[ "$BRANCH" == "exploratory/post-revision-20260630" ]] || fail "Expected branch exploratory/post-revision-20260630, got $BRANCH"
echo "[OK] Branch: $BRANCH"

assert_media_path "$OUTPUT_ROOT"
assert_media_path "$LOG_ROOT"
assert_media_path "$TMPDIR"
echo "[OK] Output/log/tmp roots are on /media/diego/Datos"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$TMPDIR"
tmp_test="$(mktemp "$TMPDIR/phase2_tmp.XXXXXX")" || fail "TMPDIR is not writable: $TMPDIR"
rm -f "$tmp_test"
export TMPDIR
echo "[OK] TMPDIR writable and exported: $TMPDIR"

"$PYTHON" -m py_compile "$REPO_ROOT/scripts/run_vae_clf_ad_ablation.py" || fail "py_compile failed for ablation runner"
bash -n "$LAUNCH_SCRIPT" || fail "bash -n failed for launch script"
bash -n "$0" || fail "bash -n failed for guarded wrapper"
echo "[OK] Syntax checks passed"

[[ -f "$SCRIPT_DIR/protocol_amendment_003_phase2_beta_robustness.md" ]] || fail "Missing protocol amendment 003"
[[ -f "$MANIFEST" ]] || fail "Missing manifest: $MANIFEST"

manifest_check="$("$PYTHON" - "$MANIFEST" <<'PY'
import pandas as pd, sys
df = pd.read_csv(sys.argv[1])
expected = {{
    "single_ch5": "5",
    "pair_ch1_3": "1 3",
    "pair_ch2_5": "2 5",
    "control_ch102": "1 0 2",
}}
checks = [
    len(df) == 4,
    set(df.run_label) == set(expected),
    all(str(df.set_index("run_label").loc[k, "channels"]) == v for k, v in expected.items()),
    df.beta_vae.astype(float).eq(3.75).all(),
    df.status.eq("new").all(),
    not df.channels.str.split().apply(lambda x: len(x) == 3 and x != ["1","0","2"]).any(),
]
print("PASS" if all(checks) else "FAIL")
PY
)"
[[ "$manifest_check" == "PASS" ]] || fail "Manifest validation failed"
echo "[OK] Manifest validation passed"

ensure_symlink "$PACKAGE_RUN_OUTPUTS" "$OUTPUT_ROOT"
ensure_symlink "$PACKAGE_RUN_LOGS" "$LOG_ROOT"

echo "[INFO] Free space:"
df -h /media/diego/Datos /

if [[ "$MODE" == "--tmux" ]]; then
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    fail "tmux session already exists: $SESSION"
  fi
  tmux new-session -d -s "$SESSION" "cd '$REPO_ROOT' && export TMPDIR='$TMPDIR' && stdbuf -oL -eL bash '$LAUNCH_SCRIPT' 2>&1 | tee '$STDOUT_LOG'"
  echo "[INFO] tmux session launched: $SESSION"
  echo "[INFO] Monitor: tmux attach -t $SESSION"
else
  cd "$REPO_ROOT"
  export TMPDIR
  stdbuf -oL -eL bash "$LAUNCH_SCRIPT" 2>&1 | tee "$STDOUT_LOG"
fi
"""
    (PACKAGE / "guarded_launch_phase2_beta375.sh").write_text(guarded, encoding="utf-8")


def write_resource_estimate(timing_df: pd.DataFrame) -> None:
    finalist_labels = ["single_ch5", "pair_ch1_3", "pair_ch2_5"]
    finalist_timing = timing_df[timing_df["candidate"].isin(finalist_labels)].copy()
    pair_timing = timing_df[timing_df["candidate"].str.startswith("pair_ch", na=False)].copy()
    single_timing = timing_df[timing_df["candidate"].str.startswith("single_ch", na=False)].copy()
    all_mean = timing_df["minutes"].mean() if not timing_df.empty else float("nan")
    finalist_sum = finalist_timing["minutes"].sum()
    control_proxy = pair_timing["minutes"].mean()
    total_est = finalist_sum + control_proxy
    text = f"""# Resource estimate

Observed timing source: Fase 1A/1B launch logs under `{FASE_LOG_ROOT}`.

Training configuration is identical to Fase 1A/1B except `beta_vae=3.75`, so this estimate uses observed beta=2.50 wall-clock as a planning proxy.

## Observed timings

- Logs with parsed `Pipeline completed in ...s`: {len(timing_df)}
- Mean all parsed candidates: {all_mean:.2f} min
- Mean parsed singles: {single_timing['minutes'].mean():.2f} min
- Mean parsed pairs: {pair_timing['minutes'].mean():.2f} min

## Phase 2 estimate

Parsed finalist times:

{finalist_timing[['candidate','minutes']].to_markdown(index=False) if not finalist_timing.empty else 'No finalist timing rows parsed.'}

Control `[1,0,2]` is a 3-channel run and has no direct Fase 1A/1B timing analogue. Estimated with mean parsed pair timing as a conservative local proxy: {control_proxy:.2f} min.

Estimated sequential wall-clock: **{total_est:.2f} min ({total_est/60.0:.2f} h)** for four candidates.

This is a planning estimate only. Actual beta=3.75 early stopping can shift runtime.
"""
    (PACKAGE / "resource_estimate.md").write_text(text, encoding="utf-8")


def write_preflight_status(manifest: pd.DataFrame, patches: dict[str, bool]) -> None:
    branch = current_branch()
    checks = {
        "branch_expected": branch == "exploratory/post-revision-20260630",
        "tensor_exists": TENSOR.exists(),
        "metadata_exists": METADATA.exists(),
        "runner_exists": RUNNER.exists(),
        "gate_exists": GATE.exists(),
        "shortlist_exists": SHORTLIST.exists(),
        "run_root_absent_before_launch": not RUN_ROOT.exists(),
        "log_root_absent_before_launch": not LOG_ROOT.exists(),
        **patches,
        "manifest_rows_4": len(manifest) == 4,
        "manifest_no_triples_except_control_ch102": not manifest[
            manifest["run_label"].ne("control_ch102")
        ]["cardinality"].ge(3).any(),
        "all_beta_3p75": manifest["beta_vae"].astype(float).eq(3.75).all(),
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    rows = pd.DataFrame([{"check": k, "pass": bool(v)} for k, v in checks.items()])
    text = f"""# Preflight status

Overall status: **{status}**

Branch observed: `{branch}`

Output root: `{RUN_ROOT}`

Log root: `{LOG_ROOT}`

No training was launched by this preflight.

## Checks

{rows.to_markdown(index=False)}

## Candidate manifest

{manifest.to_markdown(index=False)}

## Exact launch commands

Foreground:

```bash
bash {PACKAGE / 'guarded_launch_phase2_beta375.sh'} --foreground
```

tmux:

```bash
bash {PACKAGE / 'guarded_launch_phase2_beta375.sh'} --tmux
```

Dry-run manifest traversal only:

```bash
bash {PACKAGE / 'launch_phase2_beta375.sh'} --dry-run
```
"""
    (PACKAGE / "preflight_status.md").write_text(text, encoding="utf-8")


def main() -> None:
    PACKAGE.mkdir(parents=True, exist_ok=True)
    write_protocol_amendment()
    manifest = write_manifest()
    timing = parse_log_seconds()
    timing.to_csv(PACKAGE / "observed_fase1ab_candidate_timings.csv", index=False)
    write_launch_scripts()
    patches = script_contains_required_patches()
    write_resource_estimate(timing)
    write_preflight_status(manifest, patches)
    command_log = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "package": str(PACKAGE),
        "run_root": str(RUN_ROOT),
        "log_root": str(LOG_ROOT),
        "did_launch_training": False,
        "did_launch_triples": False,
        "did_modify_gate_criteria": False,
        "did_modify_prior_amendments": False,
        "files_written": sorted(p.name for p in PACKAGE.iterdir() if p.is_file()),
    }
    (PACKAGE / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")
    print(json.dumps(command_log, indent=2))


if __name__ == "__main__":
    main()
