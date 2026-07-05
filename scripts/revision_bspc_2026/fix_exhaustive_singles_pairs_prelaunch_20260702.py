#!/usr/bin/env python3
"""Apply pre-launch corrections to Fase 1A+1B exhaustive singles/pairs package.

This script writes only small derived package files. It does not launch
training, create symlinks, or touch pilot/model outputs.
"""
from __future__ import annotations

import csv
import itertools
import json
import os
import stat
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path("/home/diego/proyectos/vae_AD")
PKG = ROOT / "results/revision_bspc_2026/post_revision_exploratory_20260630/fast_exhaustive_singles_pairs_preflight_20260702"
OUTPUT_ROOT = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026/post_revision_exploratory_20260630/fast_exhaustive_singles_pairs_20260702")
LOG_ROOT = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026/_launch_logs/post_revision_exploratory_20260630/fast_exhaustive_singles_pairs_20260702")
PILOT_BASE = Path("/media/diego/Datos/vae_AD_results/revision_bspc_2026/post_revision_exploratory_20260630/fast_channelmean_loss_ablation_beta_matched_20260702")
REUSE_SINGLE_1 = PILOT_BASE / "beta_cal_ch1_beta250"
PILOT_PAIR_5_1 = PILOT_BASE / "beta_cal_ch51_beta250"
TMPDIR = Path("/media/diego/Datos/tmp_vae")

CHANNEL_NAMES = {
    0: "Pearson_OMST_GCE_Signed_Weighted",
    1: "Pearson_Full_FisherZ_Signed",
    2: "MI_KNN_Symmetric",
    3: "dFC_AbsDiffMean",
    4: "dFC_StdDev",
    5: "DistanceCorr",
    6: "Granger_F_lag1",
}


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def has_metrics(path: Path) -> bool:
    return path.is_dir() and any(path.glob("all_folds_metrics_*.csv"))


def output_dir_for(chans: tuple[int, ...]) -> Path:
    if len(chans) == 1:
        return OUTPUT_ROOT / f"single_ch{chans[0]}"
    return OUTPUT_ROOT / f"pair_ch{chans[0]}_{chans[1]}"


def build_manifest() -> pd.DataFrame:
    rows = []
    cid = 1
    for r in (1, 2):
        for chans in itertools.combinations(range(7), r):
            status = "reuse" if chans == (1,) else "new"
            source = str(REUSE_SINGLE_1) if status == "reuse" else ""
            rows.append(
                {
                    "candidate_id": cid,
                    "channels": " ".join(map(str, chans)),
                    "channel_names": "+".join(CHANNEL_NAMES[c] for c in chans),
                    "cardinality": len(chans),
                    "status": status,
                    "source_dir_if_reused": source,
                    "output_dir": str(output_dir_for(chans)),
                    "run_label": f"single_ch{chans[0]}" if len(chans) == 1 else f"pair_ch{chans[0]}_{chans[1]}",
                }
            )
            cid += 1
    df = pd.DataFrame(rows)
    # Fail closed if generation drifts.
    pairs = df[df["cardinality"].eq(2)]["channels"].str.split().apply(lambda xs: tuple(map(int, xs)))
    assert len(df) == 28
    assert int(df["cardinality"].eq(1).sum()) == 7
    assert int(df["cardinality"].eq(2).sum()) == 21
    assert int(df["status"].eq("reuse").sum()) == 1
    assert int(df["status"].eq("new").sum()) == 27
    assert pairs.apply(lambda x: x[0] < x[1]).all()
    assert df["channels"].nunique() == 28
    return df


def launch_script() -> str:
    return f"""#!/usr/bin/env bash
# Manifest-driven Fase 1A + 1B exhaustive singles/pairs launch.
# Do not run directly unless the guarded wrapper has passed checks.
set -euo pipefail

DRY_RUN=0
if [[ "${{1:-}}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

PYTHON="/home/diego/anaconda3/envs/vae_ad/bin/python"
ABLATION_SCRIPT="/home/diego/proyectos/vae_AD/scripts/run_vae_clf_ad_ablation.py"
MANIFEST="{PKG}/planned_candidates_singles_pairs.csv"
TENSOR="/media/diego/Datos/vae_AD_data/revision_bspc_2026/adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors/GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
METADATA="/home/diego/proyectos/vae_AD/results/revision_bspc_2026/adni_035_metadata_rescue_preflight/patched_metadata_candidate.csv"
OUTPUT_ROOT="{OUTPUT_ROOT}"
LOG_ROOT="{LOG_ROOT}"
TMPDIR="{TMPDIR}"
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
  --beta_vae 2.50
  --vae_abort_if_val_split_fails
  --strict_metadata_intersection
  --classifier_use_class_weight
  --save_vae_training_history
  --vae_checkpoint_select_high_beta_only
)

n_manifest=0
n_new=0
n_reuse=0
n_skipped=0
n_run=0

echo "========================================================"
echo "Fase 1A + 1B exhaustive singles/pairs, manifest-driven"
echo "Output root: $OUTPUT_ROOT"
echo "Log root:    $LOG_ROOT"
echo "Dry-run:     $DRY_RUN"
echo "========================================================"

tail -n +2 "$MANIFEST" | while IFS=, read -r candidate_id channels channel_names cardinality status source_dir output_dir run_label; do
  n_manifest=$((n_manifest + 1))
  if [[ "$status" == "reuse" ]]; then
    n_reuse=$((n_reuse + 1))
    if has_metrics "$output_dir"; then
      echo "[REUSE] $run_label channels=[$channels] output symlink has metrics: $output_dir"
    elif [[ "$DRY_RUN" == "1" && -n "$source_dir" ]] && has_metrics "$source_dir"; then
      echo "[DRY-RUN][REUSE] $run_label channels=[$channels] source has metrics; launch wrapper will create symlink: $source_dir -> $output_dir"
    else
      echo "[ABORT] Reuse candidate $run_label lacks metrics at expected output path: $output_dir"
      exit 11
    fi
    continue
  fi

  n_new=$((n_new + 1))
  if has_metrics "$output_dir"; then
    n_skipped=$((n_skipped + 1))
    echo "[SKIP] $run_label channels=[$channels] metrics already present."
    continue
  fi

  read -r -a ch_arr <<< "$channels"
  log_file="${{LOG_ROOT}}/${{run_label}}_$(date +%Y%m%d_%H%M%S).log"
  echo "[PLAN] candidate $candidate_id / $run_label channels=[$channels] output=$output_dir"

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
  n_run=$((n_run + 1))
done

echo "========================================================"
echo "[DONE] Manifest traversal complete."
echo "Use find -L for completion counting:"
echo "  find -L '$OUTPUT_ROOT' -mindepth 2 -maxdepth 2 -type f -name 'all_folds_metrics_*.csv' | wc -l"
echo "Fase 1C remains deferred."
echo "========================================================"
"""


def guarded_script() -> str:
    return f"""#!/usr/bin/env bash
# Guarded wrapper for manifest-driven Fase 1A + 1B exhaustive singles/pairs.
set -euo pipefail

MODE="${{1:-}}"
if [[ "$MODE" != "--foreground" && "$MODE" != "--tmux" ]]; then
  echo "Usage: $0 --foreground | --tmux"
  exit 2
fi

PYTHON="/home/diego/anaconda3/envs/vae_ad/bin/python"
REPO_ROOT="/home/diego/proyectos/vae_AD"
SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
LAUNCH_SCRIPT="${{SCRIPT_DIR}}/launch_exhaustive_singles_pairs.sh"
MANIFEST="${{SCRIPT_DIR}}/planned_candidates_singles_pairs.csv"
OUTPUT_ROOT="{OUTPUT_ROOT}"
LOG_ROOT="{LOG_ROOT}"
PACKAGE_RUN_OUTPUTS="${{SCRIPT_DIR}}/run_outputs"
PACKAGE_RUN_LOGS="${{SCRIPT_DIR}}/run_logs"
TMPDIR="{TMPDIR}"
SESSION="exhaustive_singles_pairs_20260702"
PILOT_SINGLE="{REUSE_SINGLE_1}"
EXPECTED_SINGLE_LINK="${{OUTPUT_ROOT}}/single_ch1"
STDOUT_LOG="${{OUTPUT_ROOT}}/guarded_launch_stdout.log"

echo "========================================================"
echo "Guarded launch: Fase 1A + 1B exhaustive singles/pairs"
echo "  Corrected prelaunch amendment: 002a"
echo "  Candidates: 28 total, 27 new, 1 reused"
echo "  Pair order: canonical ascending i<j"
echo "  Mode: $MODE"
echo "========================================================"

fail() {{
  echo "[ABORT] $*" >&2
  exit 1
}}

assert_media_path() {{
  local p="$1"
  [[ "$p" == /media/diego/Datos/* ]] || fail "Path is not under /media/diego/Datos: $p"
}}

has_metrics() {{
  local dir="$1"
  local found=""
  if [[ -d "$dir" || -L "$dir" ]]; then
    found="$(find -L "$dir" -maxdepth 1 -type f -name 'all_folds_metrics_*.csv' -print -quit 2>/dev/null || true)"
  fi
  [[ -n "$found" ]]
}}

ensure_symlink() {{
  local link="$1"
  local target="$2"
  if [[ ! -e "$target" && ! -d "$target" ]]; then
    fail "Symlink target does not exist: $target"
  fi
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

BRANCH="$(git -C "$REPO_ROOT" branch --show-current)"
[[ "$BRANCH" == "exploratory/post-revision-20260630" ]] || fail "Expected branch exploratory/post-revision-20260630, got $BRANCH"
echo "[OK] Branch: $BRANCH"

assert_media_path "$OUTPUT_ROOT"
assert_media_path "$LOG_ROOT"
assert_media_path "$TMPDIR"
echo "[OK] Output/log/tmp roots are on /media/diego/Datos"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$TMPDIR"
tmp_test="$(mktemp "$TMPDIR/fast_exhaustive_tmp.XXXXXX")" || fail "TMPDIR is not writable: $TMPDIR"
rm -f "$tmp_test"
export TMPDIR
echo "[OK] TMPDIR writable and exported: $TMPDIR"

"$PYTHON" -m py_compile "$REPO_ROOT/scripts/run_vae_clf_ad_ablation.py" || fail "py_compile failed for ablation runner"
bash -n "$LAUNCH_SCRIPT" || fail "bash -n failed for launch script"
bash -n "$0" || fail "bash -n failed for guarded wrapper"
echo "[OK] Syntax checks passed"

[[ -f "$SCRIPT_DIR/protocol_amendment_002a_prelaunch_corrections.md" ]] || fail "Missing protocol_amendment_002a_prelaunch_corrections.md"
[[ -f "$MANIFEST" ]] || fail "Missing manifest: $MANIFEST"

python_manifest_check="$("$PYTHON" - "$MANIFEST" <<'PY'
import pandas as pd, sys
df = pd.read_csv(sys.argv[1])
pairs = df[df.cardinality.eq(2)].channels.str.split().apply(lambda x: tuple(map(int,x)))
checks = [
    len(df)==28,
    df.cardinality.eq(1).sum()==7,
    df.cardinality.eq(2).sum()==21,
    df.channels.nunique()==28,
    all(a<b for a,b in pairs),
    df.status.eq('reuse').sum()==1,
    df.status.eq('new').sum()==27,
    df.loc[df.status.eq('reuse'),'channels'].tolist()==['1'],
]
print('PASS' if all(checks) else 'FAIL')
PY
)"
[[ "$python_manifest_check" == "PASS" ]] || fail "Manifest validation failed"
echo "[OK] Manifest validation passed"

has_metrics "$PILOT_SINGLE" || fail "Reusable single [1] missing metrics: $PILOT_SINGLE"
echo "[OK] Reusable single [1] metrics found"
echo "[INFO] Reversed-order pilot [5,1] is not reused as canonical [1,5]."

ensure_symlink "$EXPECTED_SINGLE_LINK" "$PILOT_SINGLE"
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


def resource_text() -> str:
    return f"""# Corrected Resource Estimate — Fase 1A + 1B Exhaustive Screen

## Correction

The canonical pair `[1,5]` is no longer reused from the beta pilot result trained as `--channels_to_use 5 1`. Channel order is part of the CNN input representation, so `[5,1]` is preserved only as an external sensitivity result.

## Candidate Counts

- Total candidates: 28
- New candidates: 27
- Reused candidates: 1 (`single [1]`)
- Singles: 7 total, 6 new, 1 reused
- Pairs: 21 total, 21 new

## Timing Estimate

Observed beta=2.50 pair time from the pilot: `1143.94 s`.

Using that observed pair time as the conservative per-candidate runtime:

- 27 new candidates x 1143.94 s = 30,886.38 s = 8.58 h
- Adding small launch/logging overhead gives the corrected planning estimate: **approximately 8.78 h**

This is still an estimate. The launcher is sequential, resume-safe, and skips any candidate with an existing `all_folds_metrics_*.csv`.

## Monitoring

Use `find -L`, not fragile glob expansion:

```bash
find -L {OUTPUT_ROOT} -mindepth 2 -maxdepth 2 -type f -name 'all_folds_metrics_*.csv' | wc -l
```

Expected completed metric files after full launch: 28 if the `single_ch1` symlink is included, 27 newly produced candidate outputs plus one reused linked output.
"""


def amendment_text() -> str:
    return f"""# Protocol Amendment 002a — Prelaunch Corrections For Fase 1A + 1B

Date: 2026-07-02

## Status

No Fase 1A/1B exhaustive singles/pairs training had been launched before this correction. The proposed output root was absent at correction time:

`{OUTPUT_ROOT}`

## Channel-Order Correction

The beta pilot pair was trained as:

`--channels_to_use 5 1`

The exhaustive manifest defines canonical unordered pair `[1,5]` as ascending channel order:

`--channels_to_use 1 5`

These are not equivalent for a fixed-seed CNN input pipeline because input-channel order is part of the actual tensor representation seen by the encoder. Therefore the reversed-order pilot result `[5,1]` must not be reused as the canonical exhaustive candidate `[1,5]`.

## Reuse Policy

Only the exact single-channel candidate is reused:

- Reuse: `single [1] = beta_cal_ch1_beta250`
- Do not reuse as canonical candidate: pilot pair `[5,1] = beta_cal_ch51_beta250`

The `[5,1]` pilot result remains unchanged and may be reported only as an external sensitivity result, not as the canonical exhaustive pair `[1,5]`.

## Manifest Policy

The corrected manifest is generated from:

- `itertools.combinations(range(7), 1)`
- `itertools.combinations(range(7), 2)`

All pair channel orders are canonical ascending order (`i < j`). The corrected totals are:

- 28 total candidates
- 27 new candidates
- 1 reused candidate
- 7 singles
- 21 pairs

## Fase 1C

Fase 1C remains deferred. No triple candidate is launched by these scripts.
"""


def final_validation_text(results: dict[str, str]) -> str:
    rows = "\n".join(f"| {k} | {v} |" for k, v in results.items())
    return f"""# Final Prelaunch Validation

| Check | Status |
|:--|:--|
{rows}

## Guardrails

- No training launched.
- No inference launched.
- No triples launched.
- Pilot outputs were not modified.
- Prior protocol amendment files were not modified.
- Manuscript, tensor, and metadata files were not modified.
- Launch scripts are manifest-driven and sequential.
- Foreground and tmux modes are supported without send-keys fallback.
"""


def main() -> None:
    started = datetime.now(timezone.utc)
    PKG.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest()
    manifest_path = PKG / "planned_candidates_singles_pairs.csv"
    manifest.to_csv(manifest_path, index=False, quoting=csv.QUOTE_MINIMAL)

    write(PKG / "protocol_amendment_002a_prelaunch_corrections.md", amendment_text())
    write(PKG / "resource_estimate.md", resource_text())
    write(PKG / "launch_exhaustive_singles_pairs.sh", launch_script())
    write(PKG / "guarded_launch_exhaustive_singles_pairs.sh", guarded_script())
    for script in ["launch_exhaustive_singles_pairs.sh", "guarded_launch_exhaustive_singles_pairs.sh"]:
        p = PKG / script
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)

    results: dict[str, str] = {}
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip()
    results["branch exploratory/post-revision-20260630"] = "PASS" if branch == "exploratory/post-revision-20260630" else f"FAIL ({branch})"
    results["manifest has 28 candidates"] = "PASS" if len(manifest) == 28 else "FAIL"
    results["manifest has 7 singles"] = "PASS" if int(manifest.cardinality.eq(1).sum()) == 7 else "FAIL"
    results["manifest has 21 pairs"] = "PASS" if int(manifest.cardinality.eq(2).sum()) == 21 else "FAIL"
    results["manifest has 28 unique channel sets"] = "PASS" if manifest.channels.nunique() == 28 else "FAIL"
    pair_ok = all(tuple(map(int, x.split()))[0] < tuple(map(int, x.split()))[1] for x in manifest.loc[manifest.cardinality.eq(2), "channels"])
    results["all pair indices ascending"] = "PASS" if pair_ok else "FAIL"
    results["exactly one reuse"] = "PASS" if int(manifest.status.eq("reuse").sum()) == 1 else "FAIL"
    results["exactly 27 new candidates"] = "PASS" if int(manifest.status.eq("new").sum()) == 27 else "FAIL"
    results["reuse is only single [1]"] = "PASS" if manifest.loc[manifest.status.eq("reuse"), "channels"].tolist() == ["1"] else "FAIL"
    results["output root under /media/diego/Datos"] = "PASS" if str(OUTPUT_ROOT).startswith("/media/diego/Datos/") else "FAIL"
    results["log root under /media/diego/Datos"] = "PASS" if str(LOG_ROOT).startswith("/media/diego/Datos/") else "FAIL"
    results["output root absent before launch"] = "PASS" if not OUTPUT_ROOT.exists() else "CHECK_EXISTS"
    results["single [1] reuse metrics exist"] = "PASS" if has_metrics(REUSE_SINGLE_1) else "FAIL"
    results["pilot [5,1] preserved external only"] = "PASS" if has_metrics(PILOT_PAIR_5_1) else "CHECK_MISSING"

    write(PKG / "final_prelaunch_validation.md", final_validation_text(results))

    command_log = {
        "experiment_name": "fast_exhaustive_singles_pairs_prelaunch_correction_002a_20260702",
        "generated_utc": started.isoformat(),
        "mode": "prelaunch_correction_only",
        "training_launched": False,
        "inference_launched": False,
        "branch": branch,
        "candidate_counts": {
            "total": 28,
            "new_to_run": 27,
            "reused_from_pilot": 1,
            "singles": 7,
            "pairs": 21,
        },
        "correction": {
            "single_ch1_reused": str(REUSE_SINGLE_1),
            "pair_5_1_pilot_not_reused_as_1_5": str(PILOT_PAIR_5_1),
            "canonical_pair_1_5_status": "new",
        },
        "output_root": str(OUTPUT_ROOT),
        "log_root": str(LOG_ROOT),
        "tmpdir": str(TMPDIR),
        "validation": results,
        "commands_to_run_after_generation": [
            "bash -n launch_exhaustive_singles_pairs.sh",
            "bash -n guarded_launch_exhaustive_singles_pairs.sh",
            "/home/diego/anaconda3/envs/vae_ad/bin/python -m py_compile scripts/run_vae_clf_ad_ablation.py",
            "bash launch_exhaustive_singles_pairs.sh --dry-run",
        ],
        "outputs": [
            "protocol_amendment_002a_prelaunch_corrections.md",
            "planned_candidates_singles_pairs.csv",
            "resource_estimate.md",
            "launch_exhaustive_singles_pairs.sh",
            "guarded_launch_exhaustive_singles_pairs.sh",
            "final_prelaunch_validation.md",
            "command_log.json",
        ],
        "guardrails": {
            "no_training": True,
            "no_inference": True,
            "no_triples": True,
            "no_pilot_output_modification": True,
            "no_manuscript_tensor_metadata_edits": True,
        },
    }
    (PKG / "command_log.json").write_text(json.dumps(command_log, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
