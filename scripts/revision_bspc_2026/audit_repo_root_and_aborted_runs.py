#!/usr/bin/env python3
"""Audit suspicious repo-root files and known aborted AUC-sprint runs.

Read-only by default: this script reports cleanup candidates and writes suggested
commands, but never deletes files or run outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/auc_sprint_adni_only/repo_cleanup_audit"
DEFAULT_ABORTED_LAYERNORM = (
    PROJECT_ROOT
    / "results/revision_bspc_2026/auc_sprint_adni_only/adni_expanded_v4_beta25_ch4_1_0_layernorm"
)

SUSPICIOUS_ROOT_NAMES = {
    "PY",
    "SubjectID,",
    "SubjectID:",
    "]",
    "candidate_for_new_inference,",
    "check_dir",
    "checks",
    "cnt[.nii.gz]",
    "cols",
    "df[is_new_vs_v3]",
    "else:",
    "for",
    "from",
    "has_check_mat,",
    "has_check_png,",
    "has_check_txt,",
    "has_realign_dir,",
    "has_resultsAAL3_mat,",
    "has_resultsAAL3_txt,",
    "if",
    "import",
    "in_historical:",
    "in_martin59:",
    "in_ours_philips3:",
    "in_v3:",
    "is_new_vs_v3:",
    "ours_philips",
    "out",
    "realign_dir",
    "res_dir",
    "rows",
    "}",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--layernorm-path", type=Path, default=DEFAULT_ABORTED_LAYERNORM)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def prepare_output_dir(path: Path, overwrite: bool) -> Path:
    path = resolve(path)
    path.mkdir(parents=True, exist_ok=True)
    generated = [
        "untracked_weird_root_files.csv",
        "aborted_run_candidates.csv",
        "recommended_cleanup_commands.sh",
        "README.md",
        "audit_manifest.json",
    ]
    existing = [path / name for name in generated if (path / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{path} already contains audit outputs; pass --overwrite")
    if overwrite:
        for file_path in existing:
            file_path.unlink()
    return path


def git_status_untracked() -> List[str]:
    proc = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "-z"],
        cwd=PROJECT_ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
    )
    parts = proc.stdout.split(b"\0")
    paths: List[str] = []
    for item in parts:
        if not item:
            continue
        text = item.decode("utf-8", errors="replace")
        status = text[:2]
        path = text[3:]
        if status == "??":
            paths.append(path)
    return paths


def file_info(path: Path) -> Dict[str, Any]:
    try:
        st = path.lstat()
    except FileNotFoundError:
        return {"exists": False}
    return {
        "exists": True,
        "is_file": path.is_file(),
        "is_dir": path.is_dir(),
        "is_symlink": path.is_symlink(),
        "size_bytes": int(st.st_size),
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
        "realpath": str(path.resolve()) if path.exists() else "",
    }


def suspicious_reason(rel_path: str) -> str:
    name = Path(rel_path).name
    if name in SUSPICIOUS_ROOT_NAMES:
        return "known_weird_root_token_from_shell_fragment"
    if "/" not in rel_path and len(name) <= 24 and "." not in name and name not in {"README", "LICENSE"}:
        return "extensionless_untracked_root_file"
    if "/" not in rel_path and any(ch in name for ch in "[],:{}"):
        return "punctuated_untracked_root_file"
    return ""


def audit_untracked_root_files() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for rel_path in git_status_untracked():
        if "/" in rel_path:
            continue
        reason = suspicious_reason(rel_path)
        if not reason:
            continue
        abs_path = PROJECT_ROOT / rel_path
        info = file_info(abs_path)
        rows.append(
            {
                "path": rel_path,
                "reason": reason,
                "exists": info.get("exists"),
                "is_file": info.get("is_file"),
                "is_dir": info.get("is_dir"),
                "is_symlink": info.get("is_symlink"),
                "size_bytes": info.get("size_bytes"),
                "mtime_utc": info.get("mtime_utc"),
                "recommended_action": "review_then_rm_file" if info.get("is_file") else "review_manually",
                "safe_cleanup_command": f"rm -f -- {shlex.quote(rel_path)}" if info.get("is_file") else "",
            }
        )
    return sorted(rows, key=lambda r: str(r["path"]))


def dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            file_path = Path(root) / name
            try:
                total += file_path.stat().st_size
            except OSError:
                pass
    return total


def audit_aborted_layernorm(path: Path) -> List[Dict[str, Any]]:
    path = resolve(path)
    real = path.resolve() if path.exists() else path
    fold_dirs = sorted(real.glob("fold_*")) if real.exists() else []
    files = sorted(p for p in real.rglob("*") if p.is_file()) if real.exists() else []
    checkpoints = [p for p in files if p.suffix in {".pt", ".pth"} or "checkpoint" in p.name.lower()]
    logs = [p for p in files if p.suffix in {".log", ".txt"} or "log" in p.name.lower()]
    return [
        {
            "run_name": "adni_expanded_v4_beta25_ch4_1_0_layernorm",
            "path": str(path),
            "exists": path.exists(),
            "is_symlink": path.is_symlink(),
            "realpath": str(real),
            "realpath_exists": real.exists(),
            "n_fold_dirs": len(fold_dirs),
            "n_files": len(files),
            "n_checkpoints_or_checkpoint_like": len(checkpoints),
            "n_logs_or_text": len(logs),
            "size_bytes": dir_size_bytes(real),
            "status": "partial_or_aborted_candidate" if path.exists() and files else "not_present_or_empty",
            "recommended_action": "do_not_use_as_valid_run; keep until reviewed, then remove symlink and target if confirmed aborted",
            "safe_cleanup_command_symlink_only": f"unlink {shlex.quote(str(path.relative_to(PROJECT_ROOT)))}" if path.is_symlink() else "",
            "manual_confirm_target_cleanup_command": f"rm -rf -- {shlex.quote(str(real))}" if real.exists() else "",
        }
    ]


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    outdir = prepare_output_dir(args.output_dir, args.overwrite)
    weird_rows = audit_untracked_root_files()
    aborted_rows = audit_aborted_layernorm(args.layernorm_path)
    write_csv(outdir / "untracked_weird_root_files.csv", weird_rows)
    write_csv(outdir / "aborted_run_candidates.csv", aborted_rows)

    commands = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Review before running. This script was generated by a read-only audit.",
        "",
        "# Suspicious untracked repo-root files:",
    ]
    for row in weird_rows:
        if row.get("safe_cleanup_command"):
            commands.append(row["safe_cleanup_command"])
    commands.extend(
        [
            "",
            "# Aborted LayerNorm run cleanup is intentionally commented out.",
            "# Confirm this run is not needed before removing either symlink or target.",
        ]
    )
    for row in aborted_rows:
        if row.get("safe_cleanup_command_symlink_only"):
            commands.append(f"# {row['safe_cleanup_command_symlink_only']}")
        if row.get("manual_confirm_target_cleanup_command"):
            commands.append(f"# {row['manual_confirm_target_cleanup_command']}")
    (outdir / "recommended_cleanup_commands.sh").write_text("\n".join(commands) + "\n", encoding="utf-8")

    readme = [
        "# Repo Root And Aborted Runs Audit",
        "",
        "This audit is read-only. No files were deleted.",
        "",
        f"- Suspicious untracked repo-root files: {len(weird_rows)}",
        f"- Aborted/partial LayerNorm candidates: {sum(1 for r in aborted_rows if r['status'] == 'partial_or_aborted_candidate')}",
        "",
        "The LayerNorm path is treated as aborted and not valid for scientific comparison unless a separate run audit proves otherwise.",
        "",
        "Cleanup commands were written to `recommended_cleanup_commands.sh`. Commands for the aborted LayerNorm target are commented out by design.",
    ]
    (outdir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "output_dir": str(outdir),
        "no_deletion_performed": True,
        "audited_layernorm_path": str(resolve(args.layernorm_path)),
    }
    (outdir / "audit_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("\n".join(readme))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
