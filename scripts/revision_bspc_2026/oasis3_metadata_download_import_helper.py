#!/usr/bin/env python3
"""Safe OASIS-3 metadata download/import helper.

This helper is deliberately metadata-first. It does not know or guess NITRC/XNAT
API endpoints. It can import local metadata files, or download explicitly
provided metadata URLs, while blocking imaging payloads by default.

Credentials are accepted only from environment variables or an interactive
prompt and are never written to manifests/logs.
"""

from __future__ import annotations

import argparse
import csv
import getpass
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OASIS_DATA_ROOT = Path("/media/diego/Datos/vae_AD_data/OASIS3")
DEFAULT_METADATA_RAW = DEFAULT_OASIS_DATA_ROOT / "metadata_raw"
DEFAULT_LOCAL_SYMLINK = PROJECT_ROOT / "data/oasis3"
DEFAULT_AUDIT_OUTPUT_DIR = PROJECT_ROOT / "results/revision_bspc_2026/oasis3_feasibility_audit"
AUDIT_SCRIPT = PROJECT_ROOT / "scripts/revision_bspc_2026/oasis3_feasibility_audit.py"

METADATA_EXTENSIONS = {".csv", ".tsv", ".json", ".txt", ".xml", ".xlsx", ".xls"}
ARCHIVE_EXTENSIONS = {".zip", ".tar", ".tgz", ".gz", ".bz2", ".7z", ".rar"}
BLOCKED_SUFFIXES = {
    ".nii",
    ".nii.gz",
    ".dcm",
    ".dicom",
    ".ima",
    ".mgz",
    ".mgh",
    ".img",
    ".hdr",
    ".par",
    ".rec",
    ".bval",
    ".bvec",
}
BLOCKED_PATH_TOKENS = {
    "freesurfer",
    "pet",
    "dicom",
    "nifti",
    "nii",
    "anat",
    "func/sub-",
    "dwi",
    "meg",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safe metadata-only OASIS-3 download/import helper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metadata-root", type=Path, default=DEFAULT_METADATA_RAW)
    parser.add_argument("--oasis-data-root", type=Path, default=DEFAULT_OASIS_DATA_ROOT)
    parser.add_argument("--local-symlink", type=Path, default=DEFAULT_LOCAL_SYMLINK)
    parser.add_argument("--url-manifest", type=Path, default=None, help="Optional CSV/JSON with metadata URLs to download.")
    parser.add_argument("--import-from", type=Path, nargs="*", default=[], help="Local files/directories to import metadata from.")
    parser.add_argument("--download", action="store_true", help="Actually download URLs from --url-manifest.")
    parser.add_argument("--prompt-credentials", action="store_true", help="Prompt for NITRC_USER/NITRC_PASS if not in env.")
    parser.add_argument("--prepare-symlink", action="store_true", help="Create/fix local data/oasis3 symlink when safe.")
    parser.add_argument("--run-audit", action="store_true", help="Run oasis3_feasibility_audit.py after metadata import/download.")
    parser.add_argument("--allow-metadata-archives", action="store_true", help="Allow small metadata archives; imaging archives remain blocked.")
    parser.add_argument("--max-file-mb", type=float, default=50.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--python-executable", default=sys.executable)
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_hash() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(PROJECT_ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else ""
    except Exception:
        return ""


def suffixes_lower(path_or_name: str) -> List[str]:
    path = Path(path_or_name)
    suffixes = [s.lower() for s in path.suffixes]
    if path_or_name.lower().endswith(".nii.gz") and ".nii.gz" not in suffixes:
        suffixes.append(".nii.gz")
    return suffixes


def is_blocked_payload(name_or_url: str) -> Tuple[bool, str]:
    parsed = urllib.parse.urlparse(name_or_url)
    path_text = urllib.parse.unquote(parsed.path or name_or_url).lower()
    suffixes = suffixes_lower(path_text)
    if any(suffix in BLOCKED_SUFFIXES for suffix in suffixes):
        return True, "blocked_imaging_extension"
    for token in BLOCKED_PATH_TOKENS:
        if token in path_text and not path_text.endswith(".json"):
            return True, f"blocked_path_token:{token}"
    return False, ""


def is_metadata_payload(name_or_url: str, allow_archives: bool) -> Tuple[bool, str]:
    blocked, reason = is_blocked_payload(name_or_url)
    if blocked:
        return False, reason
    parsed = urllib.parse.urlparse(name_or_url)
    path_text = urllib.parse.unquote(parsed.path or name_or_url)
    suffixes = suffixes_lower(path_text)
    if any(suffix in METADATA_EXTENSIONS for suffix in suffixes):
        return True, "metadata_extension_allowed"
    if allow_archives and any(suffix in ARCHIVE_EXTENSIONS for suffix in suffixes):
        lower = path_text.lower()
        if any(token in lower for token in ["metadata", "clinical", "data_files", "participants", "sessions", "csv"]):
            return True, "metadata_archive_allowed_by_name"
        return False, "archive_name_not_clearly_metadata"
    return False, "not_metadata_extension"


def safe_url_for_manifest(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def read_url_manifest(path: Optional[Path]) -> List[Dict[str, str]]:
    if path is None:
        return []
    path = resolve(path)
    if not path.exists():
        raise FileNotFoundError(f"URL manifest not found: {path}")
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("urls", [])
        rows = []
        for item in data:
            if isinstance(item, str):
                rows.append({"url": item, "filename": ""})
            elif isinstance(item, dict):
                rows.append({"url": str(item.get("url", "")), "filename": str(item.get("filename", ""))})
        return rows
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames and "url" in reader.fieldnames:
            for row in reader:
                rows.append({"url": str(row.get("url", "")), "filename": str(row.get("filename", ""))})
        else:
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    rows.append({"url": line, "filename": ""})
    return rows


def iter_metadata_files(paths: Sequence[Path], allow_archives: bool) -> List[Path]:
    out: List[Path] = []
    for raw_path in paths:
        path = resolve(raw_path)
        if not path.exists():
            continue
        if path.is_file():
            ok, _ = is_metadata_payload(str(path), allow_archives)
            if ok:
                out.append(path)
        elif path.is_dir():
            for child in path.rglob("*"):
                if not child.is_file():
                    continue
                ok, _ = is_metadata_payload(str(child), allow_archives)
                if ok:
                    out.append(child)
    return unique_paths(out)


def unique_paths(paths: Iterable[Path]) -> List[Path]:
    seen = set()
    out = []
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def metadata_inventory(metadata_root: Path) -> List[Dict[str, Any]]:
    if not metadata_root.exists():
        return []
    rows = []
    for path in sorted(metadata_root.rglob("*")):
        if not path.is_file():
            continue
        blocked, blocked_reason = is_blocked_payload(str(path))
        ok, metadata_reason = is_metadata_payload(str(path), allow_archives=True)
        stat = path.stat()
        rows.append(
            {
                "path": str(path),
                "relative_path": str(path.relative_to(metadata_root)),
                "size_bytes": stat.st_size,
                "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                "is_metadata_candidate": ok and not blocked,
                "blocked_reason": blocked_reason,
                "metadata_reason": metadata_reason,
            }
        )
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def get_credentials(prompt: bool, needed: bool) -> Tuple[Optional[str], Optional[str], str]:
    if not needed:
        return None, None, "not_needed"
    user = os.environ.get("NITRC_USER")
    password = os.environ.get("NITRC_PASS")
    source = []
    if user:
        source.append("NITRC_USER")
    if password:
        source.append("NITRC_PASS")
    if (not user or not password) and prompt:
        if not user:
            user = input("NITRC username: ").strip()
            source.append("interactive_user")
        if not password:
            password = getpass.getpass("NITRC password: ")
            source.append("interactive_password")
    if not user or not password:
        return None, None, "missing"
    return user, password, "+".join(source)


def destination_for_download(metadata_root: Path, url: str, filename: str) -> Path:
    if filename:
        name = Path(filename).name
    else:
        parsed = urllib.parse.urlparse(url)
        name = Path(urllib.parse.unquote(parsed.path)).name or "downloaded_metadata"
    return metadata_root / "downloads" / name


def download_one(url: str, dest: Path, user: Optional[str], password: Optional[str], max_file_mb: float, overwrite: bool) -> Dict[str, Any]:
    if dest.exists() and not overwrite:
        return {"status": "exists_skipped", "path": str(dest), "size_bytes": dest.stat().st_size}
    password_manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
    if user and password:
        parsed = urllib.parse.urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        password_manager.add_password(None, base_url, user, password)
    opener = urllib.request.build_opener(urllib.request.HTTPBasicAuthHandler(password_manager))
    request = urllib.request.Request(url, headers={"User-Agent": "vae_AD_oasis3_metadata_helper/1.0"})
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    bytes_written = 0
    max_bytes = int(max_file_mb * 1024 * 1024)
    try:
        with opener.open(request, timeout=120) as response, tmp.open("wb") as f:
            content_length = response.headers.get("Content-Length")
            if content_length and int(content_length) > max_bytes:
                return {"status": "blocked_content_length_gt_max", "path": str(dest), "content_length": content_length}
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    tmp.unlink(missing_ok=True)
                    return {"status": "blocked_download_exceeded_max", "path": str(dest), "size_bytes": bytes_written}
                f.write(chunk)
        tmp.replace(dest)
        return {"status": "downloaded", "path": str(dest), "size_bytes": bytes_written}
    except urllib.error.URLError as exc:
        tmp.unlink(missing_ok=True)
        return {"status": "failed_download", "path": str(dest), "error": str(exc)}


def copy_metadata_file(src: Path, metadata_root: Path, overwrite: bool) -> Dict[str, Any]:
    dest = metadata_root / "imported" / src.name
    if dest.exists() and not overwrite:
        return {"status": "exists_skipped", "source": str(src), "path": str(dest), "size_bytes": dest.stat().st_size}
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    return {"status": "imported", "source": str(src), "path": str(dest), "size_bytes": dest.stat().st_size}


def is_effectively_empty_local_oasis(path: Path) -> Tuple[bool, str]:
    if not path.exists() and not path.is_symlink():
        return True, "local_path_missing"
    if path.is_symlink():
        return True, "local_path_is_symlink"
    if not path.is_dir():
        return False, "local_path_exists_not_directory"
    entries = list(path.iterdir())
    if not entries:
        return True, "local_directory_empty"
    if len(entries) == 1 and entries[0].name == "manifests" and entries[0].is_dir() and not any(entries[0].iterdir()):
        return True, "local_directory_only_empty_manifests"
    return False, "local_directory_has_existing_files"


def symlink_plan(local_symlink: Path, oasis_data_root: Path) -> Dict[str, Any]:
    local_symlink = resolve(local_symlink)
    oasis_data_root = oasis_data_root
    safe, reason = is_effectively_empty_local_oasis(local_symlink)
    current_target = str(local_symlink.resolve()) if local_symlink.is_symlink() else ""
    target_matches = bool(local_symlink.is_symlink() and local_symlink.resolve() == oasis_data_root.resolve())
    if target_matches:
        action = "none_symlink_already_correct"
    elif safe:
        action = "can_create_symlink"
    else:
        action = "manual_review_required"
    return {
        "local_symlink": str(local_symlink),
        "target": str(oasis_data_root),
        "safe_to_prepare": safe,
        "reason": reason,
        "current_target": current_target,
        "target_matches": target_matches,
        "planned_action": action,
    }


def prepare_symlink(local_symlink: Path, oasis_data_root: Path, dry_run: bool) -> Dict[str, Any]:
    plan = symlink_plan(local_symlink, oasis_data_root)
    local_symlink = resolve(local_symlink)
    if dry_run or plan["planned_action"] != "can_create_symlink":
        return plan
    if local_symlink.exists() and not local_symlink.is_symlink():
        backup = local_symlink.with_name(local_symlink.name + f".pre_oasis3_symlink_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        local_symlink.rename(backup)
        plan["backup_path"] = str(backup)
    local_symlink.parent.mkdir(parents=True, exist_ok=True)
    local_symlink.symlink_to(oasis_data_root)
    plan["created"] = True
    return plan


def write_manual_checklist(path: Path, metadata_root: Path) -> None:
    lines = [
        "# OASIS-3 Metadata-Only Manual Download Checklist",
        "",
        "Use this if scripted NITRC/XNAT download URLs are not known. Do not download imaging payloads yet.",
        "",
        "1. Log in to NITRC in a browser using your OASIS-3 account.",
        "2. Open NITRC-IR / OASIS-3.",
        "3. Export subject/session metadata CSVs from XNAT if available.",
        "4. Download OASIS3_data_files clinical/cognitive CSVs if available.",
        "5. If a BIDS view/export exists, download only metadata files:",
        "   - `participants.tsv`",
        "   - `sessions.tsv` or `sub-*/sub-*_sessions.tsv`",
        "   - `*_task-rest_*_bold.json` sidecars",
        "6. Avoid `.nii`, `.nii.gz`, DICOM, FreeSurfer, PET, and large archives for now.",
        f"7. Put downloaded metadata under `{metadata_root}`.",
        "8. Rerun:",
        "```bash",
        "/home/diego/anaconda3/envs/vae_ad/bin/python scripts/revision_bspc_2026/oasis3_feasibility_audit.py --oasis-root /media/diego/Datos/vae_AD_data/OASIS3 --bids-root /media/diego/Datos/vae_AD_data/OASIS3 --overwrite",
        "```",
        "",
        "Notes:",
        "- Do not save NITRC username/password in any file.",
        "- If using browser downloads, inspect file sizes before downloading.",
        "- If only a large archive is offered, first check whether it contains only metadata; otherwise do not download it for this step.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_readme(path: Path, metadata_root: Path, symlink_info: Dict[str, Any], dry_run: bool, downloaded: List[Dict[str, Any]], imported: List[Dict[str, Any]], audit_command: Sequence[str]) -> None:
    lines = [
        "# OASIS-3 Metadata Download/Import Helper",
        "",
        f"Mode: `{'DRY_RUN' if dry_run else 'REAL'}`",
        "",
        "No imaging downloads, preprocessing, or training are performed by this helper.",
        "",
        "## Directories",
        f"- Metadata raw directory: `{metadata_root}`",
        f"- Local symlink: `{symlink_info.get('local_symlink')}`",
        f"- Symlink target: `{symlink_info.get('target')}`",
        f"- Symlink planned action: `{symlink_info.get('planned_action')}`",
        f"- Symlink safety reason: `{symlink_info.get('reason')}`",
        "",
        "## Credential Handling",
        "- Credentials are read only from `NITRC_USER` / `NITRC_PASS` or interactive prompt.",
        "- Credentials are not written to manifests, command files, or logs.",
        "",
        "## Download/Import Results",
        f"- Download rows processed: `{len(downloaded)}`",
        f"- Import rows processed: `{len(imported)}`",
        "",
        "## Feasibility Audit Command",
        "```bash",
        " ".join(audit_command),
        "```",
        "",
        "## Next Step",
        "If metadata files are still absent, follow `manual_download_checklist.md` and rerun this helper or the feasibility audit.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_audit_command(args: argparse.Namespace, oasis_data_root: Path) -> List[str]:
    return [
        args.python_executable,
        str(AUDIT_SCRIPT),
        "--oasis-root",
        str(oasis_data_root),
        "--bids-root",
        str(oasis_data_root),
        "--output-dir",
        str(DEFAULT_AUDIT_OUTPUT_DIR),
        "--overwrite",
    ]


def main() -> int:
    args = parse_args()
    metadata_root = resolve(args.metadata_root)
    oasis_data_root = resolve(args.oasis_data_root)
    local_symlink = resolve(args.local_symlink)

    metadata_root.mkdir(parents=True, exist_ok=True)
    (metadata_root / "manifests").mkdir(parents=True, exist_ok=True)
    downloaded: List[Dict[str, Any]] = []
    imported: List[Dict[str, Any]] = []
    warnings: List[str] = []

    url_rows = read_url_manifest(args.url_manifest)
    symlink_info = prepare_symlink(local_symlink, oasis_data_root, args.dry_run or not args.prepare_symlink)

    import_candidates = iter_metadata_files(args.import_from, args.allow_metadata_archives)
    for src in import_candidates:
        if args.dry_run:
            imported.append({"status": "dry_run_would_import", "source": str(src), "path": str(metadata_root / "imported" / src.name), "size_bytes": src.stat().st_size})
        else:
            imported.append(copy_metadata_file(src, metadata_root, args.overwrite))

    download_needed = bool(url_rows and args.download and not args.dry_run)
    user, password, credential_status = get_credentials(args.prompt_credentials, download_needed)
    if url_rows and args.download and credential_status == "missing":
        warnings.append("download_requested_but_credentials_missing")

    for row in url_rows:
        url = row.get("url", "").strip()
        filename = row.get("filename", "").strip()
        if not url:
            continue
        name_for_check = filename or url
        ok, reason = is_metadata_payload(name_for_check, args.allow_metadata_archives)
        dest = destination_for_download(metadata_root, url, filename)
        record = {
            "status": "pending",
            "url_host_path": safe_url_for_manifest(url),
            "filename": filename or dest.name,
            "path": str(dest),
            "metadata_policy_reason": reason,
        }
        if not ok:
            record["status"] = "blocked_by_metadata_policy"
            downloaded.append(record)
            continue
        if args.dry_run or not args.download:
            record["status"] = "dry_run_would_download" if args.dry_run else "download_not_requested"
            downloaded.append(record)
            continue
        if credential_status == "missing":
            record["status"] = "blocked_missing_credentials"
            downloaded.append(record)
            continue
        result = download_one(url, dest, user, password, args.max_file_mb, args.overwrite)
        record.update(result)
        downloaded.append(record)

    inventory = metadata_inventory(metadata_root)
    write_csv(
        metadata_root / "metadata_file_inventory.csv",
        inventory,
        [
            "path",
            "relative_path",
            "size_bytes",
            "modified_utc",
            "is_metadata_candidate",
            "blocked_reason",
            "metadata_reason",
        ],
    )
    write_manual_checklist(metadata_root / "manual_download_checklist.md", metadata_root)

    audit_command = build_audit_command(args, oasis_data_root)
    write_readme(metadata_root / "README_download_next_steps.md", metadata_root, symlink_info, args.dry_run, downloaded, imported, audit_command)

    manifest = {
        "timestamp_utc": utc_now(),
        "git_hash": git_hash(),
        "dry_run": bool(args.dry_run),
        "metadata_root": str(metadata_root),
        "oasis_data_root": str(oasis_data_root),
        "local_symlink": str(local_symlink),
        "symlink_info": symlink_info,
        "url_manifest": str(resolve(args.url_manifest)) if args.url_manifest else "",
        "download_requested": bool(args.download),
        "import_from": [str(resolve(path)) for path in args.import_from],
        "credential_status": credential_status,
        "credential_values_written": False,
        "downloaded": downloaded,
        "imported": imported,
        "warnings": warnings,
        "blocked_extensions": sorted(BLOCKED_SUFFIXES),
        "metadata_extensions_allowed": sorted(METADATA_EXTENSIONS),
        "no_imaging_download": True,
        "no_preprocessing": True,
        "no_training": True,
        "audit_command": audit_command,
    }
    (metadata_root / "metadata_download_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.run_audit and not args.dry_run:
        subprocess.run(audit_command, cwd=str(PROJECT_ROOT), check=False)

    print((metadata_root / "README_download_next_steps.md").read_text(encoding="utf-8"))
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"- {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
