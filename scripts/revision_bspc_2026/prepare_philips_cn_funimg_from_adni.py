#!/usr/bin/env python3
"""
Prepare Philips-CN FunImg inputs from downloaded ADNI DICOM folders.

This is the Philips counterpart to the GE/Siemens expansion prep flow. It is
non-destructive: it refuses to overwrite real files and never launches DPARSF.
Use --dry-run first to write a status CSV without conversion/linking.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_ROOT = Path("/home/diego/proyectos/external")

DEFAULT_SUBJECTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "RevisionPaperfMRI_2026_04_pilot"
    / "batches"
    / "philips_cn_stress_test6.txt"
)
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "data" / "RevisionPaperfMRI_2026_04" / "ADNI"
DEFAULT_STAGING_ROOT = EXTERNAL_ROOT / "adni_bridge" / "expansion"
DEFAULT_DICM2NII_ROOT = EXTERNAL_ROOT / "dicm2nii"
DEFAULT_MATLAB_BIN = Path("/usr/local/bin/matlab")
DEFAULT_STATUS_CSV = (
    PROJECT_ROOT
    / "results"
    / "revision_bspc_2026"
    / "philips_cn_stress_test_selection"
    / "prepare_philips_funimg_status.csv"
)
DEFAULT_DOWNLOAD_CSVS = [
    PROJECT_ROOT / "data" / "adni_download_now.csv",
    PROJECT_ROOT / "data" / "RevisionPaperfMRI_2026_04_4_06_2026.csv",
    PROJECT_ROOT / "data" / "RevisionPaperfMRI_2026_04_4_06_2026_SecondBatch.csv",
]

MANUFACTURER_FOLDER = "Philips"

SUBJECT_COLUMNS = ("SubjectID", "Subject", "Subject ID", "subject_id", "subject")
IMAGE_COLUMNS = ("ImageID", "Image Data ID", "Image ID", "ImageID ")
CSV_FIELDS = [
    "SubjectID",
    "ImageID",
    "DicomDir",
    "RawNiftiDir",
    "FunImgDir",
    "n_dicom",
    "n_raw_nifti",
    "n_funimg_nifti",
    "Status",
    "Message",
]


@dataclass
class Validation:
    ok: bool
    message: str
    n_nifti: int


def log(message: str) -> None:
    print(message, flush=True)


def fail(message: str, code: int = 1) -> None:
    print(f"[ERROR] {message}", file=sys.stderr, flush=True)
    raise SystemExit(code)


def strip_image_id(value: object) -> str:
    text = str(value).strip()
    if not text:
        return ""
    text = text.split(".")[0]
    if text.upper().startswith("I") and text[1:].isdigit():
        return text[1:]
    return text


def image_dir_names(image_id: str) -> list[str]:
    image_id = strip_image_id(image_id)
    if not image_id:
        return []
    return [f"I{image_id}", image_id]


def dedupe_preserving_order(values: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        value = str(value).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def load_subjects(subjects_file: Path, limit: Optional[int]) -> list[str]:
    if not subjects_file.exists():
        fail(f"No existe --subjects-file: {subjects_file}")
    subjects = []
    for line in subjects_file.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        subjects.append(line.split()[0])
    subjects = dedupe_preserving_order(subjects)
    if limit is not None:
        subjects = subjects[:limit]
    if not subjects:
        fail(f"No subjects found in {subjects_file}")
    return subjects


def detect_dialect(path: Path, text: str) -> csv.Dialect:
    if path.suffix.lower() == ".tsv":
        return csv.excel_tab
    sample = "\n".join(text.splitlines()[:20])
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;")
    except csv.Error:
        return csv.excel


def load_image_id_map(paths: list[Path]) -> dict[str, str]:
    image_ids: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8-sig")
        if not text.strip():
            continue
        dialect = detect_dialect(path, text)
        reader = csv.DictReader(text.splitlines(), dialect=dialect)
        if not reader.fieldnames:
            continue
        subject_col = next((c for c in SUBJECT_COLUMNS if c in reader.fieldnames), None)
        image_col = next((c for c in IMAGE_COLUMNS if c in reader.fieldnames), None)
        if not subject_col or not image_col:
            continue
        for row in reader:
            sid = str(row.get(subject_col, "")).strip()
            image_id = strip_image_id(row.get(image_col, ""))
            if sid and image_id and sid not in image_ids:
                image_ids[sid] = image_id
    return image_ids


def nifti_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    files = []
    for child in directory.iterdir():
        if child.is_file() or child.is_symlink():
            name = child.name.lower()
            if name.endswith(".nii") or name.endswith(".nii.gz"):
                files.append(child)
    return sorted(files)


def count_dicoms(directory: Optional[Path]) -> int:
    if directory is None or not directory.exists():
        return 0
    return sum(1 for path in directory.rglob("*.dcm") if path.is_file())


def validate_nifti_dir(directory: Path) -> Validation:
    files = nifti_files(directory)
    if not files:
        return Validation(False, "No NIfTI files found", 0)

    try:
        import nibabel as nib
    except ImportError:
        return Validation(True, "NIfTI files exist; nibabel unavailable, shape skipped", len(files))

    shapes = []
    errors = []
    for path in files:
        try:
            img = nib.load(str(path))
            shape = tuple(int(x) for x in img.shape)
            shapes.append(shape)
            if len(shape) >= 4 and shape[3] > 1:
                return Validation(True, f"4D NIfTI OK: {path.name} shape={shape}", len(files))
        except Exception as exc:  # report validation issue but keep batch going
            errors.append(f"{path.name}: {exc}")

    if len(files) > 1 and shapes:
        spatial = [shape[:3] for shape in shapes if len(shape) >= 3]
        if len(spatial) == len(files) and len(set(spatial)) == 1:
            return Validation(True, f"Compatible 3D frame series OK: {len(files)} files shape={spatial[0]}", len(files))

    details = "; ".join(errors[:3]) if errors else f"shapes={shapes[:5]}"
    return Validation(False, f"NIfTI is not 4D and not a compatible 3D frame series: {details}", len(files))


def find_dicom_dir_from_image_id(subject_root: Path, image_id: str) -> Optional[Path]:
    if not image_id or not subject_root.exists():
        return None
    wanted = set(image_dir_names(image_id))
    for path in subject_root.rglob("*"):
        if path.is_dir() and path.name in wanted and count_dicoms(path) > 0:
            return path
    return None


def infer_dicom_dir(subject_root: Path) -> tuple[str, Optional[Path]]:
    if not subject_root.exists():
        return "", None

    candidates: list[tuple[int, Path]] = []
    for path in subject_root.rglob("*"):
        if path.is_dir():
            n_dicom = count_dicoms(path)
            if n_dicom > 0:
                candidates.append((n_dicom, path))
    if not candidates:
        return "", None
    candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    best = candidates[0][1]
    return strip_image_id(best.name), best


def matlab_escape(path: Path) -> str:
    return str(path).replace("'", "''")


def write_matlab_conversion_script(
    subject_id: str,
    dicom_dir: Path,
    raw_nifti_dir: Path,
    dicm2nii_root: Path,
    work_dir: Path,
) -> Path:
    script_path = work_dir / f"convert_{subject_id}.m"
    script = f"""try
    addpath(genpath('{matlab_escape(dicm2nii_root)}'));
    dicomDir = '{matlab_escape(dicom_dir)}';
    niftiDir = '{matlab_escape(raw_nifti_dir)}';
    if exist(niftiDir, 'dir') ~= 7
        mkdir(niftiDir);
    end
    fprintf('dicm2nii input: %s\\n', dicomDir);
    fprintf('dicm2nii output: %s\\n', niftiDir);
    dicm2nii(dicomDir, niftiDir, '.nii.gz');
    exit(0);
catch ME
    fprintf(2, 'dicm2nii failed for {subject_id}: %s\\n', ME.message);
    try
        fprintf(2, '%s\\n', getReport(ME, 'extended'));
    catch
    end
    exit(1);
end
"""
    script_path.write_text(script, encoding="utf-8")
    return script_path


def run_matlab_conversion(
    subject_id: str,
    dicom_dir: Path,
    raw_nifti_dir: Path,
    work_dir: Path,
    dicm2nii_root: Path,
    matlab_bin: Path,
) -> tuple[bool, str]:
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_nifti_dir.mkdir(parents=True, exist_ok=True)
    script_path = write_matlab_conversion_script(
        subject_id=subject_id,
        dicom_dir=dicom_dir,
        raw_nifti_dir=raw_nifti_dir,
        dicm2nii_root=dicm2nii_root,
        work_dir=work_dir,
    )
    log_path = work_dir / "matlab_dicm2nii.log"
    cmd = [
        str(matlab_bin),
        "-nodisplay",
        "-nosplash",
        "-r",
        f"run('{matlab_escape(script_path)}')",
    ]
    with log_path.open("w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        return False, f"MATLAB dicm2nii failed with code {proc.returncode}; log={log_path}"
    return True, f"MATLAB dicm2nii completed; log={log_path}"


def safe_symlink(src: Path, dst: Path, dry_run: bool) -> tuple[bool, str]:
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink():
            current = dst.resolve(strict=False)
            target = src.resolve(strict=False)
            if current == target:
                return True, f"symlink already ok: {dst.name}"
            return False, f"existing symlink points elsewhere; refusing to modify: {dst}"
        return False, f"refusing to overwrite real file: {dst}"

    if dry_run:
        return True, f"would create symlink: {dst.name}"
    os.symlink(src, dst)
    return True, f"created symlink: {dst.name}"


def link_funimg(raw_nifti_dir: Path, funimg_dir: Path, dry_run: bool) -> tuple[bool, str]:
    files = nifti_files(raw_nifti_dir)
    header = raw_nifti_dir / "dcmHeaders.mat"
    link_sources = list(files)
    if header.exists():
        link_sources.append(header)
    if not link_sources:
        return False, "no files to link"

    messages = [f"would ensure directory: {funimg_dir}" if dry_run else f"ensured directory: {funimg_dir}"]
    if not dry_run:
        funimg_dir.mkdir(parents=True, exist_ok=True)

    for src in link_sources:
        ok, msg = safe_symlink(src, funimg_dir / src.name, dry_run=dry_run)
        messages.append(msg)
        if not ok:
            return False, "; ".join(messages)

    validation = validate_nifti_dir(raw_nifti_dir if dry_run else funimg_dir)
    if not validation.ok:
        return False, "; ".join(messages + [validation.message])
    return True, "; ".join(messages + [validation.message])


def status_row(
    subject_id: str,
    image_id: str,
    dicom_dir: Optional[Path],
    raw_nifti_dir: Path,
    funimg_dir: Path,
    status: str,
    message: str,
) -> dict[str, str]:
    raw_validation = validate_nifti_dir(raw_nifti_dir)
    fun_validation = validate_nifti_dir(funimg_dir)
    return {
        "SubjectID": subject_id,
        "ImageID": image_id,
        "DicomDir": str(dicom_dir) if dicom_dir else "",
        "RawNiftiDir": str(raw_nifti_dir),
        "FunImgDir": str(funimg_dir),
        "n_dicom": str(count_dicoms(dicom_dir)),
        "n_raw_nifti": str(raw_validation.n_nifti),
        "n_funimg_nifti": str(fun_validation.n_nifti),
        "Status": status,
        "Message": message,
    }


def process_subject(
    subject_id: str,
    image_id_map: dict[str, str],
    source_root: Path,
    staging_root: Path,
    dicm2nii_root: Path,
    matlab_bin: Path,
    convert: bool,
    link: bool,
    dry_run: bool,
) -> dict[str, str]:
    subject_root = source_root / subject_id
    image_id = image_id_map.get(subject_id, "")
    dicom_dir = find_dicom_dir_from_image_id(subject_root, image_id)
    if dicom_dir is None:
        inferred_image_id, inferred_dir = infer_dicom_dir(subject_root)
        image_id = image_id or inferred_image_id
        dicom_dir = inferred_dir

    raw_nifti_dir = staging_root / "raw_nifti" / MANUFACTURER_FOLDER / subject_id / "func"
    work_dir = staging_root / "conversion_work" / MANUFACTURER_FOLDER / subject_id
    funimg_dir = staging_root / "dparsf_single" / MANUFACTURER_FOLDER / "FunImg" / subject_id

    fun_validation = validate_nifti_dir(funimg_dir)
    if fun_validation.ok:
        return status_row(
            subject_id,
            image_id,
            dicom_dir,
            raw_nifti_dir,
            funimg_dir,
            "ALREADY_READY",
            fun_validation.message,
        )

    raw_validation = validate_nifti_dir(raw_nifti_dir)
    if raw_validation.ok:
        if link:
            ok, msg = link_funimg(raw_nifti_dir, funimg_dir, dry_run=dry_run)
            return status_row(
                subject_id,
                image_id,
                dicom_dir,
                raw_nifti_dir,
                funimg_dir,
                "DRY_RUN_WOULD_LINK" if dry_run and ok else "LINKED_READY" if ok else "LINK_FAILED",
                msg,
            )
        return status_row(
            subject_id,
            image_id,
            dicom_dir,
            raw_nifti_dir,
            funimg_dir,
            "RAW_NIFTI_READY",
            raw_validation.message + "; pass --link to create FunImg symlinks",
        )

    n_dicom = count_dicoms(dicom_dir)
    if dicom_dir is None or n_dicom == 0:
        return status_row(
            subject_id,
            image_id,
            dicom_dir,
            raw_nifti_dir,
            funimg_dir,
            "MISSING_DICOM",
            f"No DICOM directory found under {subject_root}",
        )

    if dry_run:
        return status_row(
            subject_id,
            image_id,
            dicom_dir,
            raw_nifti_dir,
            funimg_dir,
            "DRY_RUN_WOULD_CONVERT",
            "DICOM found; would run MATLAB dicm2nii and then link FunImg",
        )

    if not convert:
        return status_row(
            subject_id,
            image_id,
            dicom_dir,
            raw_nifti_dir,
            funimg_dir,
            "WOULD_CONVERT",
            "DICOM found; pass --convert to run MATLAB dicm2nii",
        )

    if not dicm2nii_root.exists():
        return status_row(
            subject_id,
            image_id,
            dicom_dir,
            raw_nifti_dir,
            funimg_dir,
            "CONVERSION_FAILED",
            f"dicm2nii root does not exist: {dicm2nii_root}",
        )
    if not matlab_bin.exists() and shutil.which(str(matlab_bin)) is None:
        return status_row(
            subject_id,
            image_id,
            dicom_dir,
            raw_nifti_dir,
            funimg_dir,
            "CONVERSION_FAILED",
            f"MATLAB binary not found: {matlab_bin}",
        )

    ok, convert_msg = run_matlab_conversion(
        subject_id=subject_id,
        dicom_dir=dicom_dir,
        raw_nifti_dir=raw_nifti_dir,
        work_dir=work_dir,
        dicm2nii_root=dicm2nii_root,
        matlab_bin=matlab_bin,
    )
    if not ok:
        return status_row(
            subject_id,
            image_id,
            dicom_dir,
            raw_nifti_dir,
            funimg_dir,
            "CONVERSION_FAILED",
            convert_msg,
        )

    post_validation = validate_nifti_dir(raw_nifti_dir)
    if not post_validation.ok:
        return status_row(
            subject_id,
            image_id,
            dicom_dir,
            raw_nifti_dir,
            funimg_dir,
            "INVALID_NIFTI",
            convert_msg + "; " + post_validation.message,
        )

    ok, link_msg = link_funimg(raw_nifti_dir, funimg_dir, dry_run=False)
    return status_row(
        subject_id,
        image_id,
        dicom_dir,
        raw_nifti_dir,
        funimg_dir,
        "CONVERTED_READY" if ok else "LINK_FAILED",
        convert_msg + "; " + link_msg,
    )


def write_status_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Philips/FunImg/<SubjectID> inputs for DPARSF from ADNI DICOM using MATLAB dicm2nii.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--subjects-file", type=Path, default=DEFAULT_SUBJECTS_FILE)
    parser.add_argument("--limit", type=int, default=2, help="Use only the first N subjects from --subjects-file.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING_ROOT)
    parser.add_argument("--dicm2nii-root", type=Path, default=DEFAULT_DICM2NII_ROOT)
    parser.add_argument("--matlab-bin", type=Path, default=DEFAULT_MATLAB_BIN)
    parser.add_argument("--download-csv", type=Path, action="append", default=None)
    parser.add_argument("--status-csv", type=Path, default=DEFAULT_STATUS_CSV)
    parser.add_argument("--convert", action="store_true", help="Run MATLAB dicm2nii for missing raw NIfTIs.")
    parser.add_argument("--link", action="store_true", help="Create FunImg symlinks from valid raw NIfTIs.")
    parser.add_argument("--dry-run", action="store_true", help="Do not convert or link; only discover and write status CSV.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subjects = load_subjects(args.subjects_file, args.limit)
    download_csvs = args.download_csv if args.download_csv else DEFAULT_DOWNLOAD_CSVS
    image_id_map = load_image_id_map(download_csvs)
    dry_run = bool(args.dry_run or (not args.convert and not args.link))
    link = bool(args.link or args.convert)

    log(f"[INFO] Subjects: {len(subjects)}")
    log(f"[INFO] Manufacturer folder: {MANUFACTURER_FOLDER}")
    log(f"[INFO] Source root: {args.source_root}")
    log(f"[INFO] Raw NIfTI root: {args.staging_root / 'raw_nifti' / MANUFACTURER_FOLDER}")
    log(f"[INFO] DPARSF FunImg root: {args.staging_root / 'dparsf_single' / MANUFACTURER_FOLDER / 'FunImg'}")
    log(f"[INFO] Mode: {'dry-run/scan-only' if dry_run else 'convert/link' if args.convert else 'link-only'}")

    rows = []
    for subject_id in subjects:
        row = process_subject(
            subject_id=subject_id,
            image_id_map=image_id_map,
            source_root=args.source_root,
            staging_root=args.staging_root,
            dicm2nii_root=args.dicm2nii_root,
            matlab_bin=args.matlab_bin,
            convert=args.convert,
            link=link,
            dry_run=dry_run,
        )
        rows.append(row)
        log(f"[{row['Status']}] {subject_id}: {row['Message']}")

    write_status_csv(rows, args.status_csv)
    log(f"[INFO] Status CSV: {args.status_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
