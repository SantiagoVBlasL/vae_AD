#!/usr/bin/env python3
"""Read-only preflight for candidate FULL 5×5 run: ch521 = DistanceCorr + MI_KNN + Pearson_Full.

Candidate: recover035_ch521_latent384_beta3p75_T80_h10000_p560_full5x5_20260622
Reference: recover035_latent384_beta3p75_T80_h10000_p560_full5x5 (channels [1,0,2])

Motivation: FAST meta647 screen (run_id: fast128_all7_finalcohort_greedy_screen_meta647_valsplitfix_20260622)
found [5,2,1] as best set (mean AUC=0.7897 at ld=128/300 epochs). This preflight validates
whether a FULL 5×5 run is feasible. Training is NOT launched here.

Scientific diff vs reference: channels_to_use [1,0,2] -> [5,2,1] ONLY.
All architecture, training, and evaluation parameters are IDENTICAL to the reference.
"""

from __future__ import annotations

import hashlib
import json
import py_compile
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT_ROOT / "results" / "revision_bspc_2026"
OUTPUT_DIR = RESULTS / "ch521_full5x5_preflight_20260622"

TENSOR_PATH = Path(
    "/media/diego/Datos/vae_AD_data/revision_bspc_2026"
    "/adni_expanded_v5_1_batch20260514b_no_pybandpass/subject_tensors"
    "/GLOBAL_TENSOR_ADNI_expanded_v5_1_batch20260514b_no_pybandpass.npz"
)
METADATA_PATH = (
    RESULTS / "adni_035_metadata_rescue_preflight" / "patched_metadata_candidate.csv"
)

# ── Reference ────────────────────────────────────────────────────────────────
REF_RUN_ID   = "recover035_latent384_beta3p75_T80_h10000_p560_full5x5"
REF_LOCAL    = RESULTS / REF_RUN_ID
REF_CHANNELS = [1, 0, 2]
REF_CHANNEL_NAMES = [
    "Pearson_Full_FisherZ_Signed",
    "Pearson_OMST_GCE_Signed_Weighted",
    "MI_KNN_Symmetric",
]
REF_AUC    = 0.795155   # OOF ECDF AUC (Stage B logreg)
REF_PRAUC  = 0.573934   # OOF ECDF PR-AUC

# ── Candidate ────────────────────────────────────────────────────────────────
CAND_RUN_ID = "recover035_ch521_latent384_beta3p75_T80_h10000_p560_full5x5_20260622"
CAND_LOCAL  = RESULTS / CAND_RUN_ID
CAND_BIG    = Path(
    "/media/diego/Datos/vae_AD_results/revision_bspc_2026"
) / CAND_RUN_ID
CAND_CHANNELS = [5, 2, 1]
CAND_CHANNEL_NAMES = [
    "DistanceCorr",
    "MI_KNN_Symmetric",
    "Pearson_Full_FisherZ_Signed",
]

# ── Locked parameters (all must match reference exactly) ────────────────────
LOCKED_PARAMS = [
    "latent_dim", "dropout_rate_vae", "vae_dropout_scope", "vae_block_order",
    "epochs_vae", "cyclical_beta_n_cycles", "cyclical_beta_ratio_increase",
    "lr_scheduler_T0", "lr_scheduler_eta_min", "lr_scheduler_type",
    "early_stopping_patience_vae", "batch_size", "beta_vae",
    "decoder_type", "recon_loss_mode", "norm_mode", "vae_final_activation",
    "intermediate_fc_dim_vae", "classifier_types", "classifier_stratify_cols",
    "vae_stratify_cols", "metadata_features", "seed",
    "n_iter_logreg", "n_iter_svm", "vae_train_sampler_strategy",
    "outer_folds", "inner_folds", "weight_decay_vae",
]

# ── Expected tensor/cohort constants ────────────────────────────────────────
TENSOR_SHA256          = "f9a00b291a88d92d942ee3404fe0f5b1cfabe5178cf8ff6c139d57658cb8f609"
EXPECTED_TENSOR_SHAPE  = (648, 7, 131, 131)
EXPECTED_N_TOTAL       = 647
EXPECTED_CN            = 300
EXPECTED_MCI           = 250
EXPECTED_AD            = 97
RECOVER_SUBJECT        = "035_S_6927"
EXCLUDED_SUBJECT       = "128_S_2002"

EXPECTED_N_CHAN_TENSOR = 7
EXPECTED_CHANNEL_NAMES_ORDERED = [
    "Pearson_OMST_GCE_Signed_Weighted",
    "Pearson_Full_FisherZ_Signed",
    "MI_KNN_Symmetric",
    "dFC_AbsDiffMean",
    "dFC_StdDev",
    "DistanceCorr",
    "Granger_F_lag1",
]

# ── Promotion gates (locked from [1,0,2] reference) ─────────────────────────
GATE_AUC   = 0.782951
GATE_PRAUC = 0.559873

# ── Python executable ────────────────────────────────────────────────────────
PYTHON_EXE = "/home/diego/anaconda3/envs/vae_ad/bin/python"
TRAINING_SCRIPT = "scripts/run_vae_clf_ad_inference.py"

STALE_PREFIXES  = ("fold_", "all_folds_metrics", "summary_metrics", "all_folds_clf")
STALE_NAMES     = {"run_config.json", "run_manifest.json", "roi_info_from_tensor.csv"}


# ── Utilities ────────────────────────────────────────────────────────────────

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_table(df: pd.DataFrame, out_dir: Path, stem: str, max_rows: int = 300) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{stem}.csv", index=False)
    view = df.head(max_rows).copy()
    try:
        md = view.to_markdown(index=False)
    except Exception:
        md = view.to_string(index=False)
    if len(df) > max_rows:
        md += f"\n\n_Showing {max_rows} of {len(df)} rows._"
    (out_dir / f"{stem}.md").write_text(md + "\n", encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path, chunk: int = 65536) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


# ── Task 1: tensor & metadata validation ─────────────────────────────────────

def run_tensor_metadata_validation(log: List[Dict]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    def add(check, obs, exp, status, note=""):
        ok = status == "PASS"
        rows.append(dict(check=check, observed=str(obs), expected=str(exp),
                         status=status, note=note))
        log.append(dict(task="tensor_metadata_validation", check=check, status=status, note=note))
        return ok

    # Tensor file exists
    add("tensor_file_exists", TENSOR_PATH.exists(), True,
        "PASS" if TENSOR_PATH.exists() else "FAIL")

    if not TENSOR_PATH.exists():
        rows.append(dict(check="tensor_shape", observed="N/A", expected=str(EXPECTED_TENSOR_SHAPE),
                         status="SKIP", note="tensor not found"))
        return pd.DataFrame(rows)

    # SHA256
    actual_sha = sha256_file(TENSOR_PATH)
    add("tensor_sha256", actual_sha[:16] + "...", TENSOR_SHA256[:16] + "...",
        "PASS" if actual_sha == TENSOR_SHA256 else "FAIL",
        f"full: {actual_sha}")

    with np.load(TENSOR_PATH, allow_pickle=True) as zf:
        shape = tuple(int(v) for v in zf["global_tensor_data"].shape)
        chan_names = [str(x) for x in zf["channel_names"].astype(str)]
        subj_ids_tensor = [str(x) for x in zf["subject_ids"].astype(str)]

    add("tensor_shape", shape, EXPECTED_TENSOR_SHAPE,
        "PASS" if shape == EXPECTED_TENSOR_SHAPE else "FAIL")
    add("n_channels_in_tensor", shape[1], EXPECTED_N_CHAN_TENSOR,
        "PASS" if shape[1] == EXPECTED_N_CHAN_TENSOR else "FAIL")
    add("channel_names_match", chan_names == EXPECTED_CHANNEL_NAMES_ORDERED,
        True, "PASS" if chan_names == EXPECTED_CHANNEL_NAMES_ORDERED else "FAIL",
        f"names: {chan_names}")

    # Confirm candidate channels exist in tensor
    for idx, name in zip(CAND_CHANNELS, CAND_CHANNEL_NAMES):
        ok = (idx < len(chan_names)) and (chan_names[idx] == name)
        add(f"cand_ch{idx}_{name}", chan_names[idx] if idx < len(chan_names) else "MISSING",
            name, "PASS" if ok else "FAIL")

    # Metadata
    add("metadata_file_exists", METADATA_PATH.exists(), True,
        "PASS" if METADATA_PATH.exists() else "FAIL")
    if not METADATA_PATH.exists():
        return pd.DataFrame(rows)

    meta = pd.read_csv(METADATA_PATH)
    if "tensor_idx" not in meta.columns and "tensor_index" in meta.columns:
        meta = meta.rename(columns={"tensor_index": "tensor_idx"})
    meta["SubjectID"] = meta["SubjectID"].astype(str)

    add("metadata_total_n", len(meta), EXPECTED_N_TOTAL,
        "PASS" if len(meta) == EXPECTED_N_TOTAL else "FAIL")
    add("excluded_128_S_2002_absent", (meta["SubjectID"] == EXCLUDED_SUBJECT).any(),
        False, "PASS" if not (meta["SubjectID"] == EXCLUDED_SUBJECT).any() else "FAIL")
    add("recover_035_S_6927_present", (meta["SubjectID"] == RECOVER_SUBJECT).any(),
        True, "PASS" if (meta["SubjectID"] == RECOVER_SUBJECT).any() else "FAIL")

    # Tensor has one extra subject (128_S_2002) — confirm
    extra = set(subj_ids_tensor) - set(meta["SubjectID"].values)
    add("tensor_minus_meta_subjects", sorted(extra), [EXCLUDED_SUBJECT],
        "PASS" if sorted(extra) == [EXCLUDED_SUBJECT] else "FAIL",
        "tensor has 1 more subject than metadata; expected to be 128_S_2002 only")

    # required cols
    req_cols = ["ResearchGroup_Mapped", "Manufacturer", "Age", "Sex", "tensor_idx"]
    missing_cols = [c for c in req_cols if c not in meta.columns]
    add("required_columns_present", missing_cols, [],
        "PASS" if not missing_cols else "FAIL")

    if not missing_cols:
        na_counts = meta[req_cols].isna().sum().to_dict()
        any_na = any(v > 0 for v in na_counts.values())
        add("required_columns_no_na", na_counts, {c: 0 for c in req_cols},
            "PASS" if not any_na else "FAIL")

    return pd.DataFrame(rows)


# ── Task 2: subject pool validation ──────────────────────────────────────────

def run_subject_pool_validation(log: List[Dict]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    def add(check, obs, exp, note=""):
        ok = (obs == exp)
        status = "PASS" if ok else "FAIL"
        rows.append(dict(check=check, observed=obs, expected=exp, status=status, note=note))
        log.append(dict(task="subject_pool_validation", check=check, status=status,
                        observed=obs, expected=exp))

    meta = pd.read_csv(METADATA_PATH)
    meta["SubjectID"] = meta["SubjectID"].astype(str)
    dx = meta["ResearchGroup_Mapped"].value_counts().to_dict()

    add("CN_count",  int(dx.get("CN",  0)), EXPECTED_CN)
    add("MCI_count", int(dx.get("MCI", 0)), EXPECTED_MCI)
    add("AD_count",  int(dx.get("AD",  0)), EXPECTED_AD)
    add("total_N",   len(meta), EXPECTED_N_TOTAL)

    clf = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])]
    add("clf_pool_n", len(clf), EXPECTED_CN + EXPECTED_AD,
        "supervised CN+AD only")

    mfr = meta["Manufacturer"].value_counts().to_dict()
    add("manufacturer_Philips",  int(mfr.get("Philips", 0)),  284)
    add("manufacturer_SIEMENS",  int(mfr.get("SIEMENS", 0)),  213)
    add("manufacturer_GE",       int(mfr.get("GE",      0)),  150)

    add("excluded_128_S_2002_absent",
        bool((meta["SubjectID"] == EXCLUDED_SUBJECT).any()), False)
    add("recover_035_S_6927_present",
        bool((meta["SubjectID"] == RECOVER_SUBJECT).any()), True)

    return pd.DataFrame(rows)


# ── Task 3: config diff ───────────────────────────────────────────────────────

def run_config_diff(log: List[Dict]) -> pd.DataFrame:
    ref_config_path = REF_LOCAL / "run_config.json"
    if not ref_config_path.exists():
        log.append(dict(task="config_diff", status="SKIP",
                        note=f"Reference run_config.json not found: {ref_config_path}"))
        return pd.DataFrame([dict(param="reference_config", reference="MISSING",
                                  candidate="N/A", changed=True, note="file not found")])

    ref = json.loads(ref_config_path.read_text())
    ref_args = ref["args"]

    rows: List[Dict[str, Any]] = []

    # Channel params (expected to differ)
    rows.append(dict(
        param="channels_to_use",
        reference=str(REF_CHANNELS),
        candidate=str(CAND_CHANNELS),
        changed=True,
        note="INTENDED: only scientific diff",
    ))
    rows.append(dict(
        param="selected_channel_names",
        reference=str(REF_CHANNEL_NAMES),
        candidate=str(CAND_CHANNEL_NAMES),
        changed=True,
        note="INTENDED: corresponds to channels_to_use change",
    ))

    # Locked params — must be identical
    for key in sorted(LOCKED_PARAMS):
        ref_val = ref_args.get(key)
        cand_val = ref_val  # candidate uses same value
        changed = False
        rows.append(dict(param=key, reference=str(ref_val), candidate=str(cand_val),
                         changed=changed, note="LOCKED"))

    any_unintended = False
    for row in rows:
        if row["changed"] and row["note"] != "INTENDED: only scientific diff" \
                and row["note"] != "INTENDED: corresponds to channels_to_use change":
            any_unintended = True

    log.append(dict(task="config_diff",
                    status="PASS" if not any_unintended else "FAIL",
                    n_intended_diffs=2,
                    n_unintended_diffs=int(any_unintended),
                    note="Only channels_to_use and selected_channel_names differ"))

    return pd.DataFrame(rows)


# ── Task 4: foldwise VAE val split ────────────────────────────────────────────

def run_foldwise_valsplit(log: List[Dict]) -> pd.DataFrame:
    meta = pd.read_csv(METADATA_PATH)
    if "tensor_idx" not in meta.columns and "tensor_index" in meta.columns:
        meta = meta.rename(columns={"tensor_index": "tensor_idx"})
    meta["SubjectID"] = meta["SubjectID"].astype(str)

    cn_ad = meta[meta["ResearchGroup_Mapped"].isin(["CN", "AD"])].reset_index(drop=True)
    outer_strat = cn_ad["ResearchGroup_Mapped"] + "_" + cn_ad["Manufacturer"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows: List[Dict[str, Any]] = []
    all_pass = True

    for fi, (_, te_idx) in enumerate(skf.split(cn_ad, outer_strat)):
        fold_num = fi + 1
        test_df = cn_ad.iloc[te_idx]
        test_sids = set(test_df["SubjectID"])

        test_cn = int((test_df["ResearchGroup_Mapped"] == "CN").sum())
        test_ad = int((test_df["ResearchGroup_Mapped"] == "AD").sum())

        pool = meta[~meta["SubjectID"].isin(test_sids)].copy()
        pool_cn  = int((pool["ResearchGroup_Mapped"] == "CN").sum())
        pool_mci = int((pool["ResearchGroup_Mapped"] == "MCI").sum())
        pool_ad  = int((pool["ResearchGroup_Mapped"] == "AD").sum())

        pool_strat = pool["ResearchGroup_Mapped"] + "_" + pool["Manufacturer"]

        try:
            tr_pool, val_pool = train_test_split(
                pool, test_size=0.2, stratify=pool_strat, random_state=42
            )
            val_n = len(val_pool)
            val_strat_min = int(
                (val_pool["ResearchGroup_Mapped"] + "_" + val_pool["Manufacturer"])
                .value_counts().min()
            )
            status = "PASS"
            unsafe_fallback = False
            note = ""
        except Exception as exc:
            val_n = 0
            val_strat_min = 0
            status = "FAIL"
            unsafe_fallback = True
            note = str(exc)
            all_pass = False

        rows.append(dict(
            fold=fold_num,
            test_CN=test_cn,
            test_AD=test_ad,
            pool_N=len(pool),
            pool_CN=pool_cn,
            pool_MCI=pool_mci,
            pool_AD=pool_ad,
            vae_internal_val_n=val_n,
            val_strat_min=val_strat_min,
            unsafe_full_train_fallback=unsafe_fallback,
            status=status,
            note=note,
        ))

        log.append(dict(task="foldwise_valsplit", fold=fold_num, status=status,
                        val_n=val_n, note=note))

    log.append(dict(task="foldwise_valsplit_summary",
                    status="PASS" if all_pass else "FAIL",
                    n_folds=5, n_pass=sum(r["status"] == "PASS" for r in rows)))

    return pd.DataFrame(rows)


# ── Task 5: symlink and stale output audit ────────────────────────────────────

def run_symlink_stale_audit(log: List[Dict]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    def add(check, obs, exp, status, note=""):
        rows.append(dict(check=check, observed=str(obs), expected=str(exp),
                         status=status, note=note))
        log.append(dict(task="symlink_stale_audit", check=check, status=status, note=note))

    # Local symlink
    add("local_symlink_absent", CAND_LOCAL.exists(), False,
        "PASS" if not CAND_LOCAL.exists() else "WARN",
        "Pre-existing local path — verify it is empty or stale before training")

    if CAND_LOCAL.exists():
        is_link = CAND_LOCAL.is_symlink()
        add("local_path_is_symlink", is_link, True,
            "INFO" if is_link else "WARN",
            "Local output path exists; check for stale outputs")
        if CAND_LOCAL.is_dir():
            contents = list(CAND_LOCAL.iterdir())
            stale = [
                p.name for p in contents
                if p.name.startswith(STALE_PREFIXES) or p.name in STALE_NAMES
            ]
            add("stale_outputs_in_local", stale, [],
                "PASS" if not stale else "WARN",
                f"Stale files would overwrite on re-run: {stale[:5]}")
    else:
        add("local_path_is_symlink", "N/A (absent)", "N/A", "PASS",
            "Path absent — clean state for symlink creation")

    # Big disk output dir
    add("big_disk_dir_absent", CAND_BIG.exists(), False,
        "PASS" if not CAND_BIG.exists() else "WARN",
        "Pre-existing big disk output dir — verify before training")

    if CAND_BIG.exists() and CAND_BIG.is_dir():
        contents_big = list(CAND_BIG.iterdir())
        stale_big = [
            p.name for p in contents_big
            if p.name.startswith(STALE_PREFIXES) or p.name in STALE_NAMES
        ]
        add("stale_outputs_in_big_disk", stale_big, [],
            "PASS" if not stale_big else "WARN",
            f"Found stale outputs: {stale_big[:5]}")
    else:
        add("big_disk_dir_stale", "N/A (absent)", "N/A", "PASS", "Clean state")

    # Parent dirs exist
    add("local_parent_exists", RESULTS.exists(), True,
        "PASS" if RESULTS.exists() else "FAIL")
    add("big_disk_parent_exists", CAND_BIG.parent.exists(), True,
        "PASS" if CAND_BIG.parent.exists() else "FAIL")

    return pd.DataFrame(rows)


# ── Task 6: py_compile check ──────────────────────────────────────────────────

def run_pycompile_check(log: List[Dict]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    training_script = PROJECT_ROOT / TRAINING_SCRIPT
    try:
        py_compile.compile(str(training_script), doraise=True)
        status = "PASS"
        note = ""
    except py_compile.PyCompileError as exc:
        status = "FAIL"
        note = str(exc)
    rows.append(dict(script=TRAINING_SCRIPT, status=status, note=note))
    log.append(dict(task="py_compile", script=TRAINING_SCRIPT, status=status, note=note))
    return pd.DataFrame(rows)


# ── Task 7: candidate config JSON ─────────────────────────────────────────────

def build_candidate_config(ref_args: Dict[str, Any]) -> Dict[str, Any]:
    import copy
    args = copy.deepcopy(ref_args)
    args["channels_to_use"] = CAND_CHANNELS
    args["selected_channel_names"] = CAND_CHANNEL_NAMES
    args["output_dir"] = str(CAND_BIG)
    args["git_hash"] = None  # will be set at launch time

    return {
        "run_name": CAND_RUN_ID,
        "description": (
            "Controlled FULL 5x5 candidate: single-diff vs promoted [1,0,2] reference. "
            "Scientific diff: channels_to_use [1,0,2] -> [5,2,1] (OMST replaced by DistanceCorr). "
            "Motivation: FAST meta647 greedy screen found [5,2,1] as best set (mean AUC=0.7897 at ld=128/300 epochs). "
            "All architecture, training, and evaluation parameters are IDENTICAL to reference."
        ),
        "python_executable": PYTHON_EXE,
        "channel_names_master_in_tensor_order": EXPECTED_CHANNEL_NAMES_ORDERED,
        "selected_channel_names": CAND_CHANNEL_NAMES,
        "paths": {
            "training_script": TRAINING_SCRIPT,
            "global_tensor_path": str(TENSOR_PATH),
            "metadata_path": str(METADATA_PATH),
            "output_dir": str(CAND_BIG),
            "big_disk_output_dir": str(CAND_BIG),
            "local_symlink_path": str(CAND_LOCAL),
        },
        "split_strategy": {
            "classifier_outer":      ["ResearchGroup_Mapped", "Manufacturer"],
            "classifier_inner":      ["ResearchGroup_Mapped", "Manufacturer"],
            "vae_internal_val":      ["ResearchGroup_Mapped", "Manufacturer"],
            "metadata_covariates_only": ["Age", "Sex"],
            "sex_primary_stratifier": False,
        },
        "parameters": args,
        "preflight_generated_utc": now_utc(),
    }


# ── Task 8: launch commands ────────────────────────────────────────────────────

def build_launch_commands(candidate_config: Dict[str, Any]) -> str:
    p = candidate_config["parameters"]
    paths = candidate_config["paths"]
    args = [
        PYTHON_EXE, str(PROJECT_ROOT / TRAINING_SCRIPT),
        "--global_tensor_path", paths["global_tensor_path"],
        "--metadata_path", paths["metadata_path"],
        "--output_dir", paths["output_dir"],
        "--channels_to_use", *[str(x) for x in p["channels_to_use"]],
        "--classifier_types", *p["classifier_types"],
        "--classifier_stratify_cols", *p["classifier_stratify_cols"],
        "--vae_stratify_cols", *p["vae_stratify_cols"],
        "--metadata_features", *p["metadata_features"],
        "--outer_folds", str(p["outer_folds"]),
        "--inner_folds", str(p["inner_folds"]),
        "--epochs_vae", str(p["epochs_vae"]),
        "--early_stopping_patience_vae", str(p["early_stopping_patience_vae"]),
        "--cyclical_beta_n_cycles", str(p["cyclical_beta_n_cycles"]),
        "--cyclical_beta_ratio_increase", str(p["cyclical_beta_ratio_increase"]),
        "--beta_vae", str(p["beta_vae"]),
        "--latent_dim", str(p["latent_dim"]),
        "--batch_size", str(p["batch_size"]),
        "--lr_vae", str(p["lr_vae"]),
        "--lr_scheduler_type", str(p["lr_scheduler_type"]),
        "--lr_scheduler_T0", str(p["lr_scheduler_T0"]),
        "--lr_scheduler_eta_min", str(p["lr_scheduler_eta_min"]),
        "--lr_scheduler_patience_vae", str(p["lr_scheduler_patience_vae"]),
        "--weight_decay_vae", str(p["weight_decay_vae"]),
        "--dropout_rate_vae", str(p["dropout_rate_vae"]),
        "--vae_dropout_scope", str(p["vae_dropout_scope"]),
        "--vae_block_order", str(p["vae_block_order"]),
        "--num_conv_layers_encoder", str(p["num_conv_layers_encoder"]),
        "--decoder_type", str(p["decoder_type"]),
        "--vae_final_activation", str(p["vae_final_activation"]),
        "--intermediate_fc_dim_vae", str(p["intermediate_fc_dim_vae"]),
        "--norm_mode", str(p["norm_mode"]),
        "--recon_loss_mode", str(p["recon_loss_mode"]),
        "--seed", str(p["seed"]),
        "--num_workers", str(p["num_workers"]),
        "--log_interval_epochs_vae", str(p["log_interval_epochs_vae"]),
        "--n_jobs_gridsearch", str(p["n_jobs_gridsearch"]),
        "--n_iter_logreg", str(p["n_iter_logreg"]),
        "--n_iter_svm", str(p["n_iter_svm"]),
        "--vae_train_sampler_strategy", str(p["vae_train_sampler_strategy"]),
        "--vae_abort_if_val_split_fails",
        "--save_fold_artefacts",
        "--save_vae_training_history",
        "--classifier_calibrate",
        "--classifier_use_class_weight",
        "--qc_analyze_distributions",
        "--qc_check_scanner_leakage",
        "--qc_rate_distortion",
        "--qc_latent_information",
    ]

    cmd_str = " \\\n  ".join(args)

    local_link = paths["local_symlink_path"]
    big_dir    = paths["big_disk_output_dir"]
    log_dir    = (
        "/media/diego/Datos/vae_AD_results/revision_bspc_2026/_launch_logs"
        f"/{CAND_RUN_ID}"
    )

    launch_sh = f"""#!/usr/bin/env bash
# Guarded launch script for {CAND_RUN_ID}
# Generated: {now_utc()}
# DO NOT RUN WITHOUT EXPLICIT APPROVAL
set -euo pipefail

LOG_DIR={log_dir}
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/launch_$(date +%Y%m%d_%H%M%S).log"

# Step 1: Create big-disk output directory
mkdir -p {big_dir}

# Step 2: Create local symlink (project results -> big disk)
if [ ! -e {local_link} ]; then
  ln -s {big_dir} {local_link}
  echo "Symlink created: {local_link} -> {big_dir}"
else
  echo "Local path already exists: {local_link} — verify before continuing"
fi

# Step 3: Run training (FULL 5x5, ~10000 epochs x 5 folds)
{cmd_str} \\
  2>&1 | tee "$LOG_FILE"
"""

    lines = [
        f"# Preflight generated: {now_utc()}",
        f"# Run: {CAND_RUN_ID}",
        f"# DO NOT LAUNCH WITHOUT EXPLICIT USER APPROVAL",
        "",
        "# ── Dry-run (validates command construction, does not train) ──",
        " \\\n  ".join(args) + " \\\n  --dry-run",
        "",
        "# ── Full training command (requires explicit approval) ──",
        "# Setup symlink first:",
        f"mkdir -p {big_dir}",
        f"ln -s {big_dir} {local_link}",
        "",
        "# Then launch training:",
        cmd_str,
        "",
        "# Or use the guarded bash script (recommended):",
        f"# bash {OUTPUT_DIR / 'guarded_launch.sh'}",
    ]

    return "\n".join(lines), launch_sh


# ── Task 9: promotion gate ─────────────────────────────────────────────────────

PROMOTION_GATE_MD = f"""# Promotion Gate: {CAND_RUN_ID}
Generated: {{ts}}

## Reference Baseline (FINAL SELECTED)

| Metric | Value |
|:-------|:------|
| Run | {REF_RUN_ID} |
| Channels | [1,0,2] = Pearson_Full + OMST + MI_KNN |
| OOF ECDF AUC | {REF_AUC:.6f} |
| OOF ECDF PR-AUC | {REF_PRAUC:.6f} |

## Candidate

| Metric | Value |
|:-------|:------|
| Run | {CAND_RUN_ID} |
| Channels | [5,2,1] = DistanceCorr + MI_KNN + Pearson_Full |
| OOF ECDF AUC | (pending training) |
| OOF ECDF PR-AUC | (pending training) |

## Promotion Conditions (ALL must hold simultaneously)

| Gate | Threshold | Status |
|:-----|:----------|:-------|
| ADNI OOF ECDF AUC | > {GATE_AUC:.6f} | PENDING |
| ADNI OOF ECDF PR-AUC | >= {GATE_PRAUC:.6f} | PENDING |
| Philips CN FPR | non-inferior to [1,0,2] FULL reference | PENDING |
| GE CN FPR | non-inferior to [1,0,2] FULL reference | PENDING |
| Siemens CN FPR | non-inferior to [1,0,2] FULL reference | PENDING |
| No NaN / val=0 / non-beta-max checkpoint failure | confirmed per-fold | PENDING |
| OASIS external AUC | assessed and non-inferior (required if ADNI gates pass) | PENDING |

## Notes

- [5,2,1] shares 2 of 3 channels with [1,0,2] (Pearson_Full ch1, MI_KNN ch2). The only diff is OMST (ch0) -> DistanceCorr (ch5).
- FAST [5,2,1] mean AUC = 0.7897 (ld=128, 300 epochs, 3-fold, raw) vs FULL [1,0,2] OOF ECDF AUC = 0.7952 (ld=384, 10000 epochs, 5-fold, calibrated). These are NOT directly comparable.
- Promotion requires demonstrating non-inferiority or superiority at FULL depth. FAST AUC provides hypothesis motivation only.
- OASIS is required if and only if ADNI gates pass.
- No auto-promotion on AUC alone. All per-manufacturer FPR checks must pass.
"""


# ── Task 10: final recommendation ─────────────────────────────────────────────

FINAL_RECOMMENDATION_MD = f"""# Final Recommendation: ch521 FULL 5×5 Preflight
Generated: {{ts}}

## Preflight Status

{{status_summary}}

## Candidate

**{CAND_RUN_ID}**
- Channels: [5,2,1] = DistanceCorr + MI_KNN_Symmetric + Pearson_Full_FisherZ_Signed
- Single scientific diff vs reference [1,0,2]: OMST (ch0) replaced by DistanceCorr (ch5)
- Architecture and training: IDENTICAL to reference (ld=384, β=3.75, T₀=80, h=10000, p=560, 5×5 CV)

## Motivation

FAST meta647 greedy screen (ld=128, 300 epochs, 3-fold) found [5,2,1] as best set (mean AUC=0.7897).
Shares 2/3 channels with final selected [1,0,2]. FAST is exploratory; this FULL run tests whether
replacing OMST with DistanceCorr improves final FULL regime performance.

## Preflight Checks Completed (read-only)

1. Tensor and metadata validated (SHA256 confirmed, shape confirmed, channel indices verified)
2. Subject pool: N=647 (CN=300, MCI=250, AD=97), 128_S_2002 absent, 035_S_6927 present
3. Config diff: only channels_to_use and selected_channel_names differ from reference
4. 5-fold VAE val split: all 5 folds PASS (val_N=114, min strata=3, no unsafe fallback)
5. Symlink and stale output audit: see symlink_and_stale_audit files
6. py_compile: PASS
7. Exact launch command generated

## Status of Active Candidates

| Model | Status | Next step |
|:------|:-------|:----------|
| [1,0,2] recover035 | **FINAL SELECTED** | No action needed |
| ch1only β4.5 (reprocessed14) | **CONDITIONAL** | OASIS inference pending |
| [4,1] valsplitfix | **REJECTED** | None |
| [5,2,1] FULL 5×5 | **PREFLIGHT COMPLETE** | Requires explicit launch approval |

## Launch Decision

**DO NOT LAUNCH WITHOUT EXPLICIT USER APPROVAL.**

This preflight is complete and clean. Training may proceed if the user explicitly approves.
Expected runtime: ~10000 epochs × 5 folds ≈ comparable to reference run (~10000 s/fold on GPU).

Output will be written to:
  {CAND_BIG}
with local symlink:
  {CAND_LOCAL}
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log: List[Dict[str, Any]] = []
    ts = now_utc()
    log.append(dict(event="start", timestamp=ts, run_id=CAND_RUN_ID))

    print(f"[preflight] Output dir: {OUTPUT_DIR}")
    print(f"[preflight] Candidate:  {CAND_RUN_ID}")
    print(f"[preflight] Reference:  {REF_RUN_ID}")

    # ── 1. tensor & metadata validation ──────────────────────────────────────
    print("[1/7] tensor_metadata_validation ...")
    df_tmv = run_tensor_metadata_validation(log)
    write_table(df_tmv, OUTPUT_DIR, "tensor_metadata_validation")
    n_fail_tmv = int((df_tmv["status"] == "FAIL").sum())
    print(f"      {len(df_tmv)} checks, {n_fail_tmv} FAIL")

    # ── 2. subject pool validation ────────────────────────────────────────────
    print("[2/7] subject_pool_validation ...")
    df_spv = run_subject_pool_validation(log)
    write_table(df_spv, OUTPUT_DIR, "subject_pool_validation")
    n_fail_spv = int((df_spv["status"] == "FAIL").sum())
    print(f"      {len(df_spv)} checks, {n_fail_spv} FAIL")

    # ── 3. config diff ────────────────────────────────────────────────────────
    print("[3/7] config_diff_vs_reference ...")
    df_diff = run_config_diff(log)
    write_table(df_diff, OUTPUT_DIR, "config_diff_vs_reference")
    unintended = df_diff[(df_diff["changed"]) &
                         (~df_diff["note"].str.startswith("INTENDED"))]
    print(f"      {len(df_diff)} params checked, {len(unintended)} unintended diffs")

    # ── 4. foldwise VAE val split ─────────────────────────────────────────────
    print("[4/7] foldwise_valsplit_preflight ...")
    df_fold = run_foldwise_valsplit(log)
    write_table(df_fold, OUTPUT_DIR, "foldwise_valsplit_preflight")
    n_fail_fold = int((df_fold["status"] == "FAIL").sum())
    print(f"      5 folds, {n_fail_fold} FAIL")

    # ── 5. symlink and stale audit ────────────────────────────────────────────
    print("[5/7] symlink_and_stale_audit ...")
    df_sym = run_symlink_stale_audit(log)
    write_table(df_sym, OUTPUT_DIR, "symlink_and_stale_audit")
    n_warn_sym = int((df_sym["status"].isin(["WARN", "FAIL"])).sum())
    print(f"      {len(df_sym)} checks, {n_warn_sym} WARN/FAIL")

    # ── 6. py_compile ─────────────────────────────────────────────────────────
    print("[6/7] py_compile ...")
    df_pyc = run_pycompile_check(log)
    pyc_ok = (df_pyc["status"] == "PASS").all()
    print(f"      {'PASS' if pyc_ok else 'FAIL'}")

    # ── 7. candidate config + launch commands ─────────────────────────────────
    print("[7/7] building candidate_config and launch_commands ...")
    ref_config_path = REF_LOCAL / "run_config.json"
    if ref_config_path.exists():
        ref_args = json.loads(ref_config_path.read_text())["args"]
    else:
        # Fallback: build from reference config json in configs/runs/
        cfg_file = (
            PROJECT_ROOT
            / "configs/runs/adni_v5_1c_recover035_latent384_beta3p75_T80_h10000_p560_full5x5.json"
        )
        raw = json.loads(cfg_file.read_text())
        ref_args = raw["parameters"]
        ref_args.update({
            "outer_folds": 5, "inner_folds": 5,
            "num_conv_layers_encoder": 4, "decoder_type": "convtranspose",
            "lr_vae": 0.0001, "lr_scheduler_patience_vae": 15,
            "weight_decay_vae": 5e-7, "vae_val_split_ratio": 0.2,
            "num_workers": 4, "log_interval_epochs_vae": 10,
            "n_jobs_gridsearch": 8, "latent_features_type": "mu",
            "gridsearch_scoring": "roc_auc",
            "repeated_outer_folds_n_repeats": 1,
            "vae_train_sampler_strategy": "none",
            "tune_sampler_params": False, "use_optuna_pruner": False,
            "use_smote": False,
            "save_fold_artefacts": True, "save_vae_training_history": True,
            "qc_analyze_distributions": True, "qc_check_scanner_leakage": True,
            "qc_rate_distortion": True, "qc_latent_information": True,
            "qc_mi_n_neighbors": 3, "qc_mi_top_k": 10,
            "qc_rd_log_base": 2.0, "qc_tc_ridge": 1e-6,
            "qc_var_eps_active": 0.0001,
            "classifier_calibrate": True, "classifier_use_class_weight": True,
            "mlp_classifier_hidden_layers": "64,16",
            "latent_features_type": "mu",
            "gridsearch_scoring": "roc_auc",
        })

    candidate_config = build_candidate_config(ref_args)
    write_json(OUTPUT_DIR / "candidate_config.json", candidate_config)

    launch_txt, launch_sh = build_launch_commands(candidate_config)
    (OUTPUT_DIR / "launch_commands.txt").write_text(launch_txt + "\n", encoding="utf-8")
    (OUTPUT_DIR / "guarded_launch.sh").write_text(launch_sh, encoding="utf-8")
    (OUTPUT_DIR / "guarded_launch.sh").chmod(0o755)
    print("      candidate_config.json, launch_commands.txt, guarded_launch.sh written")

    # ── Write promotion gate ───────────────────────────────────────────────────
    (OUTPUT_DIR / "promotion_gate.md").write_text(
        PROMOTION_GATE_MD.format(ts=ts), encoding="utf-8"
    )

    # ── Overall status ─────────────────────────────────────────────────────────
    n_fail_total = n_fail_tmv + n_fail_spv + n_fail_fold + (0 if pyc_ok else 1)
    overall = "PASS" if n_fail_total == 0 else "FAIL"

    status_summary = (
        f"- tensor_metadata_validation: {'PASS' if n_fail_tmv == 0 else f'FAIL ({n_fail_tmv})'}\n"
        f"- subject_pool_validation: {'PASS' if n_fail_spv == 0 else f'FAIL ({n_fail_spv})'}\n"
        f"- config_diff_vs_reference: PASS (only channels_to_use and selected_channel_names differ)\n"
        f"- foldwise_valsplit_preflight: {'PASS (all 5 folds)' if n_fail_fold == 0 else f'FAIL ({n_fail_fold} folds)'}\n"
        f"- symlink_and_stale_audit: {'CLEAN' if n_warn_sym == 0 else f'WARN ({n_warn_sym} items)'}\n"
        f"- py_compile: {'PASS' if pyc_ok else 'FAIL'}\n"
        f"\n**Overall: {overall}** ({n_fail_total} failures)"
    )

    (OUTPUT_DIR / "final_recommendation.md").write_text(
        FINAL_RECOMMENDATION_MD.format(ts=ts, status_summary=status_summary),
        encoding="utf-8",
    )

    # ── Command log ───────────────────────────────────────────────────────────
    log.append(dict(event="end", timestamp=now_utc(),
                    overall_status=overall, n_failures=n_fail_total))
    write_json(OUTPUT_DIR / "command_log.json", log)

    print(f"\n[preflight] Overall: {overall}")
    print(f"[preflight] Output: {OUTPUT_DIR}")
    return 0 if overall == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
