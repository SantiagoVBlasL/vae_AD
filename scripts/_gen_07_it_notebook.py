"""
Generator for notebooks/07_it_information_theory.ipynb  (v4 — final closing pass)
Run once: python scripts/_gen_07_it_notebook.py

v4 changes (from v3):
  RUN_MODE system: "full_recompute" raises RuntimeError if dependencies missing;
    "artifact_review" loads cached tables/figures and labels them explicitly.
  Dependency preflight cell: checks python/numpy/scipy/pandas/sklearn/torch/optuna/betavae_xai.
  torch.manual_seed(SEED) added; all random ops seeded.
  Output dir clean/legacy logic: superseded files moved to _legacy/ in full_recompute.
  §10 narrative corrected: MI_KNN_Symmetric is significant (p_bonf≈0.006, r_rb≈-0.264,
    CN > AD); §10 figure suptitle and §11 summary now data-driven.
  Figure quality: 300 dpi PNG + PDF vector export.
  LaTeX table export helper: key results tables exported as .tex.
  Manifest v4: per-section status = recomputed / loaded_from_cache / skipped / exploratory.

v3 changes (from v2):
  ROOT CAUSE FIX: test_indices.npy and train_dev_indices.npy are LOCAL pool indices,
    NOT global tensor indices. v2 used them to index GLOBAL_TENSOR, producing wrong
    subjects for all re-inference-dependent analyses (§4, §5, §6, §8).
    v3 uses test_tensor_idx.npy / train_dev_tensor_idx.npy (global tensor indices).
    pipeline_log is used to build SubjectID→tensor_idx for MCI subjects.

  §4  Re-inference now encodes the correct subjects (global tensor idx)
  §5  MI computed on correctly-aligned latent-label pairs; metric renamed to
       heuristic_marginal_diagnostic_ratio (was heuristic_diagnostic_purity)
  §6  Added per-(strategy,k,fold) long table; bootstrap CI uncertainty bands
  §7  Added explicit derivation markdown for H=h2(sigma(m_raw)); per-fold entropy table
  §8  Fixed pooling (N=183); added dedicated pooling audit cell + 08_ising_pooling_audit.csv
  §9  Kept as exploratory; no changes
  §10 Added version-history note; MCI descriptive analysis via pipeline_log mapping;
       10_von_neumann_group_descriptives_all_groups.csv
"""
import json, textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def md(source: str):
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip()}


def code(source: str):
    return {
        "cell_type": "code", "execution_count": None,
        "metadata": {}, "outputs": [],
        "source": textwrap.dedent(source).strip(),
    }


cells = []

# ── TITLE ──────────────────────────────────────────────────────────────────
cells.append(md("""
# 07 — Information-Theory Audit of the β-VAE + Logistic-Regression AD Pipeline
### Version 4 — Final Closing Pass (Publication-Grade)

**Scope** This notebook examines the best ADNI β-VAE run (`vae_3channels_beta65_pro`) from the
viewpoints of information bottleneck theory, latent mutual information, logistic regression as a
maximum-entropy readout, von Neumann spectral entropy, and ordinal spectral complexity.

**v3 critical fix (preserved)** `test_indices.npy` contains *local pool indices* (0–182 into the
AD+CN training pool), **not** global tensor indices.  v2 encoded wrong subjects.
v3/v4 use `test_tensor_idx.npy` / `train_dev_tensor_idx.npy` throughout.

**v4 additions** Run-mode system (fail-closed), dependency preflight, 300 dpi + PDF figures,
LaTeX table exports, §10 narrative corrected to reflect channel-specific findings.

**Changelog** See `notebooks/07_information_theory_changelog.md` (v1→v2→v3→v4).

| § | Title | Status |
|---|-------|--------|
| 0 | Configuration & run mode | always runs |
| 1 | Provenance & artifact discovery | always runs |
| 2 | Theoretical framing | markdown only |
| 3 | Shared utilities | always runs |
| 4 | Information bottleneck audit (training history + re-inference) | requires checkpoints |
| 5 | Latent MI & confounding [**train-set only**] | requires re-inference |
| 6 | Top-k readout [**nested clean + per-fold long table**] | requires re-inference |
| 7 | Score-level IT & calibration [**raw vs calibrated**] | from fold predictions |
| 8 | MaxEnt / Ising scaffold [**N=183 corrected**] | requires re-inference |
| 9 | Spectral ordinal complexity proxy [**exploratory**] | requires FC tensor |
| 10 | Von Neumann entropy [corrected + MCI descriptive] | requires FC tensor |
| 11 | Execution report & outputs | always runs |
"""))

# ── §0 CONFIGURATION ───────────────────────────────────────────────────────
cells.append(md("## §0 — Configuration"))
cells.append(code("""
import os, sys, shutil
from pathlib import Path

# ── RUN MODE ────────────────────────────────────────────────────────────────
# "full_recompute"  : requires all dependencies; raises RuntimeError if any missing;
#                     moves stale legacy outputs to _legacy/ before starting
# "artifact_review" : loads existing cached tables/figures; labels them explicitly;
#                     does NOT require betavae_xai or torch
RUN_MODE = "full_recompute"   # change to "artifact_review" on restricted environments

_VALID_MODES = ("full_recompute", "artifact_review")
if RUN_MODE not in _VALID_MODES:
    raise ValueError(f"RUN_MODE must be one of {_VALID_MODES}, got {RUN_MODE!r}")

def _find_project_root(marker="pyproject.toml"):
    here = Path().resolve()
    for p in [here, *here.parents]:
        if (p / marker).exists():
            return p
    return here

PROJECT_ROOT = _find_project_root()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

RESULTS_DIR = PROJECT_ROOT / "results" / "vae_3channels_beta65_pro"
DATA_DIR    = PROJECT_ROOT / "data"
OUTPUT_DIR  = RESULTS_DIR / "information_theory_output"
LEGACY_DIR  = OUTPUT_DIR / "_legacy"

for _d in [OUTPUT_DIR / "Figures", OUTPUT_DIR / "Tables", OUTPUT_DIR / "Logs", LEGACY_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# In full_recompute: quarantine known superseded legacy files so they don't
# appear as authoritative outputs. They are moved, not deleted.
_LEGACY_FILES = [
    "Tables/05_latent_dim_statistics.csv",
    "Tables/06_topk_auc_by_strategy.csv",
    "Tables/09_spectral_entropy_proxy.csv",
    "Tables/10_von_neumann_group_stats.csv",
]
if RUN_MODE == "full_recompute":
    for _lf in _LEGACY_FILES:
        _src = OUTPUT_DIR / _lf
        if _src.exists():
            _dst = LEGACY_DIR / _src.name
            shutil.move(str(_src), str(_dst))
            print(f"[LEGACY] moved {_src.name} -> _legacy/")

DO_TIME_SERIES  = True
DO_MAXENT       = True
DO_VON_NEUMANN  = True
DO_BETA_SWEEP   = False
SEED            = 42

N_FOLDS       = 5
LATENT_DIM    = 256
BETA_VAE      = 6.5
CHANNELS_IDX  = [1, 0, 2]
CHANNEL_NAMES = [
    "Pearson_Full_FisherZ_Signed",
    "Pearson_OMST_GCE_Signed_Weighted",
    "MI_KNN_Symmetric",
]
N_CHANNELS    = len(CHANNEL_NAMES)
IMAGE_SIZE    = 131
META_FEATURES = ["Age", "Sex"]

print(f"RUN_MODE     : {RUN_MODE}")
print(f"PROJECT_ROOT : {PROJECT_ROOT}")
print(f"RESULTS_DIR  : {RESULTS_DIR}")
print(f"OUTPUT_DIR   : {OUTPUT_DIR}")
"""))

# ── §1 IMPORTS ─────────────────────────────────────────────────────────────
cells.append(md("## §1 — Dependency Preflight & Imports"))
cells.append(code("""
# ── Dependency preflight ─────────────────────────────────────────────────
import sys, importlib
_preflight = {}

for _pkg in ["numpy", "scipy", "pandas", "sklearn", "matplotlib", "seaborn", "joblib"]:
    try:
        _m = importlib.import_module(_pkg)
        _preflight[_pkg] = getattr(_m, "__version__", "ok")
    except ImportError:
        _preflight[_pkg] = "MISSING"

try:
    import torch
    _preflight["torch"] = torch.__version__
    _TORCH_AVAILABLE = True
except ImportError:
    _preflight["torch"] = "MISSING"
    _TORCH_AVAILABLE = False

try:
    import optuna
    _preflight["optuna"] = optuna.__version__
except ImportError:
    _preflight["optuna"] = "MISSING"

_BETAVAE_AVAILABLE = False
try:
    from betavae_xai.models import ConvolutionalVAE
    _BETAVAE_AVAILABLE = True
    _preflight["betavae_xai"] = "ok"
except ImportError as _e:
    _preflight["betavae_xai"] = f"MISSING ({_e})"

print(f"{'Package':<18} {'Status'}")
print("-" * 40)
for _k, _v in _preflight.items():
    _status = "[OK]" if _v != "MISSING" and not str(_v).startswith("MISSING") else "[MISSING]"
    print(f"{_k:<18} {_status}  {_v}")

_missing_hard = [k for k, v in _preflight.items()
                 if str(v).startswith("MISSING") and k not in ("betavae_xai", "optuna", "torch")]
_missing_reinf = [k for k, v in _preflight.items()
                  if str(v).startswith("MISSING") and k in ("betavae_xai", "optuna", "torch")]

if RUN_MODE == "full_recompute":
    if _missing_hard:
        raise RuntimeError(
            f"[FATAL] full_recompute mode requires: {_missing_hard}.\\n"
            "Install missing packages or switch to RUN_MODE='artifact_review'.\\n"
            "See environment.yml for exact versions."
        )
    if _missing_reinf:
        raise RuntimeError(
            f"[FATAL] full_recompute mode requires re-inference dependencies: {_missing_reinf}.\\n"
            "§§4-6 and §8 cannot run without betavae_xai + torch + optuna.\\n"
            "Install missing packages or switch to RUN_MODE='artifact_review'.\\n"
            "See environment.yml for exact versions (optuna>=4.0, torch>=2.5)."
        )
    print("\\n[OK] All dependencies present for full_recompute mode.")
else:
    if _missing_reinf:
        print(f"\\n[INFO] artifact_review mode: re-inference deps missing ({_missing_reinf})")
        print("       Sections 4-6, 8 will load from cached artifacts.")
    if _missing_hard:
        raise RuntimeError(
            f"[FATAL] Even artifact_review requires core packages: {_missing_hard}"
        )
    print("\\n[OK] Core dependencies present for artifact_review mode.")
"""))

cells.append(code("""
import warnings, json, time, re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from IPython.display import display
from scipy.special import xlogy, expit as sigmoid
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.covariance import LedoitWolf

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(SEED)
if _TORCH_AVAILABLE:
    import torch as _torch_mod
    _torch_mod.manual_seed(SEED)
    _DEVICE = _torch_mod.device("cpu")
    print(f"[OK] torch {_torch_mod.__version__}, device={_DEVICE}")
else:
    _DEVICE = None

print(f"Python {sys.version.split()[0]}  |  RUN_MODE={RUN_MODE}  |  SEED={SEED}")
print(f"betavae_xai available: {_BETAVAE_AVAILABLE}")
"""))

# ── §1 ARTIFACT DISCOVERY ──────────────────────────────────────────────────
cells.append(code("""
import glob as _glob, re

fold_dirs = sorted([d for d in RESULTS_DIR.iterdir()
                    if d.is_dir() and re.match(r"fold_\\d+", d.name)])
N_FOLDS_FOUND = len(fold_dirs)

fold_pred_paths   = {}
fold_history_paths = {}
fold_logreg_paths  = {}
fold_norm_paths    = {}
fold_vae_paths     = {}
# v3 fix: use test_tensor_idx.npy / train_dev_tensor_idx.npy (global tensor indices)
# test_indices.npy / train_dev_indices.npy are LOCAL pool indices — do not use for GLOBAL_TENSOR
fold_test_tidx_paths  = {}  # test_tensor_idx.npy  (global)
fold_train_tidx_paths = {}  # train_dev_tensor_idx.npy (global)

for fd in fold_dirs:
    k = int(re.search(r"(\\d+)", fd.name).group(1))
    def _ep(path): return path if path.exists() else None
    fold_pred_paths[k]       = _ep(fd / f"test_predictions_logreg.csv")
    fold_history_paths[k]    = _ep(fd / f"vae_train_history_fold_{k}.joblib")
    fold_logreg_paths[k]     = _ep(fd / f"classifier_logreg_raw_pipeline_fold_{k}.joblib")
    fold_norm_paths[k]       = _ep(fd / "vae_norm_params.joblib")
    fold_vae_paths[k]        = _ep(fd / f"vae_model_fold_{k}.pt")
    fold_test_tidx_paths[k]  = _ep(fd / "test_tensor_idx.npy")       # global tensor idx
    fold_train_tidx_paths[k] = _ep(fd / "train_dev_tensor_idx.npy")  # global tensor idx

all_history_paths  = sorted(RESULTS_DIR.glob("all_folds_vae_training_history_*.joblib"))
all_history_path   = all_history_paths[0] if all_history_paths else None
tensor_candidates  = sorted(DATA_DIR.glob("**/*GLOBAL_TENSOR*.npz"))
global_tensor_path = tensor_candidates[0] if tensor_candidates else None
meta_path          = DATA_DIR / "SubjectsData_AAL3_procesado2.csv"

print(f"fold dirs found: {N_FOLDS_FOUND}  |  global_tensor: {'FOUND' if global_tensor_path else 'MISSING'}")
for ki in sorted(fold_vae_paths):
    print(f"  fold {ki}: vae={'OK' if fold_vae_paths[ki] else 'X'}  "
          f"pred={'OK' if fold_pred_paths.get(ki) else 'X'}  "
          f"test_tidx={'OK' if fold_test_tidx_paths.get(ki) else 'X'}  "
          f"train_tidx={'OK' if fold_train_tidx_paths.get(ki) else 'X'}")
"""))

cells.append(code("""
# ── Load metadata + build SubjectID→tensor_idx via pipeline log ─────────
meta_df   = None
tidx_map  = {}   # SubjectID -> global tensor_idx for ALL subjects (AD+CN+MCI)

# Primary source: pipeline log (one row per global tensor position)
_plog_candidates = sorted(DATA_DIR.glob("**/*pipeline_log*.csv"))
if _plog_candidates:
    _plog = pd.read_csv(_plog_candidates[0])
    # Row position == tensor_idx
    _plog["tensor_idx"] = np.arange(len(_plog))
    tidx_map = dict(zip(_plog["id"].str.strip(), _plog["tensor_idx"].astype(int)))
    print(f"[OK] pipeline log: {len(_plog)} rows  ->  tensor_idx map built for {len(tidx_map)} subjects")

# Fallback: fold subject files (AD+CN only)
if not tidx_map:
    _subj_parts = []
    for fd_ in fold_dirs:
        for fn_ in ["test_subjects_fold.csv", "train_dev_subjects_fold.csv"]:
            _sp = fd_ / fn_
            if _sp.exists():
                _subj_parts.append(pd.read_csv(_sp))
    if _subj_parts:
        _all_subj = pd.concat(_subj_parts, ignore_index=True).drop_duplicates("SubjectID")
        tidx_map = dict(zip(_all_subj["SubjectID"], _all_subj["tensor_idx"].astype(int)))
        print(f"[FALLBACK] Built tensor_idx map from fold subject files: {len(tidx_map)} subjects")

if meta_path.exists():
    meta_df = pd.read_csv(meta_path)
    meta_df["tensor_idx"] = meta_df["SubjectID"].map(tidx_map)
    _grp_col = next((c for c in ["ResearchGroup_Mapped","Group","ResearchGroup","Diagnosis"]
                     if c in meta_df.columns), None)
    if _grp_col:
        meta_df["y_true"] = (meta_df[_grp_col].str.strip().str.upper() == "AD").astype(float)
        print(f"Diagnosis column: {_grp_col}")
        print(f"Group distribution: {dict(meta_df[_grp_col].value_counts())}")
        print(f"Subjects with tensor_idx: {meta_df['tensor_idx'].notna().sum()}")
    print(f"metadata shape: {meta_df.shape}")

# Save discovery manifest
_manifest = {
    "created_utc"         : time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "notebook_version"    : "v3",
    "results_dir"         : str(RESULTS_DIR),
    "output_dir"          : str(OUTPUT_DIR),
    "n_fold_dirs_found"   : N_FOLDS_FOUND,
    "all_history_path"    : str(all_history_path) if all_history_path else None,
    "global_tensor_path"  : str(global_tensor_path) if global_tensor_path else None,
    "tidx_map_n_subjects" : len(tidx_map),
    "tidx_source"         : "pipeline_log" if _plog_candidates else "fold_subject_files",
}
with open(OUTPUT_DIR / "discovery_manifest.json", "w") as f:
    json.dump(_manifest, f, indent=2)
print("Discovery manifest saved.")
"""))

# ── LATENT RE-INFERENCE BLOCK ──────────────────────────────────────────────
cells.append(md("""
### Latent Re-Inference  [v3: global tensor indices corrected]

**v3 fix (critical):** Previous versions used `test_indices.npy` / `train_dev_indices.npy`
which contain **local pool indices** (0–182 into the AD+CN subset), not global tensor indices.
Indexing `GLOBAL_TENSOR` with these local indices encoded the wrong subjects.

v3 uses `test_tensor_idx.npy` / `train_dev_tensor_idx.npy` which contain the correct
global tensor row indices matching `tensor_idx` in the fold prediction CSVs.

The encoder is run in `eval()` / `torch.no_grad()` — weights are never modified.
"""))
cells.append(code("""
fold_mu_dict           = {}  # k -> [N_test,  256]  test-set mu  (correct subjects)
fold_logvar_dict       = {}  # k -> [N_test,  256]
fold_y_dict            = {}  # k -> [N_test]         from fold prediction CSV (ground truth)
fold_tensor_idx_dict   = {}  # k -> [N_test]         global tensor indices (for §8 pooling audit)
fold_mu_train_dict     = {}  # k -> [N_train, 256]   training-set mu
fold_y_train_dict      = {}  # k -> [N_train]        from train_dev_subjects_fold.csv

_REINFERENCE_AVAILABLE = False
_reinference_log       = []

if not (_BETAVAE_AVAILABLE and _TORCH_AVAILABLE):
    print("[SKIP] Re-inference: betavae_xai or torch unavailable")
elif global_tensor_path is None:
    print("[SKIP] Re-inference: global tensor not found")
else:
    _td = np.load(global_tensor_path)
    _key = next((k for k in _td.files if "tensor" in k.lower()), _td.files[0])
    GLOBAL_TENSOR = _td[_key].astype(np.float32)
    TENSOR_3CH    = GLOBAL_TENSOR[:, CHANNELS_IDX, :, :]
    print(f"Global tensor: {GLOBAL_TENSOR.shape}  3-ch: {TENSOR_3CH.shape}")

    def _apply_norm(x3, params):
        out = x3.copy(); R = x3.shape[-1]; off = ~np.eye(R, dtype=bool)
        for c, p in enumerate(params):
            out[:, c, off] = (out[:, c, off] - float(p["mean"])) / (float(p["std"]) + 1e-12)
        return out

    def _build_vae(vae_path):
        vae = ConvolutionalVAE(input_channels=N_CHANNELS, latent_dim=LATENT_DIM,
                               image_size=IMAGE_SIZE, intermediate_fc_dim_config="quarter",
                               dropout_rate=0.15, num_conv_layers_encoder=4,
                               decoder_type="convtranspose").to(_DEVICE)
        state = torch.load(vae_path, map_location=_DEVICE, weights_only=False)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        vae.load_state_dict(state); vae.eval()
        return vae

    def _encode(vae, tidx_arr, norm_params):
        X = _apply_norm(TENSOR_3CH[tidx_arr], norm_params).astype(np.float32)
        with torch.no_grad():
            mu, lv = vae.encode(torch.from_numpy(X).to(_DEVICE))
        return mu.cpu().numpy(), lv.cpu().numpy()

    def _labels_from_pred_csv(k):
        p = fold_pred_paths.get(k)
        if p and p.exists():
            df_ = pd.read_csv(p)
            return dict(zip(df_["tensor_idx"].astype(int), df_["y_true"].astype(float)))
        return {}

    def _labels_from_subj_file(k, fname):
        for fd_ in fold_dirs:
            if int(re.search(r"(\\d+)", fd_.name).group(1)) == k:
                sp = fd_ / fname
                if sp.exists():
                    df_ = pd.read_csv(sp)
                    return dict(zip(
                        df_["tensor_idx"].astype(int),
                        (df_["ResearchGroup_Mapped"].str.upper() == "AD").astype(float).values
                    ))
        return {}

    for _k in sorted(fold_vae_paths.keys()):
        try:
            vp  = fold_vae_paths[_k]
            np_ = fold_norm_paths.get(_k)
            te  = fold_test_tidx_paths.get(_k)   # global tensor idx
            tr  = fold_train_tidx_paths.get(_k)  # global tensor idx
            if not (vp and np_ and te and vp.exists() and np_.exists() and te.exists()):
                _reinference_log.append(f"fold {_k}: SKIPPED (missing artifacts)")
                continue
            norm = joblib.load(np_)
            vae  = _build_vae(vp)

            # Test set  (global tensor indices)
            test_tidx = np.load(te).astype(int)
            mu_te, lv_te = _encode(vae, test_tidx, norm)
            fold_mu_dict[_k]         = mu_te
            fold_logvar_dict[_k]     = lv_te
            fold_tensor_idx_dict[_k] = test_tidx  # store for audit

            # Labels from prediction CSV (authoritative)
            lbl_map = _labels_from_pred_csv(_k)
            fold_y_dict[_k] = np.array([lbl_map.get(i, np.nan) for i in test_tidx])

            # Train set  (global tensor indices)
            if tr and tr.exists():
                train_tidx = np.load(tr).astype(int)
                mu_tr, _   = _encode(vae, train_tidx, norm)
                fold_mu_train_dict[_k]  = mu_tr
                tr_lbl = _labels_from_subj_file(_k, "train_dev_subjects_fold.csv")
                fold_y_train_dict[_k] = np.array([tr_lbl.get(i, np.nan) for i in train_tidx])

            n_nan_test = int(np.isnan(fold_y_dict[_k]).sum())
            _reinference_log.append(
                f"fold {_k}: OK  N_test={len(test_tidx)} "
                f"(nan_y={n_nan_test})  "
                f"N_train={len(fold_mu_train_dict.get(_k, []))}"
            )
            _REINFERENCE_AVAILABLE = True
        except Exception as exc:
            _reinference_log.append(f"fold {_k}: FAILED -- {exc}")

    print("\\nRe-inference log:")
    for l in _reinference_log: print(f"  {l}")
    print(f"\\n_REINFERENCE_AVAILABLE = {_REINFERENCE_AVAILABLE}")
"""))

# ── §2 THEORETICAL FRAMING ─────────────────────────────────────────────────
cells.append(md(r"""
## §2 — Theoretical Framing

### 2.1  β-VAE and Information Bottleneck

$$
\mathcal{L}_\beta = \mathbb{E}_{q_\phi(z|x)}\bigl[\log p_\theta(x|z)\bigr]
  - \beta\, D_\text{KL}\bigl(q_\phi(z|x) \| p(z)\bigr)
$$

With β = 6.5, the encoder is penalised for using more *rate* (bits) than necessary.
This does **not** guarantee disentanglement; it produces a distributed compressed
representation.  Disentanglement in the strict sense (factorial aggregate posterior)
would additionally require $TC_q(Z) \approx 0$, which is not guaranteed or claimed here.

### 2.2  KL Decomposition and Active Units

$$
\mathbb{E}_x[KL_k] \approx \tfrac{1}{2}(\mu_k^2 + \sigma_k^2 - \ln\sigma_k^2 - 1)
$$

Active unit: $AU_k = \mathbf{1}[\text{Var}_x[\mu_k(x)] > \varepsilon]$, $\varepsilon=0.01$.

**Total Correlation** (Gaussian approximation):
$$
TC_q(Z) \approx \tfrac{1}{2}\Bigl(\sum_k \ln\hat\sigma^2_k - \ln|\hat\Sigma|\Bigr)
$$
We use the Ledoit-Wolf shrinkage estimator for $\hat\Sigma$ to improve stability when
$N < D$.  Even with shrinkage, TC is *exploratory* because the Gaussian assumption is
approximate.

### 2.3  Logistic Regression as Maximum-Entropy Readout and Entropy Identity

The logistic regression classifier defines:
$$
p_\text{raw}(y=1\mid z) = \sigma(w^\top z + b)
$$
where $\sigma$ is the sigmoid function.  The **raw linear margin** is:
$$
m_\text{raw} = w^\top z + b = \text{logit}(p_\text{raw})
$$
The **predictive entropy** in bits is:
$$
H_\text{raw}(Y\mid z) = -p_\text{raw}\log_2 p_\text{raw} - (1-p_\text{raw})\log_2(1-p_\text{raw})
  = h_2\!\left(\sigma(m_\text{raw})\right)
$$
This identity — entropy as a deterministic function of the margin — holds **only for the
raw linear margin $m_\text{raw}$**, because $p_\text{raw}$ is itself a deterministic function
of $m_\text{raw}$ via the sigmoid.

**After calibration**, the calibrated probability $p_\text{cal}$ is obtained by a separate
mapping (e.g., Platt scaling / isotonic regression) applied to $p_\text{raw}$.  The calibrated
logit $s_\text{cal} = \text{logit}(p_\text{cal})$ is **not** equal to $m_\text{raw}$, and the
theoretical curve $h_2(\sigma(\cdot))$ must **not** be overlaid in the calibrated panel.

| Score | Definition | Entropy |
|-------|-----------|---------|
| $m_\text{raw}$ | $w^\top z + b$ (linear margin) | $H = h_2(\sigma(m_\text{raw}))$ — theoretical curve applies |
| $s_\text{cal}$ | $\text{logit}(p_\text{cal})$ after calibration | No closed-form — theoretical curve does NOT apply |

### 2.4  Marginal MI and Heuristic Marginal Diagnostic Ratio

$I(z_k; Y)$ computed here is the **marginal** (not conditional) mutual information between
individual latent dimension $z_k$ and the diagnosis label $Y$.  It does not account for
dependencies between dimensions or confounders.

The *heuristic marginal diagnostic ratio* is defined as:
$$
r_k = \frac{I(z_k; Y)}{I(z_k; Y) + I(z_k; \text{Site}) + I(z_k; \text{Age}) + I(z_k; \text{Sex}) + \varepsilon}
$$
**Caveats:** (i) marginal MI, not conditional; (ii) does not prove deconfounding;
(iii) underestimates distributed multivariate confounding; (iv) not a canonical IT quantity.

### 2.5  Von Neumann Entropy

$$
S(\rho) = -\text{tr}(\rho\log\rho), \quad \rho = \tilde C / \text{tr}(\tilde C),
\quad \tilde C = C - \lambda_{\min}(C)I
$$
"""))

# ── §3 SHARED UTILITIES ────────────────────────────────────────────────────
cells.append(md("## §3 — Shared Utilities"))
cells.append(code("""
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.constrained_layout.use": True,
})
PALETTE = {"AD": "#d62728", "CN": "#1f77b4", "MCI": "#ff7f0e"}

def savefig(name, fig=None):
    \"\"\"Save figure as 300 dpi PNG and PDF vector; display inline.\"\"\"
    fig = fig or plt.gcf()
    png_path = OUTPUT_DIR / "Figures" / name
    pdf_path = OUTPUT_DIR / "Figures" / name.replace(".png", ".pdf")
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(pdf_path, bbox_inches="tight")
    display(fig)
    plt.close(fig)
    print(f"  [fig] {png_path.name}  [pdf] {pdf_path.name}")

def save_table(df, name):
    p = OUTPUT_DIR / "Tables" / name
    df.to_csv(p, index=False)
    print(f"  [tbl] {p.name}  ({len(df)} rows)")

def save_latex(df, name, caption="", label="", float_fmt="%.4f", index=False):
    \"\"\"Export a DataFrame as a LaTeX table (.tex) alongside the CSV.\"\"\"
    try:
        tex_path = OUTPUT_DIR / "Tables" / name.replace(".csv", ".tex")
        tex_str  = df.to_latex(index=index, float_format=float_fmt,
                               caption=caption if caption else None,
                               label=label if label else None,
                               escape=True)
        tex_path.write_text(tex_str)
        print(f"  [tex] {tex_path.name}")
    except Exception as _e:
        print(f"  [WARN] LaTeX export failed for {name}: {_e}")

def bootstrap_ci(x, fn=np.mean, n_boot=2000, ci=0.95, seed=SEED):
    rng = np.random.default_rng(seed)
    s = [fn(rng.choice(x, len(x), replace=True)) for _ in range(n_boot)]
    return float(np.percentile(s, (1-ci)/2*100)), float(np.percentile(s, (1+ci)/2*100))

def per_dim_kl(mu_arr, logvar_arr=None):
    mu = np.asarray(mu_arr, float)
    if logvar_arr is not None:
        lv  = np.asarray(logvar_arr, float)
        var = np.exp(lv)
        return (0.5*(mu**2 + var - lv - 1)).mean(0)
    return (0.5*mu**2).mean(0)

def active_units(mu_arr, eps=0.01):
    v = np.var(mu_arr, axis=0); m = v > eps
    return {"n_active": int(m.sum()), "frac_active": float(m.mean()), "var": v, "mask": m}

def tc_gaussian_lw(mu_arr, ridge=1e-9):
    \"\"\"Total Correlation under Gaussian approx with Ledoit-Wolf shrinkage (exploratory).\"\"\"
    try:
        lw = LedoitWolf().fit(mu_arr)
        C  = lw.covariance_ + np.eye(mu_arr.shape[1])*ridge
        _, logdet = np.linalg.slogdet(C)
        return float(max(0., 0.5*(np.sum(np.log(np.diag(C)+1e-30)) - logdet)))
    except Exception:
        return np.nan

def mi_classif(Z, y, nn=3):
    Z = np.asarray(Z, float); y = np.asarray(y).ravel()
    mask = ~np.isnan(y); Z, y = Z[mask], y[mask].astype(int)
    if len(np.unique(y)) < 2 or len(y) < 5:
        return np.full(Z.shape[1], np.nan)
    return mutual_info_classif(Z, y, discrete_features=False,
                               n_neighbors=max(1, min(nn, len(y)-1)), random_state=SEED)

def mi_regress(Z, y_cont, nn=3):
    Z = np.asarray(Z, float); y = np.asarray(y_cont, float).ravel()
    mask = ~np.isnan(y); Z, y = Z[mask], y[mask]
    if len(y) < 5 or np.std(y) < 1e-9:
        return np.full(Z.shape[1], np.nan)
    return mutual_info_regression(Z, y, discrete_features=False,
                                  n_neighbors=max(1, min(nn, len(y)-1)), random_state=SEED)

def ece_brier(y_true, y_prob, n_bins=10):
    yt = np.asarray(y_true, float); yp = np.clip(np.asarray(y_prob, float), 1e-7, 1-1e-7)
    mask = ~np.isnan(yt); yt, yp = yt[mask], yp[mask]
    edges = np.linspace(0,1,n_bins+1); ece = 0.
    for lo, hi in zip(edges[:-1], edges[1:]):
        idx = (yp>=lo)&(yp<hi)
        if idx.sum(): ece += idx.sum()/len(yt)*abs(yt[idx].mean()-yp[idx].mean())
    return {"ece": float(ece), "brier": float(brier_score_loss(yt,yp)),
            "log_loss": float(log_loss(yt,yp))}

def bin_entropy_bits(p, eps=1e-9):
    p = np.clip(np.asarray(p, float), eps, 1-eps)
    return -(p*np.log2(p) + (1-p)*np.log2(1-p))

def ordinal_patterns(series, D=5, tau=1):
    n = len(series)-(D-1)*tau
    return [tuple(np.argsort(series[i:i+D*tau:tau])) for i in range(n)] if n>0 else []

def spectral_complexity(series, D=5, tau=1):
    import itertools; from math import factorial, log
    pats = ordinal_patterns(series, D, tau)
    if not pats: return np.nan, np.nan
    from collections import Counter
    cnt = Counter(pats); total = sum(cnt.values()); nP = factorial(D)
    all_p = list(itertools.permutations(range(D)))
    Pf = np.array([cnt.get(p,0)/total for p in all_p])
    Pe = np.full(nP, 1/nP)
    H  = -np.sum(xlogy(Pf, Pf)) / log(nP)
    def kl(a,b): return np.sum(xlogy(a, np.where(b>0, a/b, 1)))
    def js(a,b):
        m=0.5*(a+b); return float(0.5*kl(a,m)+0.5*kl(b,m))
    Q = js(Pf, Pe)
    p_star=0.5*(Pe+np.eye(nP)[0]); Qmax=js(p_star, Pe)
    return float(H), float(H * Q / Qmax if Qmax>0 else H*Q)

def von_neumann_entropy(matrix, eps=1e-12):
    C = np.asarray(matrix, float)
    ev = np.linalg.eigvalsh(C); lmin = ev.min()
    if lmin < 0: ev -= lmin
    tr = ev.sum()
    if tr < eps: return np.nan
    rho = np.clip(ev/tr, 0, None)
    return float(-np.sum(xlogy(rho, rho+eps)))

print("[OK] §3 utilities loaded")
"""))

# ── §4 IB AUDIT ────────────────────────────────────────────────────────────
cells.append(md("""
## §4 — Information Bottleneck Audit

We audit the encoder's compression behaviour via:
1. Training-history R–D proxy trajectories (training path, not a Pareto frontier)
2. Per-fold KL per dimension, active units, and Gaussian TC (Ledoit-Wolf shrinkage)

> **v3 note:** Re-inference now uses the correct global tensor indices, so per-fold
> statistics correspond to the actual held-out test subjects.
>
> **Important:** The encoder produces a **distributed** representation, not a disentangled
> one.  β=6.5 compresses the rate but does **not** guarantee factorial aggregate posterior.
> TC is exploratory; Gaussian TC with Ledoit-Wolf shrinkage is numerically stable but
> the Gaussian approximation remains approximate with N≈37 test subjects.
"""))
cells.append(code("""
histories = None
if all_history_path and all_history_path.exists():
    histories = joblib.load(all_history_path)
    print(f"[OK] {len(histories)} fold histories, {len(histories[0]['train_recon'])} epochs each")
elif any(p and p.exists() for p in fold_history_paths.values()):
    histories = [joblib.load(p) for k,p in sorted(fold_history_paths.items()) if p and p.exists()]
    print(f"[OK] {len(histories)} per-fold histories loaded")
else:
    print("[SKIP] No training histories found")
"""))
cells.append(code("""
if histories is None:
    print("[SKIP] §4b: no histories")
else:
    n_h = len(histories)
    fig, axes = plt.subplots(3, n_h, figsize=(4*n_h, 9))
    if n_h == 1: axes = axes[:, None]
    for fi, h in enumerate(histories):
        ep  = np.arange(1, len(h["train_recon"])+1)
        beta_ = np.asarray(h["beta"], float)
        Dtr = np.asarray(h["train_recon"], float)
        Dvl = np.asarray(h["val_recon"],   float)
        Rtr = np.asarray(h["train_kld"],   float)
        Rvl = np.asarray(h["val_kld"],     float)
        axes[0,fi].plot(ep,Dtr,lw=0.8,label="train"); axes[0,fi].plot(ep,Dvl,lw=0.8,label="val",alpha=0.8)
        axes[0,fi].set_title(f"Fold {fi+1}"); axes[0,fi].set_ylabel("Recon MSE"); axes[0,fi].legend(fontsize=7)
        axes[1,fi].plot(ep,Rtr,lw=0.8,color="C2"); axes[1,fi].plot(ep,Rvl,lw=0.8,color="C3",alpha=0.8)
        ax1b=axes[1,fi].twinx(); ax1b.plot(ep,beta_,lw=0.5,color="grey",alpha=0.4)
        ax1b.set_ylabel("beta"); axes[1,fi].set_ylabel("KLD nats")
        sc = axes[2,fi].scatter(Rtr[::5],Dtr[::5],c=beta_[::5],cmap="plasma",s=3,alpha=0.6)
        plt.colorbar(sc,ax=axes[2,fi],label="beta"); axes[2,fi].set_xlabel("R=KLD"); axes[2,fi].set_ylabel("D=Recon")
        axes[2,fi].set_title("R-D proxy (train path, NOT Pareto frontier)")
    fig.suptitle(
        "Training R-D proxy  |  beta varies per epoch (cyclical annealing)\\n"
        "This is NOT a cross-beta Pareto frontier.  Single operating point: beta=6.5.",
        fontsize=10
    )
    savefig("04_training_curves_rd_proxy.png", fig)
"""))
cells.append(code("""
if not _REINFERENCE_AVAILABLE:
    print("[SKIP] §4c: no re-inference")
else:
    kl_folds, au_folds, tc_folds = {}, {}, {}
    for k in sorted(fold_mu_dict):
        mu  = fold_mu_dict[k]; lv = fold_logvar_dict.get(k)
        kl_folds[k] = per_dim_kl(mu, lv)
        au_folds[k] = active_units(mu)
        tc_folds[k] = tc_gaussian_lw(mu)

    kl_matrix = np.stack([kl_folds[k] for k in sorted(kl_folds)], 0)
    kl_mean, kl_std = kl_matrix.mean(0), kl_matrix.std(0)

    rows_ib = []
    for k in sorted(au_folds):
        y_k   = fold_y_dict.get(k, np.array([]))
        n_ad  = int((y_k==1).sum()); n_cn = int((y_k==0).sum()); n_nan = int(np.isnan(y_k).sum())
        rows_ib.append({
            "fold": k,
            "n_test": len(fold_mu_dict[k]),
            "n_AD": n_ad, "n_CN": n_cn, "n_nan_y": n_nan,
            "n_active_units": au_folds[k]["n_active"],
            "frac_active_units": round(au_folds[k]["frac_active"], 4),
            "mean_kl_nats": round(float(kl_folds[k].mean()), 4),
            "tc_gaussian_lw_nats": round(float(tc_folds[k]), 4),
            "tc_note": "exploratory - Gaussian approx + LW shrinkage",
        })
    df_ib = pd.DataFrame(rows_ib)
    save_table(df_ib, "04_ib_fold_summary.csv")
    print(df_ib.to_string(index=False))

    order = np.argsort(-kl_mean)
    fig, axes = plt.subplots(1,3,figsize=(15,4))
    axes[0].bar(np.arange(LATENT_DIM), kl_mean[order], width=1., yerr=kl_std[order],
                error_kw={"elinewidth":0.4})
    axes[0].axhline(0.01,color="red",lw=1,ls="--",label="eps=0.01 (AU)")
    axes[0].set_xlabel("Latent dim (sorted by KL)"); axes[0].set_ylabel("E[KL_k] nats")
    axes[0].set_title("Per-dim KL (mean +/- std)"); axes[0].legend()
    fks = sorted(au_folds)
    axes[1].bar([str(k) for k in fks],[au_folds[k]["n_active"] for k in fks])
    axes[1].set_xlabel("Fold"); axes[1].set_ylabel("# Active units"); axes[1].set_title("Active units (eps=0.01)")
    k0 = sorted(fold_mu_dict)[0]; top32 = np.argsort(-kl_mean)[:32]
    sns.heatmap(np.corrcoef(fold_mu_dict[k0][:,top32].T),ax=axes[2],
                cmap="RdBu_r",center=0,vmin=-1,vmax=1,square=True,
                xticklabels=False,yticklabels=False)
    axes[2].set_title(f"Latent corr - top-32 KL dims (fold {k0} test)")
    savefig("04_kl_active_units_latent_corr.png", fig)
    save_table(
        pd.DataFrame({"dim":np.arange(LATENT_DIM),"kl_mean":kl_mean,"kl_std":kl_std,
                      **{f"kl_fold{k}":kl_folds[k] for k in sorted(kl_folds)}}),
        "04_kl_per_dimension.csv"
    )
"""))

# ── §5 LATENT MI (TRAIN-ONLY) ──────────────────────────────────────────────
cells.append(md("""
## §5 — Latent MI and Confounding  [Training Set Only]

> **Methodological note — v2/v3 correction:** MI rankings are computed **exclusively
> on training-set latents** (`fold_mu_train_dict`).  Using test-set latents to rank
> dimensions and then evaluate the same test set creates information leakage.
>
> **v3 additional note:** Training-set latents now correctly encode training-set subjects
> (via `train_dev_tensor_idx.npy`).  v2 encoded local-pool-indexed subjects.
>
> All MI estimates are **marginal** $I(z_k; Y)$ — one dimension at a time.  This is not
> conditional MI and does not account for multivariate structure.

### Heuristic Marginal Diagnostic Ratio [not a canonical IT quantity]

$$
r_k = \\frac{I(z_k; Y)}{I(z_k; Y) + I(z_k; \\text{Site}) + I(z_k; \\text{Age}) + I(z_k; \\text{Sex}) + \\varepsilon}
$$

**Caveats** (mandatory disclosure):
- This is a **marginal** ratio — it does not prove conditional deconfounding
- It **underestimates** distributed multivariate confounding (dimensions can jointly encode
  confounders even when each individually appears diagnostic)
- It is not a canonical information-theoretic quantity and should not be cited as such
"""))
cells.append(code("""
_MI_AVAILABLE = False
if not _REINFERENCE_AVAILABLE or meta_df is None:
    print("[SKIP] §5: requires re-inference and metadata")
else:
    _MI_AVAILABLE = True
    _site_col = next((c for c in ["Manufacturer","Site","site","Scanner","Site3"]
                      if c in meta_df.columns), None)
    meta_idx_full = meta_df.set_index("tensor_idx") if "tensor_idx" in meta_df.columns else meta_df

    def _get_meta_vals(tidx_arr, col):
        return np.array([
            meta_idx_full.loc[i, col] if i in meta_idx_full.index else np.nan
            for i in tidx_arr
        ])

    mi_per_fold_train = {}

    for k in sorted(fold_mu_train_dict):
        Z_tr   = fold_mu_train_dict[k]
        y_tr   = fold_y_train_dict.get(k, np.full(len(Z_tr), np.nan))
        train_tidx = np.load(fold_train_tidx_paths[k]).astype(int)

        fm = {}
        fm["Y_AD_CN"] = mi_classif(Z_tr, y_tr)

        if _site_col and _site_col in meta_idx_full.columns:
            from sklearn.preprocessing import LabelEncoder
            sv = pd.Series(_get_meta_vals(train_tidx, _site_col)).fillna("Unknown").values
            le = LabelEncoder(); se = le.fit_transform(sv)
            fm["Site"] = mi_classif(Z_tr, se) if len(np.unique(se))>=2 else np.full(LATENT_DIM,np.nan)
        else:
            fm["Site"] = np.full(LATENT_DIM, np.nan)

        for col, fn in [("Age", mi_regress), ("Sex", mi_classif)]:
            if col in meta_idx_full.columns:
                vals = _get_meta_vals(train_tidx, col)
                if col == "Sex":
                    from sklearn.preprocessing import LabelEncoder
                    sv2 = pd.Series(vals).fillna("Unknown").astype(str).values
                    le2 = LabelEncoder(); se2 = le2.fit_transform(sv2)
                    fm[col] = mi_classif(Z_tr, se2) if len(np.unique(se2))>=2 else np.full(LATENT_DIM,np.nan)
                else:
                    fm[col] = fn(Z_tr, vals)
            else:
                fm[col] = np.full(LATENT_DIM, np.nan)

        mi_per_fold_train[k] = fm

    variables = list(next(iter(mi_per_fold_train.values())).keys())
    mi_agg     = {v: np.nanmean(np.stack([mi_per_fold_train[k][v] for k in sorted(mi_per_fold_train)],0),0) for v in variables}
    mi_agg_std = {v: np.nanstd( np.stack([mi_per_fold_train[k][v] for k in sorted(mi_per_fold_train)],0),0) for v in variables}

    print("[OK] §5 train-only MI computed")
    for v, arr in mi_agg.items():
        print(f"  {v:12s}: mean={np.nanmean(arr):.4f}  max={np.nanmax(arr):.4f}")

    rows_mi = []
    for k in sorted(mi_per_fold_train):
        for v in variables:
            rows_mi.append({"fold":k,"variable":v,
                            "mi_sum_nats": float(np.nansum(mi_per_fold_train[k][v])),
                            "mi_mean_nats":float(np.nanmean(mi_per_fold_train[k][v])),
                            "mi_max_nats": float(np.nanmax(mi_per_fold_train[k][v])),
                            "data_split":"train_only"})
    save_table(pd.DataFrame(rows_mi), "05_latent_mi_summary_by_fold.csv")
"""))
cells.append(code("""
if not _MI_AVAILABLE:
    print("[SKIP] §5b")
else:
    eps_p = 1e-9
    mi_Y    = mi_agg["Y_AD_CN"]
    mi_site = np.nan_to_num(mi_agg.get("Site",np.zeros(LATENT_DIM)), nan=0.)
    mi_age  = np.nan_to_num(mi_agg.get("Age", np.zeros(LATENT_DIM)), nan=0.)
    mi_sex  = np.nan_to_num(mi_agg.get("Sex", np.zeros(LATENT_DIM)), nan=0.)
    mi_Y_nn = np.nan_to_num(mi_Y, nan=0.)

    # Heuristic marginal diagnostic ratio (renamed from v1/v2 "purity")
    # CAVEAT: marginal, not conditional; not a canonical IT quantity
    heuristic_mdr = mi_Y_nn / (mi_Y_nn + mi_site + mi_age + mi_sex + eps_p)

    logreg_w_folds = {}
    for k, p in fold_logreg_paths.items():
        if p and p.exists():
            try:
                clf = joblib.load(p)
                w = clf.named_steps["model"].coef_[0, :LATENT_DIM]
                logreg_w_folds[k] = w
            except Exception as e:
                print(f"  [WARN] fold {k} weight extraction: {e}")
    w_mean = np.stack(list(logreg_w_folds.values())).mean(0) if logreg_w_folds else np.zeros(LATENT_DIM)
    w_abs  = np.abs(w_mean)

    df_latent = pd.DataFrame({
        "dim": np.arange(LATENT_DIM),
        "mi_Y_train_nats": mi_Y,
        "mi_Y_std":        mi_agg_std["Y_AD_CN"],
        "mi_site_nats":    mi_agg.get("Site", np.full(LATENT_DIM,np.nan)),
        "mi_age_nats":     mi_agg.get("Age",  np.full(LATENT_DIM,np.nan)),
        "mi_sex_nats":     mi_agg.get("Sex",  np.full(LATENT_DIM,np.nan)),
        "heuristic_marginal_diagnostic_ratio": heuristic_mdr,   # renamed in v3
        "w_mean": w_mean, "w_abs": w_abs,
        "kl_mean": kl_mean if "kl_mean" in dir() else np.nan,
    })
    save_table(df_latent, "05_latent_mi_train_only.csv")
    print("[OK] §5b latent stats saved")
    print(f"Top-5 dims by I(z;Y): {np.argsort(-mi_Y_nn)[:5].tolist()}")
    print(f"Top-5 heuristic MDR: {np.argsort(-heuristic_mdr)[:5].tolist()}")
"""))
cells.append(code("""
if not _MI_AVAILABLE:
    print("[SKIP] §5c")
else:
    order_Y = np.argsort(-mi_Y_nn)
    fig, axes = plt.subplots(2,2,figsize=(13,9))
    axes[0,0].bar(np.arange(LATENT_DIM), mi_Y_nn[order_Y], width=1., color="C0", alpha=0.8)
    axes[0,0].set_xlabel("Latent dim (ranked by I(z;Y) - train)"); axes[0,0].set_ylabel("MI (nats)")
    axes[0,0].set_title("Marginal I(z_k; Y) [train set]")
    axes[0,1].bar(np.arange(LATENT_DIM), mi_site[order_Y], width=1., color="C1", alpha=0.8)
    axes[0,1].set_xlabel("Same rank order"); axes[0,1].set_ylabel("MI (nats)")
    axes[0,1].set_title("I(z_k; Site) [train set]")
    axes[1,0].scatter(mi_site, mi_Y_nn, s=10, alpha=0.5)
    top20 = order_Y[:20]
    axes[1,0].scatter(mi_site[top20], mi_Y_nn[top20], s=30, c="red", alpha=0.7, zorder=5, label="Top-20 by I(z;Y)")
    axes[1,0].set_xlabel("I(z_k;Site)"); axes[1,0].set_ylabel("I(z_k;Y)")
    axes[1,0].set_title("Diagnostic vs site information"); axes[1,0].legend(fontsize=8)
    axes[1,1].scatter(w_abs, heuristic_mdr, s=10, alpha=0.5)
    axes[1,1].axhline(0.5,color="gray",lw=1,ls="--")
    axes[1,1].set_xlabel("|w_k| (logistic weight)"); axes[1,1].set_ylabel("Heuristic marginal diagnostic ratio")
    axes[1,1].set_title("Heuristic MDR vs logistic weight\\n[marginal only, not conditional - not canonical IT]")
    fig.suptitle("§5 - Latent MI and Confounding  [train-set MI; marginal, not conditional]", fontsize=11)
    savefig("05_latent_mi_confounding.png", fig)
    print("[OK] §5c plots saved")
"""))

# ── §6 TOP-K READOUT ───────────────────────────────────────────────────────
cells.append(md("""
## §6 — Top-k Readout: Distributed vs Sparse Code  [Nested Clean + Per-Fold Uncertainty]

> **Protocol:**
> 1. **Ranking** is derived from *training-set* MI or logreg weights — never the test set.
> 2. A logistic regression is **fitted on training-set latents** for the top-k dimensions.
> 3. AUC is evaluated on the **held-out test fold** only.
>
> **Why AUC values may be quantized:** With ~37 subjects per test fold and binary outcome,
> AUC values are inherently discretized (e.g., 0.333, 0.5, 0.667, 0.833, 1.0 for n=3
> averaging). This is a consequence of small N_test, **not** a data processing artefact.
> The mean ± bootstrap-CI across folds is shown, but individual-fold uncertainty is high.
>
> **Primary ranking:** `by_w_abs` (logistic weight magnitude from the original pipeline).
>
> **Interpretation:** Evidence is consistent with a **non-sparse / distributed
> representation** — AUC increases gradually with k across all strategies.
> The exact saturation point in k is uncertain given fold-level variance with small N_test.
"""))
cells.append(code("""
if not _REINFERENCE_AVAILABLE or not _MI_AVAILABLE:
    print("[SKIP] §6: requires re-inference and train MI")
elif not any(k in fold_mu_train_dict for k in fold_mu_dict):
    print("[SKIP] §6: no folds with both train and test latents available")
else:
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline as SkPipeline

    K_GRID = [k for k in [1,2,4,8,16,24,32,48,64,96,128,160,192,224,256] if k <= LATENT_DIM]

    rank_strategies = {
        "by_w_abs"       : np.argsort(-w_abs),
        "by_mi_Y_train"  : np.argsort(-mi_Y_nn),
        "by_purity_train": np.argsort(-heuristic_mdr),
    }

    folds_ok = sorted(k for k in fold_mu_train_dict if k in fold_mu_dict
                      and k in fold_y_train_dict and k in fold_y_dict)
    print(f"Folds with both train+test latents: {folds_ok}")

    # Long table: one row per (strategy, k, fold)
    long_rows = []
    for strat, order in rank_strategies.items():
        for topk in K_GRID:
            dims = order[:topk]
            for k in folds_ok:
                Z_tr = fold_mu_train_dict[k][:, dims]; y_tr = fold_y_train_dict[k]
                Z_te = fold_mu_dict[k][:, dims];       y_te = fold_y_dict[k]
                mtr = ~np.isnan(y_tr); mte = ~np.isnan(y_te)
                n_tr = int(mtr.sum()); n_te = int(mte.sum())
                n_ad_te = int((y_te[mte]==1).sum()); n_cn_te = int((y_te[mte]==0).sum())
                if n_tr < 5 or n_te < 5 or len(np.unique(y_tr[mtr]))<2:
                    long_rows.append({"strategy":strat,"k":topk,"fold":k,
                                      "n_train":n_tr,"n_test":n_te,
                                      "n_ad_test":n_ad_te,"n_cn_test":n_cn_te,"auc":np.nan})
                    continue
                Z_tr_c,y_tr_c = Z_tr[mtr],y_tr[mtr].astype(int)
                Z_te_c,y_te_c = Z_te[mte],y_te[mte].astype(int)
                pipe = SkPipeline([("sc", StandardScaler()),
                                   ("lr", LR(max_iter=500, C=1., random_state=SEED))])
                try:
                    pipe.fit(Z_tr_c, y_tr_c)
                    auc_v = roc_auc_score(y_te_c, pipe.predict_proba(Z_te_c)[:,1]) if len(np.unique(y_te_c))>=2 else np.nan
                except Exception:
                    auc_v = np.nan
                long_rows.append({"strategy":strat,"k":topk,"fold":k,
                                  "n_train":n_tr,"n_test":n_te,
                                  "n_ad_test":n_ad_te,"n_cn_test":n_cn_te,"auc":auc_v})

    df_long = pd.DataFrame(long_rows)
    save_table(df_long, "06_topk_auc_by_strategy_by_fold.csv")
    print("[OK] Long table saved")
    print(df_long[df_long["strategy"]=="by_w_abs"].pivot(index="k",columns="fold",values="auc").round(3).to_string())
"""))
cells.append(code("""
if "df_long" not in dir():
    print("[SKIP] §6b: df_long not available")
else:
    # Aggregate: mean + bootstrap CI + std
    results_topk  = {}; results_lo = {}; results_hi = {}; results_std = {}
    for strat in df_long["strategy"].unique():
        sub_s = df_long[df_long["strategy"]==strat]
        means_ = []; los_ = []; his_ = []; stds_ = []
        for topk in K_GRID:
            aucs_ = sub_s[sub_s["k"]==topk]["auc"].dropna().values
            m_ = float(np.mean(aucs_)) if len(aucs_)>0 else np.nan
            s_ = float(np.std(aucs_))  if len(aucs_)>0 else np.nan
            if len(aucs_)>=2:
                lo_, hi_ = bootstrap_ci(aucs_, n_boot=2000)
            else:
                lo_, hi_ = m_, m_
            means_.append(m_); los_.append(lo_); his_.append(hi_); stds_.append(s_)
        results_topk[strat]  = means_
        results_lo[strat]    = los_
        results_hi[strat]    = his_
        results_std[strat]   = stds_

    df_topk = pd.DataFrame({"k": K_GRID, **results_topk})
    save_table(df_topk, "06_topk_auc_by_strategy_nested_clean.csv")

    fig, ax = plt.subplots(figsize=(9,5))
    for strat, aucs_v in results_topk.items():
        lw_ = 2 if strat == "by_w_abs" else 1
        color_ = "C0" if strat=="by_w_abs" else ("C1" if strat=="by_mi_Y_train" else "C2")
        ax.plot(K_GRID, aucs_v, marker="o", ms=4, lw=lw_, label=strat, color=color_)
        ax.fill_between(K_GRID, results_lo[strat], results_hi[strat],
                        alpha=0.15, color=color_)
    ax.set_xscale("log", base=2); ax.set_xlabel("k (top-k latent dims)")
    ax.set_ylabel("ROC-AUC (held-out test)"); ax.axhline(0.5,color="gray",lw=1,ls="--",label="chance")
    ax.set_title(
        "§6 - Top-k readout AUC (mean +/- 95% bootstrap CI across folds)\\n"
        "Ranked on train, evaluated on test. AUC quantization is expected with N_test~37."
    )
    ax.legend(); ax.grid(alpha=0.3)
    savefig("06_topk_readout_auc.png", fig)

    protocol_summary = {
        "ranking_data_split"     : "train_set_only (train_dev_tensor_idx.npy)",
        "evaluation_data_split"  : "held_out_test_fold (test_tensor_idx.npy)",
        "readout_model"          : "LogisticRegression(C=1)",
        "n_folds_used"           : len(folds_ok),
        "ranking_strategies"     : list(results_topk.keys()),
        "k_grid"                 : K_GRID,
        "uncertainty"            : "95% bootstrap CI across folds (2000 resamples)",
        "note_auc_quantization"  : (
            "With N_test~37 per fold, AUC values are inherently discretized. "
            "This is expected and is not a processing artefact. "
            "Mean across folds reduces but does not eliminate quantization."
        ),
        "interpretation"         : (
            "Evidence is consistent with distributed (non-sparse) representation. "
            "Exact saturation point is uncertain due to fold-level AUC variance with small N_test."
        ),
    }
    with open(OUTPUT_DIR / "Tables" / "06_topk_protocol_summary.json", "w") as f:
        json.dump(protocol_summary, f, indent=2)
    print("[OK] §6 done")
    print(df_topk.round(3).to_string(index=False))
"""))

# ── §7 SCORE-LEVEL IT ──────────────────────────────────────────────────────
cells.append(md(r"""
## §7 — Score-Level Information Theory and Calibration

### Explicit derivation: why the theoretical entropy curve applies only in Panel A

The logistic regression classifier defines the probability:
$$p_\text{raw}(y=1\mid z) = \sigma(w^\top z + b)$$
The **raw linear margin** is uniquely determined by $z$:
$$m_\text{raw} = w^\top z + b = \text{logit}(p_\text{raw})$$
The **predictive entropy** in bits is:
$$H_\text{raw}(Y\mid z) = -p_\text{raw}\log_2 p_\text{raw} - (1-p_\text{raw})\log_2(1-p_\text{raw})
  = h_2\!\left(\sigma(m_\text{raw})\right)$$
This is a **deterministic identity**: for any uncalibrated logistic regression, the entropy
is a fixed function of the margin $m_\text{raw}$.  Plotting $H$ vs $m_\text{raw}$ must
exactly follow the theoretical curve $h_2(\sigma(\cdot))$ up to numerical precision.

**After calibration**, $p_\text{cal}$ is obtained via a separate mapping (Platt / isotonic).
The calibrated logit $s_\text{cal} = \text{logit}(p_\text{cal}) \neq m_\text{raw}$.
The identity $H = h_2(\sigma(\cdot))$ does **not** hold for $s_\text{cal}$, so no
theoretical curve is overlaid in Panel B.

| Score | Definition | Entropy formula | Theoretical curve |
|-------|-----------|-----------------|-------------------|
| $m_\text{raw}$ | $\text{logit}(p_\text{raw})$ | $h_2(\sigma(m_\text{raw}))$ — exact | **Yes** (Panel A) |
| $s_\text{cal}$ | $\text{logit}(p_\text{cal})$ | No closed form after calibration | **No** (Panel B) |
"""))
cells.append(code("""
preds_dfs = [pd.read_csv(p).assign(fold=k) for k,p in fold_pred_paths.items() if p and p.exists()]
_PREDS_AVAILABLE = bool(preds_dfs)
if not _PREDS_AVAILABLE:
    print("[SKIP] §7: no prediction CSVs found")
else:
    preds_all = pd.concat(preds_dfs, ignore_index=True)
    print(f"[OK] predictions: {len(preds_all)} rows  fold distribution:")
    print(dict(preds_all.groupby("fold")["y_true"].value_counts()))
"""))
cells.append(code("""
if not _PREDS_AVAILABLE:
    print("[SKIP] §7b")
else:
    EPS = 1e-9
    p_raw  = np.clip(preds_all["y_score_raw"].values.astype(float), EPS, 1-EPS)
    p_cal  = np.clip(preds_all["y_score_cal"].values.astype(float), EPS, 1-EPS)
    y_true = preds_all["y_true"].values.astype(float)

    m_raw = np.log(p_raw/(1-p_raw))   # logit(p_raw) = raw linear margin
    s_cal = np.log(p_cal/(1-p_cal))   # logit(p_cal)
    H_raw = bin_entropy_bits(p_raw)   # bits
    H_cal = bin_entropy_bits(p_cal)   # bits
    group = np.where(y_true==1, "AD", "CN")

    preds_all["m_raw"] = m_raw; preds_all["s_cal"] = s_cal
    preds_all["H_raw"] = H_raw; preds_all["H_cal"] = H_cal

    rows_met = []
    for k in sorted(preds_all["fold"].unique()):
        sub = preds_all[preds_all["fold"]==k]
        yt  = sub["y_true"].values.astype(float)
        ypr = sub["y_score_raw"].values; ypc = sub["y_score_cal"].values
        try: auc_raw = roc_auc_score(yt, ypr)
        except: auc_raw = np.nan
        try: auc_cal = roc_auc_score(yt, ypc)
        except: auc_cal = np.nan
        rows_met.append({"fold":k, "n_total":len(yt),
                         "n_AD":int((yt==1).sum()), "n_CN":int((yt==0).sum()),
                         "auc_raw":auc_raw, "auc_cal":auc_cal, **ece_brier(yt, ypc)})
    df_metrics = pd.DataFrame(rows_met)
    print("Per-fold calibration metrics:")
    print(df_metrics.round(4).to_string(index=False))
    save_table(df_metrics, "07_per_fold_calibration_metrics.csv")
    save_latex(
        df_metrics.round(4),
        "07_per_fold_calibration_metrics.csv",
        caption="Per-fold classification and calibration metrics. "
                "AUC: area under ROC curve. ECE: expected calibration error. "
                "Brier: Brier score. log\\_loss: cross-entropy loss.",
        label="tab:cal_metrics"
    )

    # Entropy summary per (fold, group, score_type)
    ent_rows = []
    for k in sorted(preds_all["fold"].unique()):
        sub_k = preds_all[preds_all["fold"]==k]
        for grp in ["AD","CN"]:
            idx_ = sub_k["y_true"]==(1 if grp=="AD" else 0)
            for score, col in [("raw","H_raw"),("calibrated","H_cal")]:
                vals_ = sub_k.loc[idx_, col].values
                ent_rows.append({"fold":k,"group":grp,"score_type":score,
                                 "H_mean_bits":round(float(vals_.mean()),4),
                                 "H_std_bits": round(float(vals_.std()),4),
                                 "n": int(idx_.sum())})
    df_ent = pd.DataFrame(ent_rows)
    save_table(df_ent, "07_entropy_margin_summary.csv")
    print("[OK] §7b done")
"""))
cells.append(code(r"""
if not _PREDS_AVAILABLE:
    print("[SKIP] §7c")
else:
    m_theory = np.linspace(-6, 6, 400)
    H_theory = bin_entropy_bits(sigmoid(m_theory))

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Panel A: H vs m_raw  (with theoretical curve)
    for lbl, color in PALETTE.items():
        if lbl == "MCI": continue
        idx_ = group == lbl
        axes[0,0].scatter(m_raw[idx_], H_raw[idx_], s=8, alpha=0.35, color=color, label=lbl)
    axes[0,0].plot(m_theory, H_theory, "k-", lw=2, label="h2(sigma(m))", zorder=5)
    axes[0,0].set_xlabel("Raw linear margin m_raw = logit(p_raw)")
    axes[0,0].set_ylabel("H(Y|z) (bits)")
    axes[0,0].set_title("Panel A - Raw margin vs entropy\\n[theoretical curve: exact identity for uncalibrated logreg]")
    axes[0,0].legend(fontsize=8)

    # Panel B: H vs s_cal  (no theoretical curve)
    for lbl, color in PALETTE.items():
        if lbl == "MCI": continue
        idx_ = group == lbl
        axes[0,1].scatter(s_cal[idx_], H_cal[idx_], s=8, alpha=0.35, color=color, label=lbl)
    axes[0,1].set_xlabel("Calibrated logit s_cal = logit(p_cal)")
    axes[0,1].set_ylabel("H(Y|z) (bits)")
    axes[0,1].set_title("Panel B - Calibrated logit vs entropy\\n[no theoretical curve: calibration breaks logistic identity]")
    axes[0,1].text(0.02, 0.97, "No theoretical curve:\\ncalibration breaks h2(sigma(m)) identity",
                   transform=axes[0,1].transAxes, va="top", fontsize=7, color="gray")
    axes[0,1].legend(fontsize=8)

    # Reliability diagram
    pt_, pp_ = calibration_curve(y_true.astype(int), p_cal, n_bins=10, strategy="uniform")
    axes[0,2].plot([0,1],[0,1],"k--",lw=1,label="Perfect")
    axes[0,2].plot(pp_, pt_, "o-", color="C0", label="Calibrated logreg")
    axes[0,2].set_xlabel("Mean predicted probability"); axes[0,2].set_ylabel("Fraction AD")
    axes[0,2].set_title("Reliability diagram (pooled OOF)"); axes[0,2].legend()

    # Margin distributions
    for lbl, color in PALETTE.items():
        if lbl == "MCI": continue
        idx_ = group == lbl
        axes[1,0].hist(m_raw[idx_], bins=40, density=True, alpha=0.6, color=color, label=lbl)
    axes[1,0].set_xlabel("m_raw"); axes[1,0].set_ylabel("Density")
    axes[1,0].set_title("Raw margin distribution"); axes[1,0].legend()

    # Entropy by group (violin) with n_AD/n_CN annotations
    _n_cn_s7 = int((group=="CN").sum()); _n_ad_s7 = int((group=="AD").sum())
    data_v = [H_cal[group=="CN"], H_cal[group=="AD"]]
    vp = axes[1,1].violinplot(data_v, positions=[0,1], showmedians=True)
    for i, (pc, lbl) in enumerate(zip(vp["bodies"], ["CN","AD"])):
        pc.set_facecolor(PALETTE[lbl]); pc.set_alpha(0.6)
    axes[1,1].set_xticks([0,1])
    axes[1,1].set_xticklabels([f"CN\\n(n={_n_cn_s7})", f"AD\\n(n={_n_ad_s7})"])
    axes[1,1].set_ylabel("H_cal (bits)")
    axes[1,1].set_title("Predictive entropy by group (calibrated)\\n[OOF pooled]")

    # Per-fold AUC with summary textbox
    _mean_auc = df_metrics["auc_cal"].mean(); _mean_ece = df_metrics["ece"].mean()
    axes[1,2].plot(df_metrics["fold"], df_metrics["auc_cal"],"o-",label="AUC (cal)")
    axes[1,2].plot(df_metrics["fold"], df_metrics["ece"],"s--",color="C3",label="ECE (cal)")
    axes[1,2].axhline(_mean_auc, ls=":", color="C0", lw=1)
    axes[1,2].text(0.98, 0.98,
                   f"mean AUC={_mean_auc:.3f}\\nmean ECE={_mean_ece:.3f}",
                   transform=axes[1,2].transAxes, va="top", ha="right",
                   fontsize=8, bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    axes[1,2].set_xlabel("Fold"); axes[1,2].set_title("Per-fold AUC & ECE"); axes[1,2].legend()

    fig.suptitle("§7 — Score-Level IT & Calibration\\n"
                 "Panels A/B distinguish raw margin (with h2 identity) from calibrated logit", fontsize=11)
    savefig("07_score_it_calibration.png", fig)

    cols_out = [c for c in ["fold","SubjectID","tensor_idx","y_true",
                             "y_score_raw","y_score_cal","m_raw","s_cal",
                             "H_raw","H_cal","y_pred"] if c in preds_all.columns]
    save_table(preds_all[cols_out], "07_score_level_entropy.csv")
    print("[OK] §7 done")
"""))

# ── §8 ISING SCAFFOLD ──────────────────────────────────────────────────────
cells.append(md("""
## §8 — Pairwise Maximum-Entropy / Ising Scaffold  [Exploratory]

> **v3 fix:** Previous v2 reported N=14 subjects due to a label-alignment bug.
> `fold_subject_idx_dict` stored local pool indices instead of global tensor indices,
> so label lookup in prediction CSVs matched only ~2/37 subjects per fold by chance.
> v3 corrects this: `fold_tensor_idx_dict` stores `test_tensor_idx.npy` (global indices),
> giving full alignment with prediction CSVs → expected N=183.
>
> **Label source of truth:** AD/CN membership is taken from per-fold prediction CSVs,
> cross-validated against the global tensor indices stored in `fold_tensor_idx_dict`.
>
> **Scientific caveat:** Empirical second-order structure (non-zero $J_{ij}$) motivates,
> but does **not** yet prove, the benefit of pairwise MaxEnt or interaction-aware
> classifiers.  A full Ising fit (pseudo-likelihood or MPF) would be required.
"""))
cells.append(code("""
if not DO_MAXENT or not _REINFERENCE_AVAILABLE or not _MI_AVAILABLE:
    print("[SKIP] §8")
else:
    # ── Pooling audit: verify alignment before computing Ising quantities ────
    audit_rows = []
    Z_parts, y_parts = [], []

    for k in sorted(fold_mu_dict):
        tidx_k = fold_tensor_idx_dict.get(k, np.array([]))  # global tensor indices (v3)
        Z_k    = fold_mu_dict[k]

        # Labels from prediction CSV using global tensor_idx
        lbl_map = {}
        if fold_pred_paths.get(k) and fold_pred_paths[k].exists():
            df_p = pd.read_csv(fold_pred_paths[k])
            lbl_map = dict(zip(df_p["tensor_idx"].astype(int), df_p["y_true"].astype(float)))

        y_k = np.array([lbl_map.get(i, np.nan) for i in tidx_k])
        n_nan_k = int(np.isnan(y_k).sum())
        n_ad_k  = int((y_k==1).sum()); n_cn_k = int((y_k==0).sum())

        # Verify: tensor_idx from re-inference should exactly match prediction CSV
        pred_tidx_k = set(lbl_map.keys())
        reinf_tidx_k = set(tidx_k.tolist())
        n_match = len(reinf_tidx_k & pred_tidx_k)

        audit_rows.append({
            "fold": k,
            "n_subjects": len(tidx_k),
            "n_AD": n_ad_k, "n_CN": n_cn_k, "n_nan_y": n_nan_k,
            "n_tensor_idx_match_pred": n_match,
            "perfect_alignment": (n_match == len(tidx_k)) and (n_nan_k == 0),
        })
        Z_parts.append(Z_k)
        y_parts.append(y_k)

    df_audit = pd.DataFrame(audit_rows)
    save_table(df_audit, "08_ising_pooling_audit.csv")

    print("\\n=== §8 POOLING AUDIT ===")
    print(df_audit.to_string(index=False))
    print(f"\\nAll folds perfectly aligned: {df_audit['perfect_alignment'].all()}")

    Z_pool     = np.concatenate(Z_parts, 0)
    y_pool_raw = np.concatenate(y_parts, 0)
    mask_      = ~np.isnan(y_pool_raw)
    n_discarded = int((~mask_).sum())
    Z_pool, y_pool = Z_pool[mask_], y_pool_raw[mask_].astype(int)

    print(f"\\nPooled: N={len(y_pool)}  AD={int((y_pool==1).sum())}  CN={int((y_pool==0).sum())}")
    print(f"Discarded (NaN label): {n_discarded}")
    if len(y_pool) < 50:
        print("[WARN] Pooled N is unexpectedly small — check alignment above")
"""))
cells.append(code("""
if "y_pool" not in dir():
    print("[SKIP] §8b: pooling did not complete")
elif len(y_pool) < 10:
    print(f"[SKIP] §8b: too few pooled subjects ({len(y_pool)})")
else:
    TOP_ISING = 12
    order_ising = np.argsort(-mi_Y_nn); ising_dims = order_ising[:TOP_ISING]

    Z_sel = Z_pool[:, ising_dims]
    S = np.sign(Z_sel - np.median(Z_sel, axis=0)); S[S==0] = 1

    S_AD = S[y_pool==1]; S_CN = S[y_pool==0]
    h_AD = S_AD.mean(0); h_CN = S_CN.mean(0)
    J_AD = (S_AD.T@S_AD)/len(S_AD); J_CN = (S_CN.T@S_CN)/len(S_CN)
    np.fill_diagonal(J_AD,0.); np.fill_diagonal(J_CN,0.)

    save_table(pd.DataFrame([{
        "N_total":len(y_pool),"N_AD":int((y_pool==1).sum()),"N_CN":int((y_pool==0).sum()),
        "n_dims_selected":TOP_ISING,"dims":str(ising_dims.tolist()),
        "label_source":"fold_prediction_csv (global tensor_idx aligned)",
        "v3_fix":"correct global tensor_idx via fold_tensor_idx_dict",
    }]), "08_ising_counts_summary.csv")

    fig, axes = plt.subplots(1,3,figsize=(14,4))
    axes[0].bar(np.arange(TOP_ISING)-0.2,h_AD,0.4,label="AD",color=PALETTE["AD"],alpha=0.8)
    axes[0].bar(np.arange(TOP_ISING)+0.2,h_CN,0.4,label="CN",color=PALETTE["CN"],alpha=0.8)
    axes[0].set_xlabel("Ising dim index"); axes[0].set_ylabel("mean(s_i)")
    axes[0].set_title(f"Empirical first moments (fields)\\nN_AD={int((y_pool==1).sum())} N_CN={int((y_pool==0).sum())}")
    axes[0].legend()
    lim_ = max(abs(J_AD).max(),abs(J_CN).max(),0.01)
    axes[1].imshow(J_AD,cmap="RdBu_r",vmin=-lim_,vmax=lim_)
    axes[1].set_title("Empirical <s_i s_j> - AD"); axes[1].axis("off")
    Jd = J_AD-J_CN
    im_=axes[2].imshow(Jd,cmap="RdBu_r",vmin=-abs(Jd).max(),vmax=abs(Jd).max())
    axes[2].set_title("Delta <s_i s_j> = AD - CN"); axes[2].axis("off")
    plt.colorbar(im_,ax=axes[2],fraction=0.046)
    fig.suptitle("§8 - Ising scaffold [EXPLORATORY]  second-order moments motivate, but do not prove, MaxEnt benefit",
                 fontsize=9)
    savefig("08_ising_empirical_moments.png", fig)

    save_table(pd.DataFrame({"dim":ising_dims,"h_AD":h_AD,"h_CN":h_CN,"h_diff":h_AD-h_CN}),
               "08_ising_empirical_fields.csv")
    print("[OK] §8 scaffold done")
"""))

# ── §9 SPECTRAL ORDINAL COMPLEXITY PROXY ──────────────────────────────────
cells.append(md("""
## §9 — Spectral Ordinal Complexity Proxy  [EXPLORATORY — Do Not Cite as Temporal BP]

> **Warning — not standard Bandt-Pompe:** The current tensor contains pre-computed FC
> matrices, **not raw BOLD time series**.  Standard Bandt-Pompe analysis requires the
> original signal $x(t) \\in \\mathbb{R}^T$ per ROI.
>
> **What this section does:** We apply the permutation entropy estimator to the
> **sorted eigenvalue spectrum** $\\lambda_1 \\ge \\cdots \\ge \\lambda_{131}$ of each
> FC matrix.  Since eigenvalue ordering is by magnitude, **not by time**, this measures
> *ordinal spectral heterogeneity* — how concentrated vs diffuse the spectral energy is —
> which is **distinct from temporal complexity or causality**.
>
> This section is clearly separated from main claims and labelled exploratory.
> Do not cite or interpret as temporal complexity.
"""))
cells.append(code("""
if not DO_TIME_SERIES or global_tensor_path is None:
    print("[SKIP] §9")
else:
    if "GLOBAL_TENSOR" not in dir():
        _td2 = np.load(global_tensor_path)
        _k2  = next((k for k in _td2.files if "tensor" in k.lower()), _td2.files[0])
        GLOBAL_TENSOR = _td2[_k2].astype(np.float32)
    T3BP = GLOBAL_TENSOR[:, CHANNELS_IDX, :, :]
    N_,C_,R_,_ = T3BP.shape; D_BP=5; TAU_BP=1
    print(f"Spectral ordinal proxy: N={N_},C={C_},R={R_}, D={D_BP},tau={TAU_BP}")

    sub_map_bp = {}
    if meta_df is not None and "tensor_idx" in meta_df.columns:
        _grp_col_bp = next((c for c in ["ResearchGroup_Mapped","Group","ResearchGroup"]
                            if c in meta_df.columns), None)
        if _grp_col_bp:
            sub_map_bp = dict(zip(meta_df["tensor_idx"].dropna().astype(int),
                                  meta_df[_grp_col_bp].values))

    rows_bp = []
    for n in range(N_):
        for c in range(C_):
            ev = np.sort(np.linalg.eigvalsh(T3BP[n,c].astype(float)))[::-1]
            H_, Cmp_ = spectral_complexity(ev, D=D_BP, tau=TAU_BP)
            rows_bp.append({"subject_idx":n,"channel":CHANNEL_NAMES[c],"H_BP":H_,"C_BP":Cmp_,
                            "group":sub_map_bp.get(n,"?")})
    df_bp = pd.DataFrame(rows_bp)
    save_table(df_bp, "09_spectral_ordinal_complexity_proxy.csv")

    fig, axes = plt.subplots(1,C_,figsize=(5*C_,4))
    if C_==1: axes=[axes]
    for ci,ch in enumerate(CHANNEL_NAMES):
        ax=axes[ci]; sub=df_bp[df_bp["channel"]==ch]
        for grp,color in [("CN",PALETTE["CN"]),("AD",PALETTE["AD"])]:
            g_=sub[sub["group"].str.upper()==grp] if sub["group"].dtype==object else sub[sub["group"]==grp]
            if len(g_)>0: ax.scatter(g_["H_BP"],g_["C_BP"],s=15,alpha=0.5,color=color,label=grp)
        ax.set_xlabel("H_BP (ordinal spectral entropy)"); ax.set_ylabel("C (complexity)")
        ax.set_title(f"{ch}\\n[EXPLORATORY - eigenvalue proxy, NOT temporal BP]"); ax.legend(fontsize=8)
    fig.suptitle(
        "§9 - Spectral Ordinal Complexity Proxy  [EXPLORATORY]\\n"
        "Eigenvalues ordered by magnitude, NOT time.  Do not interpret as temporal BP.",
        fontsize=9
    )
    savefig("09_spectral_ordinal_complexity_proxy.png", fig)
    print("[OK] §9 spectral ordinal proxy saved")
"""))

# ── §10 VON NEUMANN ENTROPY ────────────────────────────────────────────────
cells.append(md("""
## §10 — Von Neumann Entropy of FC Matrices

### Version-history note for §10

| Version | Change |
|---------|--------|
| v1 | Effect size formula `r_rb = U/(n1*n2)` — **incorrect** (this is CL effect size θ, not r_rb) |
| v2 | Corrected formula: `r_rb = 2*U/(n1*n2) - 1`; added Bonferroni across 3 channels; added `common_lang_effect_size_theta` |
| v2 | Group labels fixed: `ResearchGroup_Mapped` (was using wrong column or finding no match) |
| v3 | MCI descriptive subsection added (using pipeline_log for SubjectID→tensor_idx mapping) |
| v3 | Subject mapping corrected: N_AD=95, N_CN=89 (from pipeline log); v2 had N_AD=94 from fold files only |
| v4 | §10 narrative updated: MI_KNN_Symmetric shows significant AD–CN difference (p_bonf≈0.006); |
|    | earlier drafts incorrectly described all channels as null. |

### Effect sizes

The Mann-Whitney U statistic $U \\in [0, n_1 n_2]$:
- **Common-language effect size:** $\\theta = U / (n_1 n_2)$ — probability that a random AD
  subject has higher $S$ than a random CN subject
- **Rank-biserial correlation:** $r_{rb} = 2\\theta - 1 = 2U/(n_1 n_2) - 1 \\in [-1,1]$

Bonferroni correction is applied across the three FC channels.

### Interpretation (data-driven — see §10b table for exact values)

The corrected v3/v4 results show a **channel-specific pattern**:

- **Pearson-based channels** (Pearson_Full and Pearson_OMST): Bonferroni-corrected p-values
  exceed 0.05; effect sizes are small and consistent with no detectable global spectral-entropy
  difference in linear FC at this coarse scale.
- **MI_KNN_Symmetric**: Bonferroni-corrected p-value < 0.05; modest negative rank-biserial
  correlation (CN > AD in VNE), indicating a detectable reduction in spectral entropy for AD
  relative to CN in the nonlinear MI channel.

**Important note on scope:** VNE is a single global scalar summary per matrix.
"No significant difference in Pearson channels" does **not** imply biological equivalence
of AD and CN FC spectra; the analysis is underpowered for detecting subtle channel-specific
differences with N≈90–95 per group.
The difference between "null in Pearson channels" and "significant in MI_KNN channel" is
consistent with the hypothesis that MI-based FC captures complementary structure not present
in linear correlation, but this is a post-hoc observation and should not be overclaimed.

> **Important**: The exact p-values and effect sizes are computed from the data in §10b and
> printed there. Do not treat any specific number in this text as authoritative; consult
> the §10b table and `10_von_neumann_group_stats_corrected.csv` directly.
"""))
cells.append(code("""
if not DO_VON_NEUMANN or global_tensor_path is None:
    print("[SKIP] §10")
else:
    if "GLOBAL_TENSOR" not in dir():
        _td3 = np.load(global_tensor_path)
        _k3  = next((k for k in _td3.files if "tensor" in k.lower()), _td3.files[0])
        GLOBAL_TENSOR = _td3[_k3].astype(np.float32)
    T3VN = GLOBAL_TENSOR[:, CHANNELS_IDX, :, :]
    N_,C_,R_,_ = T3VN.shape; S_MAX = float(np.log(R_))
    print(f"Von Neumann: N={N_}, C={C_}, R={R_}, S_max={S_MAX:.4f} nats")

    sub_map_vn = {}
    if meta_df is not None and "tensor_idx" in meta_df.columns:
        _grp_col_vn = next((c for c in ["ResearchGroup_Mapped","Group","ResearchGroup"]
                            if c in meta_df.columns), None)
        if _grp_col_vn:
            sub_map_vn = dict(zip(meta_df["tensor_idx"].dropna().astype(int),
                                  meta_df[_grp_col_vn].values))
            print(f"Group mapping built: {len(sub_map_vn)} subjects")

    rows_vn = []
    for n in range(N_):
        for c in range(C_):
            S_ = von_neumann_entropy(T3VN[n,c].astype(float))
            rows_vn.append({"subject_idx":n,"channel":CHANNEL_NAMES[c],
                            "S_von_neumann":S_,"S_norm":S_/S_MAX if not np.isnan(S_) else np.nan,
                            "group":sub_map_vn.get(n,"?")})
    df_vn = pd.DataFrame(rows_vn)
    save_table(df_vn, "10_von_neumann_entropy.csv")
"""))
cells.append(code("""
if "df_vn" not in dir():
    print("[SKIP] §10b")
else:
    N_TESTS = C_   # number of channels tested (Bonferroni denominator)
    stat_rows = []
    for ch in CHANNEL_NAMES:
        sub_  = df_vn[df_vn["channel"]==ch]
        ad_s  = sub_[sub_["group"].str.upper()=="AD"]["S_von_neumann"].dropna().values
        cn_s  = sub_[sub_["group"].str.upper()=="CN"]["S_von_neumann"].dropna().values
        if len(ad_s)>1 and len(cn_s)>1:
            U_, pval_ = mannwhitneyu(ad_s, cn_s, alternative="two-sided")
            n1, n2 = len(ad_s), len(cn_s)
            theta_cl = float(U_/(n1*n2))          # common-language effect size
            r_rb     = float(2*U_/(n1*n2) - 1)    # rank-biserial (corrected v2)
            p_bonf   = float(min(1., pval_*N_TESTS))
        else:
            U_=pval_=theta_cl=r_rb=p_bonf=np.nan; n1=len(ad_s); n2=len(cn_s)
        stat_rows.append({
            "channel":ch,
            "N_AD":n1,"N_CN":n2,
            "mean_S_AD":float(np.mean(ad_s)) if len(ad_s) else np.nan,
            "mean_S_CN":float(np.mean(cn_s)) if len(cn_s) else np.nan,
            "std_S_AD": float(np.std(ad_s))  if len(ad_s) else np.nan,
            "std_S_CN": float(np.std(cn_s))  if len(cn_s) else np.nan,
            "delta_S_AD_minus_CN":float(np.mean(ad_s)-np.mean(cn_s)) if (len(ad_s)and len(cn_s)) else np.nan,
            "MWU_U":float(U_),
            "p_value_uncorrected":float(pval_),
            "p_value_bonferroni":float(p_bonf),
            "common_lang_effect_size_theta":float(theta_cl),
            "rank_biserial_r_rb":float(r_rb),
            "S_max_nats":S_MAX,
        })
    df_vn_stats = pd.DataFrame(stat_rows)
    print("Von Neumann group comparison (AD vs CN, Bonferroni corrected):")
    print(df_vn_stats[["channel","N_AD","N_CN","mean_S_AD","mean_S_CN",
                        "p_value_uncorrected","p_value_bonferroni","rank_biserial_r_rb"]].round(5).to_string(index=False))
    save_table(df_vn_stats, "10_von_neumann_group_stats_corrected.csv")

    fig, axes = plt.subplots(1,C_,figsize=(5*C_,4))
    if C_==1: axes=[axes]
    for ci,ch in enumerate(CHANNEL_NAMES):
        ax=axes[ci]; sub_vn=df_vn[df_vn["channel"]==ch]
        known=[g for g in sub_vn["group"].str.upper().unique() if g in ("AD","CN")]
        data_p=[sub_vn[sub_vn["group"].str.upper()==g]["S_von_neumann"].dropna().values for g in ["CN","AD"] if g in known]
        lbl_p=[g for g in ["CN","AD"] if g in known]
        if data_p:
            bp_=ax.boxplot(data_p,labels=lbl_p,patch_artist=True,notch=False)
            for patch,lbl in zip(bp_["boxes"],lbl_p):
                patch.set_facecolor(PALETTE[lbl]); patch.set_alpha(0.7)
        ax.axhline(S_MAX,color="gray",lw=1,ls="--",label=f"S_max={S_MAX:.2f} nats")
        row_=df_vn_stats[df_vn_stats["channel"]==ch].iloc[0]
        _sig_marker = "*" if row_["p_value_bonferroni"] < 0.05 else "ns"
        ax.set_title(
            f"{ch}\\n"
            f"N_AD={int(row_['N_AD'])} N_CN={int(row_['N_CN'])}\\n"
            f"p_bonf={row_['p_value_bonferroni']:.4f} {_sig_marker}  r_rb={row_['rank_biserial_r_rb']:.3f}"
        )
        ax.set_ylabel("S(\\u03c1) (nats)"); ax.legend(fontsize=7)
    # Data-driven suptitle: do not hardcode significance claim
    _n_sig = int((df_vn_stats["p_value_bonferroni"] < 0.05).sum())
    _sig_channels = df_vn_stats.loc[df_vn_stats["p_value_bonferroni"]<0.05,"channel"].tolist()
    _null_channels = df_vn_stats.loc[df_vn_stats["p_value_bonferroni"]>=0.05,"channel"].tolist()
    if _n_sig == 0:
        _interp = "No significant AD-CN difference after Bonferroni correction (all p_bonf > 0.05)"
    elif _n_sig == C_:
        _interp = f"All {C_} channels significant after Bonferroni"
    else:
        _sig_short = [c.split("_")[0] for c in _sig_channels]
        _interp = f"Significant (Bonferroni): {', '.join(_sig_short)}"
    fig.suptitle(
        "§10 — Von Neumann Entropy  [Bonferroni corrected; r_rb = 2U/(n1*n2)-1]\\n" + _interp,
        fontsize=9
    )
    savefig("10_von_neumann_entropy_by_group.png", fig)
    save_latex(
        df_vn_stats[["channel","N_AD","N_CN","mean_S_AD","mean_S_CN",
                     "p_value_uncorrected","p_value_bonferroni","rank_biserial_r_rb"]].round(5),
        "10_von_neumann_group_stats_corrected.csv",
        caption="Von Neumann entropy: AD vs CN (Mann-Whitney U, Bonferroni corrected across 3 channels). "
                "S(\\\\rho) in nats; $r_{rb} = 2U/(n_1 n_2)-1$.",
        label="tab:vne_group_stats"
    )
    print("[OK] §10b done")
    print(f"  Significant channels (p_bonf<0.05): {_sig_channels or 'none'}")
    print(f"  Null channels (p_bonf>=0.05): {_null_channels}")
"""))

# §10 MCI descriptive
cells.append(md("""
### §10c — MCI Descriptive Analysis [Descriptive Only — No Inferential Claims]

The metadata includes MCI subjects (N_MCI≈247 in the full dataset).  The main inferential
comparison is AD vs CN only (the pipeline's design population).  MCI subjects are included
here **descriptively** to contextualise the §10b AD vs CN findings and to show where MCI
falls on the VNE spectrum.

The SubjectID→tensor_idx mapping is obtained from the pipeline log (row position in the
global tensor), which covers all 431 subjects including MCI.  No inferential statistics
are computed for MCI vs AD or MCI vs CN in this analysis.

> **Note on scope:** MCI descriptives provide context for the channel-specific AD–CN
> pattern found in §10b.  MCI falling near CN in the MI_KNN_Symmetric channel is noted
> purely descriptively; no inferential claim is made for the MCI group.
"""))
cells.append(code("""
if "df_vn" not in dir():
    print("[SKIP] §10c")
elif not tidx_map:
    print("[SKIP] §10c: no tensor_idx map available")
else:
    # Rebuild with 3-group labels using pipeline_log mapping
    meta_df_all = None
    if meta_path.exists():
        meta_df_all = pd.read_csv(meta_path)
        meta_df_all["tensor_idx"] = meta_df_all["SubjectID"].map(tidx_map)
        _grp_col_all = next((c for c in ["ResearchGroup_Mapped","Group","ResearchGroup"]
                             if c in meta_df_all.columns), None)
        if _grp_col_all:
            sub_map_all = dict(zip(meta_df_all["tensor_idx"].dropna().astype(int),
                                   meta_df_all[_grp_col_all].values))
            print(f"All-group mapping: {len(sub_map_all)} subjects")
            print(f"Group distribution: {dict(meta_df_all[_grp_col_all].value_counts())}")
        else:
            sub_map_all = {}

    if not sub_map_all:
        print("[SKIP] §10c: no group mapping")
    else:
        # Tag df_vn with 3-group label
        df_vn_all = df_vn.copy()
        df_vn_all["group_all"] = df_vn_all["subject_idx"].map(sub_map_all).fillna("?")

        desc_rows = []
        for ch in CHANNEL_NAMES:
            sub_ = df_vn_all[df_vn_all["channel"]==ch]
            for grp in ["AD","CN","MCI"]:
                vals_ = sub_[sub_["group_all"].str.upper()==grp]["S_von_neumann"].dropna().values
                if len(vals_)==0: continue
                desc_rows.append({
                    "channel":ch,"group":grp,"n":len(vals_),
                    "mean_S":round(float(vals_.mean()),4),
                    "std_S": round(float(vals_.std()),4),
                    "median_S":round(float(np.median(vals_)),4),
                    "note":"descriptive only - no inferential claim for MCI"
                })
        df_desc = pd.DataFrame(desc_rows)
        save_table(df_desc, "10_von_neumann_group_descriptives_all_groups.csv")
        print("All-group VNE descriptives:")
        print(df_desc.to_string(index=False))

        # Descriptive figure
        fig, axes = plt.subplots(1,C_,figsize=(5*C_,4))
        if C_==1: axes=[axes]
        for ci,ch in enumerate(CHANNEL_NAMES):
            ax=axes[ci]; sub_ch=df_vn_all[df_vn_all["channel"]==ch]
            grp_order = ["CN","AD","MCI"]
            data_p=[sub_ch[sub_ch["group_all"].str.upper()==g]["S_von_neumann"].dropna().values
                    for g in grp_order]
            labels_p=grp_order
            data_p_nz = [(d,l) for d,l in zip(data_p,labels_p) if len(d)>0]
            if data_p_nz:
                bp_=ax.boxplot([x[0] for x in data_p_nz],labels=[x[1] for x in data_p_nz],
                               patch_artist=True,notch=False)
                for patch,(_,lbl) in zip(bp_["boxes"],data_p_nz):
                    patch.set_facecolor(PALETTE.get(lbl,"gray")); patch.set_alpha(0.7)
            ax.set_title(f"{ch}"); ax.set_ylabel("S(rho) nats")
        fig.suptitle(
            "§10c - Von Neumann Entropy: AD / CN / MCI descriptive\\n"
            "[MCI is descriptive only; main inferential test is AD vs CN]",
            fontsize=9
        )
        savefig("10_von_neumann_all_groups_descriptive.png", fig)
        print("[OK] §10c done")
"""))

# ── §11 SUMMARY ────────────────────────────────────────────────────────────
cells.append(md("""
## §11 — Execution Report & Outputs

> **The execution report below is data-driven.** Section status is determined by what
> was actually computed in this run, not by what was expected.  Consult the printed
> report cell for the exact run-mode and section-by-section status.

### Authoritative findings (depends on run status — see §11 code cell)

| Finding | Section | Output | Status |
|---------|---------|--------|--------|
| Encoder produces **distributed** (not sparse) representation | §4 | `04_ib_fold_summary.csv` | requires full_recompute or cached artifact |
| Per-dim KL; all 256 active units at ε=0.01 | §4 | `04_kl_per_dimension.csv` | requires full_recompute or cached artifact |
| Marginal $I(z_k;Y)$ ranked on training set [v3/v4 correct subjects] | §5 | `05_latent_mi_train_only.csv` | requires full_recompute or cached artifact |
| Top-k AUC with per-fold long table + 95% bootstrap CI | §6 | `06_topk_auc_by_strategy_by_fold.csv` | requires full_recompute or cached artifact |
| Per-fold AUC, ECE, Brier, log-loss | §7 | `07_per_fold_calibration_metrics.csv` | from fold predictions |
| Raw margin vs calibrated logit entropy (with explicit derivation) | §7 | `07_score_it_calibration.png` | from fold predictions |
| §8 pooling audit: N=183 (94 AD, 89 CN), all 5 folds perfect alignment | §8 | `08_ising_pooling_audit.csv` | requires full_recompute or cached artifact |
| VNE: MI_KNN_Symmetric p_bonf<0.05, CN > AD; Pearson channels null | §10 | `10_von_neumann_group_stats_corrected.csv` | from FC tensor |

> **§10 key result:** Findings are channel-specific. MI_KNN_Symmetric shows a significant
> reduction in VNE for AD relative to CN (Bonferroni-corrected p<0.05, modest r_rb).
> Pearson-based channels show no significant AD–CN difference after Bonferroni correction.
> See §10b table for exact values.  This does NOT imply biological equivalence in linear
> channels; VNE is a coarse scalar summary.

### Exploratory / scaffolded (not publication-grade without further work)

| Section | Item | Limitation |
|---------|------|------------|
| §4 | Gaussian TC (Ledoit-Wolf) | Gaussian approx; N~37 test subjects per fold |
| §5 | Heuristic marginal diagnostic ratio | Marginal, not conditional; not canonical IT |
| §6 | Top-k AUC saturation point | AUC quantization with N_test~37; interpret trends only |
| §8 | Ising first/second moments | No full Ising fit; pseudo-likelihood/MPF is TODO |
| §9 | Spectral ordinal complexity | Eigenvalue magnitude proxy, not temporal Bandt-Pompe |
| §10c | MCI VNE descriptive | Descriptive only; no inferential claims for MCI |
"""))
cells.append(code("""
# ── §11 Execution report ────────────────────────────────────────────────────
# Determine per-section status: recomputed | loaded_from_cache | skipped | exploratory
def _list_outputs(sub, exclude_legacy=True):
    d = OUTPUT_DIR / sub
    if not d.exists(): return []
    files = [f.name for f in sorted(d.iterdir())
             if f.is_file() and (not exclude_legacy or f.parent.name != "_legacy")]
    return files

def _sec_status(computed_flag, section_name):
    \"\"\"Return run status string for a section.\"\"\"
    if computed_flag:
        return "recomputed" if RUN_MODE == "full_recompute" else "loaded_from_cache"
    # Check if cached artifact exists
    _cache_check = {
        "sec4": "04_ib_fold_summary.csv",
        "sec5": "05_latent_mi_train_only.csv",
        "sec6": "06_topk_auc_by_strategy_by_fold.csv",
        "sec7": "07_per_fold_calibration_metrics.csv",
        "sec8": "08_ising_pooling_audit.csv",
        "sec9": "09_spectral_ordinal_complexity_proxy.csv",
        "sec10": "10_von_neumann_group_stats_corrected.csv",
    }
    key_ = section_name[:5]
    fn_  = _cache_check.get(key_)
    if fn_ and (OUTPUT_DIR / "Tables" / fn_).exists():
        return "skipped_but_cached_artifact_present"
    return "skipped"

_sec4_ok  = "_REINFERENCE_AVAILABLE" in dir() and _REINFERENCE_AVAILABLE
_sec5_ok  = "_MI_AVAILABLE"          in dir() and _MI_AVAILABLE
_sec6_ok  = "df_long"                in dir()
_sec7_ok  = "_PREDS_AVAILABLE"       in dir() and _PREDS_AVAILABLE
_sec8_ok  = "y_pool"                 in dir() and len(y_pool) > 50
_sec9_ok  = "df_bp"                  in dir()
_sec10_ok = "df_vn"                  in dir()
_sec10c_ok= "df_desc"               in dir()

sections_status = {
    "sec4_ib_audit"         : _sec_status(_sec4_ok,  "sec4"),
    "sec5_latent_mi_train"  : _sec_status(_sec5_ok,  "sec5"),
    "sec6_topk_nested"      : _sec_status(_sec6_ok,  "sec6"),
    "sec7_calibration"      : _sec_status(_sec7_ok,  "sec7"),
    "sec8_ising"            : _sec_status(_sec8_ok,  "sec8") + " [EXPLORATORY]",
    "sec9_spectral_proxy"   : _sec_status(_sec9_ok,  "sec9") + " [EXPLORATORY]",
    "sec10_von_neumann"     : _sec_status(_sec10_ok, "sec10"),
    "sec10c_mci_desc"       : _sec_status(_sec10c_ok,"sec10") + " [DESCRIPTIVE_ONLY]",
}

# §10 result summary (data-driven, only if table was computed)
_sec10_interp = "NOT COMPUTED THIS RUN"
if _sec10_ok and "df_vn_stats" in dir():
    _sig = df_vn_stats[df_vn_stats["p_value_bonferroni"] < 0.05]
    _null= df_vn_stats[df_vn_stats["p_value_bonferroni"] >= 0.05]
    _sig_list  = _sig["channel"].tolist()
    _null_list = _null["channel"].tolist()
    if len(_sig) == 0:
        _sec10_interp = "All channels null (p_bonf>0.05)"
    else:
        _entries = [f"{r['channel']} p_bonf={r['p_value_bonferroni']:.4f} r_rb={r['rank_biserial_r_rb']:.3f}"
                    for _, r in _sig.iterrows()]
        _sec10_interp = "Significant: " + "; ".join(_entries)
        if _null_list:
            _sec10_interp += "  |  Null: " + ", ".join([c.split("_")[0] for c in _null_list])
elif (OUTPUT_DIR / "Tables" / "10_von_neumann_group_stats_corrected.csv").exists():
    _cached_stats = pd.read_csv(OUTPUT_DIR / "Tables" / "10_von_neumann_group_stats_corrected.csv")
    _sig_cached   = _cached_stats[_cached_stats["p_value_bonferroni"] < 0.05]
    _null_cached  = _cached_stats[_cached_stats["p_value_bonferroni"] >= 0.05]
    if len(_sig_cached) == 0:
        _sec10_interp = "All channels null (p_bonf>0.05) [from cached table]"
    else:
        _entries_c = [f"{r['channel']} p_bonf={r['p_value_bonferroni']:.4f} r_rb={r['rank_biserial_r_rb']:.3f}"
                      for _, r in _sig_cached.iterrows()]
        _sec10_interp = "Significant [cached]: " + "; ".join(_entries_c)

# Determine overall notebook readiness
_pub_critical_ok = all([
    _sec7_ok,   # from fold predictions — always expected
    _sec10_ok,  # from FC tensor — always expected
])
_reinf_ok = _sec4_ok and _sec5_ok and _sec6_ok and _sec8_ok
if _pub_critical_ok and _reinf_ok:
    _pub_ready = "YES (full_recompute: all sections recomputed)"
elif _pub_critical_ok and not _reinf_ok:
    _pub_ready = ("CONDITIONAL (artifact_review: §7/§10 recomputed; "
                  "§§4-6/§8 status: see sections_status)")
else:
    _pub_ready = "NO — critical sections failed (§7 or §10 missing)"

nb_manifest = {
    "created_utc"         : time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "notebook_version"    : "v4",
    "run_mode"            : RUN_MODE,
    "python_version"      : sys.version.split()[0],
    "results_dir"         : str(RESULTS_DIR),
    "output_dir"          : str(OUTPUT_DIR),
    "v3_critical_fix"     : ("Use test_tensor_idx.npy/train_dev_tensor_idx.npy (global indices) "
                             "instead of test_indices.npy/train_dev_indices.npy (local pool indices)"),
    "v4_additions"        : ("RUN_MODE fail-closed system; dependency preflight; "
                             "300 dpi+PDF figures; LaTeX table exports; §10 narrative corrected"),
    "re_inference_log"    : _reinference_log if "_reinference_log" in dir() else [],
    "sections_status"     : sections_status,
    "sec10_interpretation": _sec10_interp,
    "publication_ready"   : _pub_ready,
    "tables"              : _list_outputs("Tables"),
    "figures"             : [f for f in _list_outputs("Figures") if f.endswith(".png")],
    "pdfs"                : [f for f in _list_outputs("Figures") if f.endswith(".pdf")],
    "legacy_quarantined"  : [f.name for f in (OUTPUT_DIR/"_legacy").iterdir()] if (OUTPUT_DIR/"_legacy").exists() else [],
}
with open(OUTPUT_DIR / "notebook_manifest.json", "w") as f:
    json.dump(nb_manifest, f, indent=2)

print("=" * 70)
print("NOTEBOOK 07 v4 — EXECUTION REPORT")
print("=" * 70)
print(f"  Run mode      : {RUN_MODE}")
print(f"  Python        : {sys.version.split()[0]}")
print(f"  Output dir    : {OUTPUT_DIR}")
print()
print("  Section status:")
for k, v in sections_status.items():
    _icon = "[OK]" if "recomputed" in v or "loaded" in v else "[--]"
    print(f"    {_icon} {k:30s}: {v}")
print()
print(f"  §10 finding   : {_sec10_interp}")
print()
print(f"  Publication-ready: {_pub_ready}")
print()
print(f"  Tables  : {nb_manifest['tables']}")
print(f"  Figures : {nb_manifest['figures']}")
print(f"  PDFs    : {nb_manifest['pdfs']}")
if nb_manifest['legacy_quarantined']:
    print(f"  Legacy quarantined: {nb_manifest['legacy_quarantined']}")
"""))

# ── Assemble ────────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4, "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name":"Python 3 (ipykernel)","language":"python","name":"python3"},
        "language_info": {"name":"python","version":"3.10.0"},
    },
    "cells": cells,
}

out = ROOT / "notebooks" / "07_it_information_theory.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"Written: {out}")
print(f"  Cells : {len(cells)}  "
      f"code={sum(1 for c in cells if c['cell_type']=='code')}  "
      f"md={sum(1 for c in cells if c['cell_type']=='markdown')}")
