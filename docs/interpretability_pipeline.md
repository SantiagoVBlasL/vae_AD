# Interpretability Pipeline — Deep-Dive Execution Trace

This document provides a complete technical reference for the interpretability pipeline
invoked from `notebooks/05_run_interpretability_ad.ipynb` (§4).

---

## 1. Full Execution Tree / Call Graph

```
notebooks/05_run_interpretability_ad.ipynb  (§4 — Pipeline execution cell)
│
│  [subprocess.run — one invocation per fold × subcommand]
│
├── python3 scripts/run_interpretability.py  shap  --shap_tag frozen  <shared_args>
├── python3 scripts/run_interpretability.py  shap  --shap_tag unfrozen <shared_args>
└── python3 scripts/run_interpretability.py  saliency --shap_tag frozen <shared_args>
         │
         │  scripts/run_interpretability.py
         │    adds src/ to sys.path
         │    from betavae_xai.interpretability.interpret_fold import main
         │    main()
         │
         └── src/betavae_xai/interpretability/interpret_fold.py :: main()
                  │
                  ├── parse_args()  →  argparse subcommands: shap | saliency | shap_edges
                  │
                  ├─── [cmd = 'shap'] ──────────────────────────────────────────────────
                  │    cmd_shap(args)
                  │      │
                  │      ├── _resolve_pipeline_path()
                  │      │     └── loads  fold_k/classifier_{clf}_{variant}_pipeline_fold_k.joblib
                  │      │
                  │      ├── _load_global_and_merge()
                  │      │     ├── np.load(GLOBAL_TENSOR.npz)
                  │      │     └── pd.read_csv(metadata.csv)
                  │      │
                  │      ├── _subset_cnad()           → filter CN / AD rows
                  │      ├── apply_normalization_params()
                  │      │
                  │      ├── build_vae()
                  │      │     └── ConvolutionalVAE(**vae_kwargs)   [models/convolutional_vae.py]
                  │      │           torch.load(fold_k/vae_model_fold_k.pt)
                  │      │
                  │      ├── vae.forward()  →  (recon, mu, logvar, z)
                  │      │
                  │      ├── [freeze_meta branch]
                  │      │     compute train-split median / mode for Age / Sex
                  │      │
                  │      ├── _build_background_from_train()   [or cached .joblib]
                  │      │     └── vae.forward() on TRAIN sample → background DataFrame
                  │      │
                  │      ├── shap.KernelExplainer  /  shap.Explainer  (SHAP library)
                  │      │     ├── background = X_proc_bg  (preprocessed, N_bg × F)
                  │      │     └── explain  X_proc_test    (N_test × F)
                  │      │
                  │      └── joblib.dump()
                  │            └── fold_k/interpretability_shap/shap_pack_{clf}_{tag}.joblib
                  │
                  ├─── [cmd = 'saliency'] ─────────────────────────────────────────────
                  │    cmd_saliency(args)
                  │      │
                  │      ├── joblib.load(shap_pack_{clf}_{tag}.joblib)   ← output of 'shap'
                  │      │
                  │      ├── get_latent_weights_from_pack()
                  │      │     extracts latent importance weights per --shap_weight_mode
                  │      │
                  │      ├── _load_global_and_merge() + apply_normalization_params()
                  │      ├── build_vae()   [same as shap path]
                  │      │
                  │      ├── [saliency_mode = 'latent']
                  │      │     ├── vanilla  → generate_saliency_vectorized()
                  │      │     │     autograd: d(Σ w_i · μ_i) / d(input_tensor)
                  │      │     ├── smoothgrad → generate_saliency_smoothgrad()
                  │      │     │     N noisy copies → mean gradient
                  │      │     └── integrated_gradients → generate_saliency_integrated_gradients()
                  │      │           captum.attr.IntegratedGradients
                  │      │           baseline = zeros | cn_median_train | cn_median_test
                  │      │
                  │      ├── [saliency_mode = 'ig_classifier_score']  (early return)
                  │      │     ├── extract_logreg_latent_weights()   [composite_edge_shap.py]
                  │      │     └── generate_saliency_ig_classifier_score()
                  │      │           IG on  score(x) = w_lat · vae.encode(x).mu + b
                  │      │
                  │      ├── _ranking_and_heatmap()
                  │      │     ├── sort edges by |Δsaliency| AD − CN
                  │      │     └── seaborn heatmap saved to interpretability_{clf}/
                  │      │
                  │      └── outputs written to fold_k/interpretability_{clf}/
                  │            ├── saliency_map_{ad|cn|diff}_{signed|abs}*.npy
                  │            ├── ranking_conexiones_*.csv
                  │            └── ranking_roi_*.csv
                  │
                  └─── [cmd = 'shap_edges']  (composite edge-level SHAP) ───────────
                       cmd_shap_edges(args)
                         │
                         ├── composite_edge_shap.py :: make_edge_index()
                         │     precompute upper-triangular row/col indices
                         │
                         ├── composite_edge_shap.py :: vectorize_tensor_to_edges()
                         │     (N,C,R,R) → (N, C×E)  flatten
                         │
                         ├── composite_edge_shap.py :: select_top_edges()
                         │   / select_top_edges_per_channel()
                         │     TRAIN-only feature selection (f_classif | mutual_info | l1_logreg)
                         │
                         ├── composite_edge_shap.py :: compute_train_edge_median()
                         │     background template for non-selected edges
                         │
                         ├── composite_edge_shap.py :: compute_frozen_meta_values()
                         │     freeze Age/Sex at TRAIN statistics
                         │
                         ├── composite_edge_shap.py :: make_edge_predict_fn()
                         │     composite predict:
                         │       K selected edges
                         │         → fill template (median for rest)
                         │         → reconstruct_tensor_from_edges()  (N,C,R,R)
                         │         → vae.encode() → mu
                         │         → pipe.predict_proba()  → P(AD)
                         │
                         ├── composite_edge_shap.py :: validate_edge_roundtrip()
                         │     sanity: direct_path ≈ edge_roundtrip_path  (atol 1e-5)
                         │
                         ├── shap.Explainer (PermutationExplainer)
                         │     background = edge_vectors of TRAIN sample
                         │     explain   = edge_vectors of TEST  set
                         │
                         ├── composite_edge_shap.py :: make_edge_mapping_df()
                         │     map flat edge index → (channel, roi_i, roi_j, name)
                         │
                         └── outputs written to fold_k/interpretability_shap/
                               ├── shap_edges_pack_{clf}_{tag}.joblib
                               └── edge_ranking_{clf}_{tag}.csv
```

### Supporting library modules (always imported, never called via CLI)

```
src/betavae_xai/
├── interpretability/
│   ├── __init__.py                 re-exports public API
│   ├── interpret_fold.py           CLI dispatcher + all cmd_* functions
│   ├── composite_edge_shap.py      pure helpers for edge-level SHAP
│   └── interpretability_utils.py   post-hoc statistical helpers
│         ├── load_shap_pack()
│         ├── bootstrap_shap_importance_ci()
│         ├── permutation_test_shap_feature()
│         ├── compute_fdr_correction()
│         ├── compute_jaccard_stability() / compute_dice_coefficient()
│         ├── compute_icc_across_folds()
│         ├── compute_cohens_d() / compute_effect_sizes_matrix()
│         ├── aggregate_saliency_by_network()
│         ├── compute_factor_predictability()
│         └── analyze_age_confounding()
├── models/
│   ├── convolutional_vae.py        ConvolutionalVAE (β-VAE, GroupNorm, CNN)
│   └── classifiers.py              sklearn-compatible wrapper heads
├── data/
│   └── preprocessing.py            dataset loading & normalization utilities
└── utils/
    ├── run_io.py                   _compute_file_sha256, _safe_json_dump
    ├── checkpoints.py              model checkpoint helpers
    └── logging.py                  unified logging configuration
```

---

## 2. Roles & Responsibilities

| Component | Role |
|-----------|------|
| `notebooks/05_run_interpretability_ad.ipynb §4` | **Orchestrator.** Drives the full loop over folds and subcommands. Builds CLI argument lists via `_shared_args()`, `_cmd_shap()`, `_cmd_saliency()`. Fires `subprocess.run()` per invocation. Keeps all configuration in a single editable cell (§1) for reproducibility. |
| `scripts/run_interpretability.py` | **Thin CLI entrypoint / path broker.** Adds `src/` to `sys.path` so the package is importable without editable-install, then delegates entirely to `betavae_xai.interpretability.interpret_fold.main()`. No logic of its own; exists to be a stable, relocatable shell-callable target. |
| `interpret_fold.py :: main() / parse_args()` | **Argument router.** Declares three subcommands (`shap`, `saliency`, `shap_edges`) with all flags via argparse, sets global seeds, and dispatches to the matching `cmd_*` function. |
| `interpret_fold.py :: cmd_shap()` | **SHAP computation engine.** Reconstructs the exact preprocessing chain used during training (VAE encode → latent+meta DataFrame), builds a leak-free background from TRAIN subjects, runs `shap.KernelExplainer` / `shap.Explainer`, and serialises a rich `shap_pack` dict (values, feature names, test labels, masks) as a `.joblib` artifact. Supports frozen/unfrozen metadata variants. |
| `interpret_fold.py :: cmd_saliency()` | **Gradient saliency engine.** Loads an existing `shap_pack` to obtain latent importance weights, then backpropagates through the VAE decoder to attribute importance to each brain-connectivity edge. Supports three methods (vanilla gradient, SmoothGrad, Integrated Gradients via captum) and two saliency modes (`latent` and `ig_classifier_score`). Writes signed/absolute saliency maps plus human-readable CSV rankings. |
| `interpret_fold.py :: cmd_shap_edges()` | **Composite edge-level SHAP engine.** Bypasses the latent bottleneck and attributes `P(AD)` directly to raw brain-edge features via a composite predict function that internally calls the full VAE→classifier chain. Uses SHAP's PermutationExplainer on the K-dimensional edge subspace. |
| `composite_edge_shap.py` | **Pure edge-space helpers.** All geometry (index precomputation, vectorise/reconstruct), TRAIN-only edge selection, frozen metadata computation, composite predict-function factory, and roundtrip validation. No CLI, no I/O; purely functional. Designed to be unit-testable in isolation. |
| `interpretability_utils.py` | **Post-hoc statistical toolkit.** Consumed by the notebook (§5) after all `.joblib` packs exist. Provides bootstrap CIs, FDR-corrected permutation tests, stability metrics (Jaccard, Dice, ICC), effect sizes (Cohen's d), network-level aggregation, disentanglement (SAP), and age-confounding analysis. |
| `ConvolutionalVAE` (`models/convolutional_vae.py`) | **Generative model backbone.** CNN β-VAE with GroupNorm. Exposes `encode()` (→ μ, logvar) and `reparameterize()` used at inference time. Loaded from `.pt` checkpoint via `build_vae()` inside interpret_fold. |
| `utils/run_io.py` | **Serialisation safety layer.** `_safe_json_dump` handles Path/numpy/torch types and writes atomically. Used to persist `run_config.json` and `feature_columns.json` during training; consumed by the notebook (§3) to reconstruct experiment configuration. |

**Why is the logic separated this way?**

- The **notebook** stays purely declarative: one editable configuration cell drives everything, and all heavy computation is pushed to subprocesses. This gives clean stdout capture, crash isolation per fold, and enables parallel fold execution with minimal notebook changes.
- The **thin wrapper script** decouples the notebook from Python import paths, making the entrypoint shell-callable and environment-agnostic.
- The **subcommand split** (`shap` vs `saliency`) enforces that saliency cannot start without a valid SHAP pack — the dependency is made explicit at the filesystem level (`.joblib` artifact). `shap_edges` is a separate, parallel branch because its input space (raw edges) and explainer (PermutationExplainer) differ fundamentally from the latent-space path.
- **`composite_edge_shap.py`** is isolated as a pure-function module because the composite predict function has a complex dependency chain (geometry, VAE, classifier) that must be unit-tested independently of the CLI.
- **`interpretability_utils.py`** lives in the analysis layer (consumed by the notebook) rather than the execution layer, keeping statistical post-processing completely decoupled from artifact generation.

---

## 3. Data Flow & Interfaces

### 3.1 Parameter passing (Notebook → Script)

The notebook cell `§4` builds CLI argument lists programmatically:

```python
# Shared args (both subcommands)
_shared_args(fold) → [
    "--run_dir",                  str(RESULTS_DIR),
    "--fold",                     str(fold),
    "--clf",                      TARGET_CLF,           # e.g. "logreg"
    "--global_tensor_path",       str(GLOBAL_TENSOR_PATH),
    "--metadata_path",            str(METADATA_PATH),
    "--channels_to_use",          *[str(c) for c in CHANNELS_TO_USE],
    "--latent_dim",               str(LATENT_DIM),
    "--latent_features_type",     "mu",
    "--num_conv_layers_encoder",  str(NUM_CONV_LAYERS_ENC),
    "--decoder_type",             DECODER_TYPE,
    "--dropout_rate_vae",         str(DROPOUT_RATE_VAE),
    "--intermediate_fc_dim_vae",  INT_FC_DIM_VAE,
    "--vae_final_activation",     VAE_FINAL_ACT,
    "--seed",                     str(SEED_GLOBAL),
    "--metadata_features",        *METADATA_FEATURES,   # e.g. ["Age", "Sex"]
]

# SHAP-specific extras
_cmd_shap(fold, tag="frozen", freeze_meta=["Age","Sex"]) → [
    "shap",
    "--kernel_nsamples",   str(SHAP_KERNEL_NSAMPLES),
    "--shap_link",         "logit",
    "--bg_mode",           "train",
    "--bg_sample_size",    str(SHAP_BG_SAMPLE_SIZE),
    "--bg_seed",           str(SEED_BG),
    "--shap_normalize",    "by_logit_median",
    "--shap_tag",          "frozen",
    "--freeze_meta",       "Age", "Sex",
    "--freeze_strategy",   "train_stats",
]

# Saliency-specific extras
_cmd_saliency(fold, shap_tag="frozen") → [
    "saliency",
    "--roi_annotation_path", str(ROI_ANNOTATION_PATH),
    "--top_k",               str(TOPK_LATENTS),
    "--shap_weight_mode",    "ad_vs_cn_diff",
    "--saliency_method",     "integrated_gradients",
    "--shap_tag",            "frozen",
    "--ig_n_steps",          str(IG_N_STEPS),
    "--ig_baseline",         "cn_median_train",
]
```

### 3.2 Training artifacts consumed (inputs to interpret pipeline)

All paths are relative to `RESULTS_DIR/fold_{k}/`:

| Artifact | Format | Producer | Consumer |
|----------|--------|----------|----------|
| `vae_model_fold_{k}.pt` | PyTorch state-dict | training script | `build_vae()` in both `cmd_shap` and `cmd_saliency` |
| `classifier_{clf}_{variant}_pipeline_fold_{k}.joblib` | sklearn Pipeline / CalibratedClassifierCV | training script | `cmd_shap` (SHAP background & predictor), `cmd_shap_edges` |
| `vae_norm_params.joblib` | `List[Dict[str,float]]` (per-channel μ/σ) | training script | `apply_normalization_params()` in all subcommands |
| `test_indices.npy` | `ndarray(int)` into CN/AD DataFrame | training script | all subcommands (reconstruct exact test split) |
| `feature_columns.json` | `{"final_feature_columns": [...]}` | training script | `cmd_saliency` (ig_classifier_score mode), `cmd_shap_edges` |
| `run_config.json` | JSON dict of all training args | training script | notebook §3 (configuration discovery) |
| `roi_order_131.joblib` | `List[str]` — AAL3 ROI names | training script | `cmd_shap`, `cmd_saliency`, `cmd_shap_edges` |

Global inputs (not fold-specific):

| Artifact | Format | Description |
|----------|--------|-------------|
| `GLOBAL_TENSOR.npz` | `ndarray(N, C, R, R)` float32 | Full connectivity tensor for all subjects |
| `metadata.csv` | CSV with Subject_ID, ResearchGroup, Age, Sex | Subject metadata |
| `roi_info_master.csv` | CSV with AAL3_Name, Macro_Lobe, Refined_Network … | ROI annotation (for saliency rankings) |

### 3.3 Interpretability artifacts produced (outputs handed back to notebook)

All paths are relative to `RESULTS_DIR/fold_{k}/`:

| Artifact | Format | Produced by | Consumed by (notebook §) |
|----------|--------|-------------|--------------------------|
| `interpretability_shap/shap_pack_{clf}_{tag}.joblib` | dict: `shap_values (N,F)`, `feature_names`, `X_test (DataFrame)`, `y_test`, `latent_feature_mask`, `test_labels` | `cmd_shap` | §4b integrity gate; §5.1 SHAP importance; §5 stability; `cmd_saliency` (shap_tag) |
| `interpretability_shap/shap_background_{mode}{tag}.joblib` | DataFrame (N_bg, F) | `cmd_shap` (cached) | re-used across runs to avoid recomputation |
| `interpretability_{clf}/saliency_map_{group}_{type}{suffix}.npy` | `ndarray(C, R, R)` float32 | `cmd_saliency` | §5.2 saliency heatmaps; network aggregation |
| `interpretability_{clf}/ranking_conexiones_{clf}{suffix}.csv` | top-K edges sorted by \|Δsaliency\| | `cmd_saliency` | §5.2 edge table; §5.3 cross-fold consensus |
| `interpretability_{clf}/ranking_roi_{clf}{suffix}.csv` | ROI-level aggregated saliency | `cmd_saliency` | §5.2 ROI bar charts |
| `interpretability_{clf}/heatmap_saliency_{clf}{suffix}.{png,svg,pdf}` | matplotlib figure | `cmd_saliency` | paper figures |
| `interpretability_shap/shap_edges_pack_{clf}_{tag}.joblib` | dict: `shap_values (N_test,K)`, `edge_names`, `edge_mapping_df` | `cmd_shap_edges` | §5.4 composite edge analysis |
| `interpretability_shap/edge_ranking_{clf}_{tag}.csv` | top-K edges by mean\|SHAP\| | `cmd_shap_edges` | §5.4 |

### 3.4 Post-hoc analysis interface (Notebook §5 → `interpretability_utils.py`)

The notebook imports utility functions directly (not via subprocess) from
`betavae_xai.interpretability`:

```python
from betavae_xai.interpretability import (
    load_shap_pack,                  # loads .joblib packs by (run_dir, fold, clf, tag)
    bootstrap_shap_importance_ci,    # pooled mean|SHAP| with 95% bootstrap CI
    permutation_test_shap_feature,   # sign-flip permutation p-value per feature
    compute_fdr_correction,          # Benjamini-Hochberg FDR
    compute_jaccard_stability,       # mean pairwise Jaccard across fold top-K sets
    compute_icc_across_folds,        # ICC(2,1) for feature ranking consistency
    compute_cohens_d,                # Cohen's d: AD vs CN latent distributions
    aggregate_saliency_by_network,   # (R,R) → (N_net, N_net) network matrix
    compute_factor_predictability,   # SAP score per factor (Age, Sex, Dx)
    analyze_age_confounding,         # Pearson r(SHAP, Age) + FDR per latent
)
```

These functions consume the `.joblib` packs produced in §4 and return
`pd.DataFrame` objects and scalar statistics that the notebook renders
directly into figures and tables.

### 3.5 End-to-end data flow summary

```
GLOBAL_TENSOR.npz ──┐
metadata.csv ────────┤
                     │  [§4 subprocess — per fold]
fold_k/ ─────────────┤
  vae_model.pt       │
  pipeline.joblib    │
  norm_params.joblib │
  test_indices.npy   │
                     ▼
             run_interpretability.py
                     │
                 interpret_fold.py
                     │
        ┌────────────┼────────────────────┐
        ▼            ▼                    ▼
    cmd_shap    cmd_saliency       cmd_shap_edges
        │            │                    │
  shap_pack_     saliency_map_       shap_edges_
  {clf}_{tag}    {type}.npy          pack_{tag}
   .joblib       ranking_*.csv        .joblib
                 heatmap_*.png/svg    edge_ranking_*.csv
        │
        └──────────────────────────────────────────────────────────┐
                                                                    │
notebooks/05_run_interpretability_ad.ipynb §5                       │
  ├── load_shap_pack() ◄──────────────────────────────────────────-─┘
  ├── bootstrap_shap_importance_ci()  →  shap_feature_importance_by_fold.csv
  ├── compute_jaccard_stability()     →  stability table
  ├── compute_icc_across_folds()      →  ICC report
  ├── aggregate_saliency_by_network() →  network heatmap
  └── analyze_age_confounding()       →  age-correlation table
```
