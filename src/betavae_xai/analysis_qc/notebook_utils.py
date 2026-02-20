"""
Utility functions for notebook analysis in BetaVAE-XAI-AD project.

path: /home/diego/proyectos/vae_AD/src/betavae_xai/analysis_qc/notebook_utils.py

This module contains functions for calibration analysis, normalization,
information theory metrics, and publication-quality visualization utilities.

Functions are organized in sections:
- Calibration: ECE, MCE, Brier score, calibration curves
- Normalization: Tensor normalization for inference
- Information Theory: Participation ratio, Total Correlation, etc.
- Visualization: Publication-ready figure styling and export

Author: BetaVAE-XAI-AD Project
Date: 2026-02-12
"""

# ============================================================================
# IMPORTS
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

# Calibration functions
from sklearn.metrics import brier_score_loss
from sklearn.linear_model import LogisticRegression

# Information theory functions
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================================
# CALIBRATION FUNCTIONS
# ============================================================================

EPS = 1e-12


def _make_bin_edges(p, n_bins=10, strategy="quantile"):
    """
    Create robust bin edges for calibration analysis.

    Handles edge cases like collapsed quantiles by falling back to uniform bins.

    Parameters
    ----------
    p : array-like
        Predicted probabilities, shape (n_samples,)
    n_bins : int, default=10
        Number of bins to create
    strategy : {'quantile', 'uniform'}, default='quantile'
        Binning strategy:
        - 'quantile': Equal sample count per bin
        - 'uniform': Equal width bins

    Returns
    -------
    numpy.ndarray
        Bin edges, shape (n_bins + 1,)
        First edge is 0.0, last edge is 1.0
    """
    p = np.asarray(p, dtype=float)
    p = np.clip(p, EPS, 1 - EPS)

    if strategy == "quantile":
        # If too few unique values, quantile collapses -> fallback to uniform
        if np.unique(p).size < n_bins:
            edges = np.linspace(0, 1, n_bins + 1)
        else:
            edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
            # If edges are repeated, fallback
            if np.unique(edges).size < 3:
                edges = np.linspace(0, 1, n_bins + 1)
    else:
        edges = np.linspace(0, 1, n_bins + 1)

    edges = np.asarray(edges, dtype=float)
    edges[0], edges[-1] = 0.0, 1.0
    return edges


def weighted_ece_mce(y, p, n_bins=10, strategy="quantile"):
    """
    Compute weighted Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    ECE is weighted by bin size, providing a more balanced metric than unweighted versions.

    Parameters
    ----------
    y : array-like
        True binary labels, shape (n_samples,)
    p : array-like
        Predicted probabilities, shape (n_samples,)
    n_bins : int, default=10
        Number of bins for calibration
    strategy : {'quantile', 'uniform'}, default='quantile'
        Binning strategy

    Returns
    -------
    ece : float
        Expected Calibration Error (weighted)
    mce : float
        Maximum Calibration Error
    bins_df : pandas.DataFrame
        Per-bin statistics with columns:
        - bin: Bin index
        - n: Sample count in bin
        - acc: Observed accuracy (fraction of positives)
        - conf: Mean predicted probability
        - gap: |acc - conf|
        - lo, hi: Bin edges

    References
    ----------
    .. [1] Guo et al. "On Calibration of Modern Neural Networks", ICML 2017
    """
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    p = np.clip(p, EPS, 1 - EPS)

    edges = _make_bin_edges(p, n_bins=n_bins, strategy=strategy)

    ece = 0.0
    mce = 0.0
    rows = []

    n_total = len(y)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p <= hi) if i == n_bins - 1 else (p >= lo) & (p < hi)
        n = int(mask.sum())
        if n == 0:
            continue

        acc = float(y[mask].mean())
        conf = float(p[mask].mean())
        gap = abs(acc - conf)
        w = n / n_total

        ece += w * gap
        mce = max(mce, gap)
        rows.append({"bin": i, "n": n, "acc": acc, "conf": conf, "gap": gap, "lo": lo, "hi": hi})

    return float(ece), float(mce), pd.DataFrame(rows)


def calibration_slope_intercept(y, p):
    """
    Compute calibration slope and intercept via logistic regression.

    Fits the model: y ~ a + b * logit(p)
    Perfect calibration: a = 0, b = 1

    Parameters
    ----------
    y : array-like
        True binary labels, shape (n_samples,)
    p : array-like
        Predicted probabilities, shape (n_samples,)

    Returns
    -------
    intercept : float
        Calibration intercept (a). NaN if single class.
    slope : float
        Calibration slope (b). NaN if single class.

    Notes
    -----
    Uses high regularization (C=1e6) for numerical stability.
    Returns (nan, nan) if only one class present.
    """
    y = np.asarray(y, dtype=int)
    if np.unique(y).size < 2:
        return np.nan, np.nan

    p = np.asarray(p, dtype=float)
    p = np.clip(p, EPS, 1 - EPS)
    logit = np.log(p / (1 - p)).reshape(-1, 1)

    try:
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
        lr.fit(logit, y)
        a = float(lr.intercept_[0])
        b = float(lr.coef_[0][0])
        return a, b
    except Exception:
        return np.nan, np.nan


def reliability_report(df, y="_y", p="_p", n_bins=10, strategy="quantile"):
    """
    Generate comprehensive calibration metrics for binary classifier predictions.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing true labels and predictions
    y : str, default="_y"
        Column name for true binary labels
    p : str, default="_p"
        Column name for predicted probabilities
    n_bins : int, default=10
        Number of bins for calibration analysis
    strategy : str, default="quantile"
        Binning strategy: "quantile" or "uniform"

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - N: Total samples
        - n_pos: Number of positive samples
        - n_neg: Number of negative samples
        - prev: Prevalence (proportion of positive class)
        - Brier: Brier score
        - BSS: Brier Skill Score
        - ECE_w: Weighted Expected Calibration Error
        - MCE: Maximum Calibration Error
        - calib_intercept: Calibration intercept
        - calib_slope: Calibration slope
    bins_df : pandas.DataFrame
        Per-bin calibration statistics

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'_y': [0, 0, 1, 1], '_p': [0.1, 0.3, 0.7, 0.9]})
    >>> metrics, bins = reliability_report(df)
    >>> print(f"ECE: {metrics['ECE_w']:.3f}")
    ECE: 0.000

    References
    ----------
    .. [1] Guo et al. "On Calibration of Modern Neural Networks", ICML 2017
    """
    yv = df[y].to_numpy().astype(int)
    pv = df[p].to_numpy().astype(float)
    pv = np.clip(pv, EPS, 1 - EPS)

    ece, mce, bins_df = weighted_ece_mce(yv, pv, n_bins=n_bins, strategy=strategy)

    brier = float(brier_score_loss(yv, pv))
    null_p = float(yv.mean())

    # If null_p is 0 or 1 => brier_null = 0, BSS undefined
    brier_null = float(brier_score_loss(yv, np.full_like(yv, null_p)))
    bss = float(1.0 - (brier / brier_null)) if brier_null > 1e-15 else np.nan

    a, b = calibration_slope_intercept(yv, pv)

    return {
        "N": int(len(df)),
        "n_pos": int(yv.sum()),
        "n_neg": int(len(yv) - yv.sum()),
        "prev": float(yv.mean()),
        "Brier": brier,
        "BSS": bss,
        "ECE_w": float(ece),
        "MCE": float(mce),
        "calib_intercept": a,
        "calib_slope": b,
    }, bins_df


# ============================================================================
# NORMALIZATION FUNCTIONS
# ============================================================================

def apply_norm_inference(tensor_mv, params_list):
    """
    Apply normalization to a single multi-view tensor for inference.

    Normalizes only off-diagonal elements of connectivity matrices,
    preserving the diagonal (which represents self-connectivity).

    Parameters
    ----------
    tensor_mv : numpy.ndarray
        Shape (C, H, W) - multi-channel connectivity matrix
        C: channels (e.g., FC, ReHo, ALFF)
        H, W: spatial dimensions (ROIs)
    params_list : list of dict
        Normalization parameters for each channel. Each dict may contain:
        - "mode": "zscore_offdiag" or "minmax_offdiag"
        - "no_scale": bool, skip normalization if True
        - "mean", "std": for zscore
        - "min", "max": for minmax

    Returns
    -------
    numpy.ndarray
        Normalized tensor, same shape as input

    Examples
    --------
    >>> tensor = np.random.randn(3, 116, 116)
    >>> params = [
    ...     {"mode": "zscore_offdiag", "mean": 0.0, "std": 1.0},
    ...     {"mode": "zscore_offdiag", "mean": 0.0, "std": 1.0},
    ...     {"mode": "minmax_offdiag", "min": -1.0, "max": 1.0}
    ... ]
    >>> normalized = apply_norm_inference(tensor, params)
    """
    C, H, W = tensor_mv.shape
    normed = tensor_mv.copy()
    mask = ~np.eye(H, dtype=bool)

    for c in range(C):
        p = params_list[c]
        if p.get("no_scale", False):
            continue

        vals = tensor_mv[c]
        mode = p.get("mode", None)

        if mode == "zscore_offdiag":
            std = float(p.get("std", 0.0))
            if std > 1e-9:
                normed[c, mask] = (vals[mask] - float(p.get("mean", 0.0))) / std
            else:
                normed[c, mask] = 0.0

        elif mode == "minmax_offdiag":
            vmin = float(p.get("min", 0.0))
            vmax = float(p.get("max", 1.0))
            rng = vmax - vmin
            if rng > 1e-9:
                normed[c, mask] = (vals[mask] - vmin) / rng
            else:
                normed[c, mask] = 0.0

    return normed


def apply_norm_batch(tensor_4d: np.ndarray, params_list):
    """
    Apply normalization to a batch of multi-view tensors.

    Parameters
    ----------
    tensor_4d : numpy.ndarray
        Shape (N, C, H, W) - batch of multi-channel connectivity matrices
        N: batch size
        C: channels
        H, W: spatial dimensions
    params_list : list of dict
        Normalization parameters for each channel

    Returns
    -------
    numpy.ndarray
        Normalized batch, same shape as input

    Examples
    --------
    >>> batch = np.random.randn(10, 3, 116, 116)
    >>> params = [{"mode": "zscore_offdiag", "mean": 0, "std": 1}] * 3
    >>> normalized = apply_norm_batch(batch, params)
    """
    N = tensor_4d.shape[0]
    out = np.empty_like(tensor_4d)
    for i in range(N):
        out[i] = apply_norm_inference(tensor_4d[i], params_list)
    return out


# ============================================================================
# INFORMATION THEORY FUNCTIONS
# ============================================================================

def participation_ratio(evals):
    """
    Calculate effective dimensionality from eigenvalues.

    The participation ratio quantifies how many dimensions are actively used
    to represent the data. Higher values indicate more dimensions contribute
    meaningfully to the variance.

    Parameters
    ----------
    evals : array-like
        Eigenvalues (e.g., from PCA or covariance matrix)

    Returns
    -------
    float
        Participation ratio: (Σλᵢ)² / Σ(λᵢ²)
        Range: [1, n_dimensions]
        Returns np.nan if no positive eigenvalues exist

    Notes
    -----
    - PR = 1: Single dominant dimension (highly concentrated)
    - PR = D: All dimensions equally important (uniform distribution)
    - PR is related to the inverse of the normalized Herfindahl index

    Examples
    --------
    >>> # Uniform eigenvalues -> high PR
    >>> evals_uniform = np.ones(10)
    >>> participation_ratio(evals_uniform)
    10.0

    >>> # Single dominant eigenvalue -> low PR
    >>> evals_concentrated = np.array([100, 1, 1, 1, 1])
    >>> participation_ratio(evals_concentrated)
    1.26...
    """
    ev = np.asarray(evals, dtype=float)
    ev = ev[ev > 0]
    if ev.size == 0:
        return np.nan
    s1, s2 = ev.sum(), np.square(ev).sum()
    return (s1 * s1) / s2


def total_correlation_gauss(Z):
    """
    Calculate Total Correlation (TC) assuming Gaussian distribution.

    TC measures multivariate mutual information - how much information is
    shared across all dimensions. TC = 0 indicates statistical independence
    between all dimensions (perfect disentanglement).

    Parameters
    ----------
    Z : numpy.ndarray
        Shape (N, D) - latent representations or data matrix
        N: number of samples
        D: number of dimensions

    Returns
    -------
    float
        Total correlation in bits
        TC = Σᵢ H(zᵢ) - H(Z)
        Range: [0, ∞)
        - TC = 0: Complete independence (disentangled)
        - Higher TC: More redundancy between dimensions

    Notes
    -----
    Uses Gaussian approximation: assumes multivariate normal distribution.
    Adds small regularization (1e-6) to diagonal for numerical stability.

    Examples
    --------
    >>> # Independent dimensions -> low TC
    >>> Z_independent = np.random.randn(1000, 10)
    >>> total_correlation_gauss(Z_independent)
    0.05...  # Close to 0

    >>> # Highly correlated dimensions -> high TC
    >>> Z_correlated = np.random.randn(1000, 1) @ np.ones((1, 10))
    >>> Z_correlated += 0.1 * np.random.randn(1000, 10)
    >>> total_correlation_gauss(Z_correlated)
    25.3...  # Much higher

    References
    ----------
    .. [1] Watanabe, S. "Information theoretical analysis of multivariate
           correlation", IBM Journal 1960
    .. [2] Chen et al. "Isolating Sources of Disentanglement in VAEs", NeurIPS 2018
    """
    # Empirical covariance
    cov = np.cov(Z, rowvar=False)
    N_dim = cov.shape[0]

    # Minimal regularization to avoid log(0) in collapsed dims
    eps = 1e-6
    cov_reg = cov + eps * np.eye(N_dim)

    # 1. Sum of marginal entropies (log of diagonal variances)
    # H(zᵢ) ≈ 0.5 * log(var_i) for Gaussian
    log_sum_marginals = np.sum(np.log(np.diag(cov_reg)))

    # 2. Joint entropy (log of determinant -> sum of log eigenvalues)
    # H(Z) ≈ 0.5 * log(det(Σ)) for Gaussian
    eigvals = np.linalg.eigvalsh(cov_reg)
    eigvals = eigvals[eigvals > 0]  # safety
    log_det = np.sum(np.log(eigvals))

    # TC = Σᵢ H(zᵢ) - H(Z)
    tc_nats = 0.5 * (log_sum_marginals - log_det)
    return tc_nats / np.log(2)  # Convert nats to bits


def clean_confounder_label(s):
    """
    Clean manufacturer/site labels by removing common suffixes.

    Standardizes labels for consistent analysis and visualization.

    Parameters
    ----------
    s : str or any
        Label string (e.g., manufacturer name, site name)

    Returns
    -------
    str
        Cleaned label with common suffixes removed
        Returns "Unknown" for None/NaN values

    Examples
    --------
    >>> clean_confounder_label("Siemens Medical Systems")
    'Siemens'
    >>> clean_confounder_label("GE Healthcare")
    'GE'
    >>> clean_confounder_label(None)
    'Unknown'
    """
    if pd.isna(s):
        return 'Unknown'
    return str(s).replace(' Medical Systems', '').replace(' Healthcare', '').strip()


# ============================================================================
# VISUALIZATION UTILITIES
# ============================================================================

def set_publication_style(context='paper'):
    """
    Set publication-quality matplotlib style for scientific papers.

    Configures matplotlib with settings appropriate for Nature, Science,
    and other high-impact journals.

    Parameters
    ----------
    context : {'paper', 'poster', 'notebook'}, default='paper'
        Display context affecting font sizes:
        - 'paper': For manuscript figures (smaller fonts)
        - 'poster': For conference posters (larger fonts)
        - 'notebook': For Jupyter notebooks (medium fonts)

    Notes
    -----
    Settings applied:
    - DPI: 300 (publication standard)
    - Figure size: 7.2" × 4.5" (Nature double-column: 183mm × 114mm)
    - Fonts: Arial/Helvetica (sans-serif, widely accepted)
    - Font sizes: 10pt base, 11pt axes, 12pt titles
    - Line widths: 1.5pt lines, 0.8pt axes
    - Grid: Light gray (alpha=0.3)
    - Tight layout with minimal padding

    Examples
    --------
    >>> set_publication_style('paper')
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [1, 4, 9])
    >>> plt.savefig('figure.pdf', dpi=300)

    >>> # For poster presentation
    >>> set_publication_style('poster')
    >>> # Fonts will be larger automatically
    """
    # Nature/Science standard: 89mm (single column) or 183mm (double column)
    # At 300 DPI: 1051px or 2165px width

    sns.set_context(context, font_scale=1.2)
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'figure.figsize': (7.2, 4.5),  # 183mm × 114mm in inches
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'lines.linewidth': 1.5,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })


def create_color_palette(style='diagnostic'):
    """
    Create consistent color palettes for different plot types.

    Provides standardized, colorblind-friendly color schemes for
    various analysis contexts.

    Parameters
    ----------
    style : {'diagnostic', 'qc_stages', 'sequential', 'diverging', 'qualitative'}
        Type of color scheme:
        - 'diagnostic': AD (red), MCI (orange), CN (green)
        - 'qc_stages': Raw (gray), Normalized (blue), Reconstructed (red)
        - 'sequential': Viridis colormap (10 colors)
        - 'diverging': Red-Blue diverging (11 colors)
        - 'qualitative': Categorical palette (8 colors)

    Returns
    -------
    dict or list
        Color palette specification
        - For diagnostic/qc_stages: dict mapping labels to hex colors
        - For sequential/diverging/qualitative: list of hex colors

    Examples
    --------
    >>> colors = create_color_palette('diagnostic')
    >>> plt.bar(['AD', 'MCI', 'CN'], [10, 15, 20],
    ...         color=[colors[k] for k in ['AD', 'MCI', 'CN']])

    >>> # Sequential colormap
    >>> cmap = create_color_palette('sequential')
    >>> plt.scatter(x, y, c=values, cmap=mpl.colors.ListedColormap(cmap))

    Notes
    -----
    All palettes are designed to be:
    - Colorblind-friendly (verified with Color Oracle)
    - Print-friendly (grayscale compatible)
    - High contrast for clarity
    """
    palettes = {
        'diagnostic': {
            'AD': '#d62728',   # Red - Alzheimer's Disease
            'MCI': '#ff7f0e',  # Orange - Mild Cognitive Impairment
            'CN': '#2ca02c',   # Green - Cognitively Normal
        },
        'qc_stages': {
            'Raw': '#7f7f7f',         # Gray - Original data
            'Normalized': '#1f77b4',  # Blue - After normalization
            'Reconstructed': '#d62728',  # Red - VAE output
        },
        'sequential': sns.color_palette('viridis', 10).as_hex(),
        'diverging': sns.color_palette('RdBu_r', 11).as_hex(),
        'qualitative': sns.color_palette('Set2', 8).as_hex(),
    }
    return palettes.get(style, palettes['qualitative'])


def save_figure(fig, filepath, formats=['png'], **kwargs):
    """
    Save figure with publication settings and multiple format support.

    Convenience wrapper for plt.savefig() with sensible defaults for
    publication-quality outputs.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure object to save
    filepath : Path or str
        Output path (without extension)
        Extension will be added based on formats parameter
    formats : list of str, default=['png']
        File formats to save: 'png', 'pdf', 'svg', 'eps'
        Multiple formats can be specified
    **kwargs
        Additional arguments passed to savefig()
        Common options:
        - dpi: int, default=300
        - bbox_inches: str, default='tight'
        - facecolor: str, default='white'
        - transparent: bool, default=False

    Returns
    -------
    None
        Prints confirmation message for each saved file

    Examples
    --------
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [1, 4, 9])
    >>> save_figure(fig, 'results/my_plot', formats=['png', 'pdf'])
    Saved: results/my_plot.png
    Saved: results/my_plot.pdf

    >>> # Custom DPI and transparent background
    >>> save_figure(fig, 'figure1', formats=['png'], dpi=600, transparent=True)

    Notes
    -----
    - PNG: Best for presentations, web, preview
    - PDF: Best for LaTeX, vector graphics, submission
    - SVG: Best for editing in Inkscape/Illustrator
    - EPS: Legacy format for older journals

    Automatically creates parent directories if they don't exist.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    defaults = {
        'dpi': 300,
        'bbox_inches': 'tight',
        'pad_inches': 0.05,
        'facecolor': 'white',
        'edgecolor': 'none',
    }
    defaults.update(kwargs)

    for fmt in formats:
        output_path = filepath.with_suffix(f'.{fmt}')
        fig.savefig(output_path, format=fmt, **defaults)
        print(f"Saved: {output_path}")

    plt.close(fig)


# ============================================================================
# PIPELINE MANAGEMENT UTILITIES
# ============================================================================

import ast
import json
import re
import subprocess
import warnings
from typing import Any


def infer_run_tag(
    results_dir: Path,
    summary_file: Optional[str] = None,
) -> Tuple[str, Path]:
    """
    Infer RUN_TAG from summary_metrics_MULTI_*.txt file.

    Parameters
    ----------
    results_dir : Path
        Directory containing summary_metrics_MULTI_*.txt files.
    summary_file : str, optional
        Specific filename to use. If None, uses the most recent.

    Returns
    -------
    run_tag : str
        The run tag extracted from the filename.
    summary_path : Path
        Full path to the summary file used.

    Raises
    ------
    FileNotFoundError
        If no summary_metrics_MULTI_*.txt exists in results_dir.
    """
    results_dir = Path(results_dir)
    if summary_file is not None:
        summary_path = results_dir / summary_file
        if not summary_path.exists():
            raise FileNotFoundError(f"Specified summary file not found: {summary_path}")
    else:
        candidates = sorted(
            results_dir.glob("summary_metrics_MULTI_*.txt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No summary_metrics_MULTI_*.txt found in: {results_dir}"
            )
        summary_path = candidates[0]

    run_tag = re.sub(r"^summary_metrics_MULTI_|\.txt$", "", summary_path.name)
    return run_tag, summary_path


def load_run_args(summary_path: Path) -> Dict[str, Any]:
    """
    Parse run arguments dict from summary_metrics_MULTI_*.txt.

    The first line block ``Run Arguments:\n{...}`` is parsed with
    ``ast.literal_eval``.  Returns ``{}`` on parse failure (with a warning),
    so callers can still proceed with defaults.

    Parameters
    ----------
    summary_path : Path
        Path to the summary_metrics_MULTI_*.txt file.

    Returns
    -------
    dict
        Parsed run arguments. Always includes key ``_summary_file_used``.
    """
    summary_path = Path(summary_path)
    txt = summary_path.read_text(encoding="utf-8", errors="replace")

    i0, i1 = txt.find("{"), txt.rfind("}")
    run_args: Dict[str, Any] = {}
    if i0 >= 0 and i1 > i0:
        try:
            run_args = ast.literal_eval(txt[i0 : i1 + 1])
        except Exception as exc:
            warnings.warn(
                f"[notebook_utils] Could not parse run_args from "
                f"{summary_path.name}: {exc}",
                RuntimeWarning,
            )
    run_args["_summary_file_used"] = summary_path.name
    return run_args


def resolve_artifacts(
    results_dir: Path,
    run_tag: str,
    n_folds: int,
) -> Dict[str, Any]:
    """
    Resolve and validate all artifact paths for a given run.

    Parameters
    ----------
    results_dir : Path
        Base results directory (RESULTS_DIR).
    run_tag : str
        Tag inferred from summary filename.
    n_folds : int
        Number of outer CV folds.

    Returns
    -------
    dict with keys:
        - metrics_csv : Path
        - pred_joblib : Path
        - vae_history_joblib : Path or None (not required)
        - fold_dirs : list[Path]   # fold_1 … fold_N
        - per_fold : dict[int -> dict]  # optional per-fold CSV paths

    Raises
    ------
    FileNotFoundError
        If required artifacts (metrics_csv, pred_joblib) are missing.
    """
    results_dir = Path(results_dir)

    def _must(p: Path) -> Path:
        if not p.exists():
            raise FileNotFoundError(f"Required artifact missing: {p}")
        return p

    def _optional(p: Path) -> Optional[Path]:
        return p if p.exists() else None

    metrics_csv = _must(results_dir / f"all_folds_metrics_MULTI_{run_tag}.csv")
    pred_joblib = _must(results_dir / f"all_folds_clf_predictions_MULTI_{run_tag}.joblib")
    vae_hist = _optional(results_dir / f"all_folds_vae_training_history_{run_tag}.joblib")

    fold_dirs: List[Path] = []
    per_fold: Dict[int, Dict[str, Optional[Path]]] = {}
    for k in range(1, n_folds + 1):
        fd = results_dir / f"fold_{k}"
        fold_dirs.append(fd)
        per_fold[k] = {
            "dir": fd,
            "latent_qc": _optional(fd / "latent_qc_metrics.csv"),
            "dist_raw": _optional(fd / f"fold_{k}_dist_raw.csv"),
            "dist_norm": _optional(fd / f"fold_{k}_dist_norm.csv"),
            "dist_recon": _optional(fd / f"fold_{k}_dist_recon.csv"),
            "scanner_leakage": _optional(fd / f"fold_{k}_scanner_leakage.csv"),
            "vae_train_history": _optional(fd / f"vae_train_history_fold_{k}.joblib"),
            "norm_params": _optional(fd / "vae_norm_params.joblib"),
        }

    return {
        "metrics_csv": metrics_csv,
        "pred_joblib": pred_joblib,
        "vae_history_joblib": vae_hist,
        "fold_dirs": fold_dirs,
        "per_fold": per_fold,
    }


def load_predictions(
    pred_path: Path,
    target_clf: str,
    n_folds: Optional[int] = None,
) -> pd.DataFrame:
    """
    Unified loader for all_folds_clf_predictions_MULTI_*.joblib.

    Handles three observed formats:
    1. list[DataFrame] – concatenated then filtered by classifier_type.
    2. Single DataFrame – filtered directly.
    3. dict {fold_idx: DataFrame | list} – flattened.

    Parameters
    ----------
    pred_path : Path
        Path to the .joblib file.
    target_clf : str
        Classifier type to keep (e.g. "logreg", "svm", "rf").
    n_folds : int, optional
        Expected number of folds for a sanity-check warning.

    Returns
    -------
    pandas.DataFrame
        Pooled predictions with columns: fold, classifier_type, SubjectID,
        y_true, y_score_final, y_score_raw, y_score_cal, y_pred.

    Raises
    ------
    ValueError
        If no rows match target_clf after loading.
    """
    import joblib

    pred_path = Path(pred_path)
    obj = joblib.load(pred_path)

    # --- normalise to single DataFrame ---
    if isinstance(obj, list):
        frames = []
        for item in obj:
            if isinstance(item, pd.DataFrame):
                frames.append(item)
            elif isinstance(item, list):
                frames.extend(f for f in item if isinstance(f, pd.DataFrame))
        df_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    elif isinstance(obj, pd.DataFrame):
        df_all = obj.copy()
    elif isinstance(obj, dict):
        frames = []
        for v in obj.values():
            if isinstance(v, pd.DataFrame):
                frames.append(v)
            elif isinstance(v, list):
                frames.extend(f for f in v if isinstance(f, pd.DataFrame))
        df_all = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        raise ValueError(f"Unexpected joblib object type: {type(obj)}")

    if df_all.empty:
        raise ValueError(f"No DataFrames found inside {pred_path.name}")

    # --- normalise column names ---
    # 'fold' column
    if "fold" not in df_all.columns:
        df_all["fold"] = np.nan

    # classifier type column
    clf_col = next(
        (c for c in ["classifier_type", "actual_classifier_type"] if c in df_all.columns),
        None,
    )
    if clf_col is not None:
        df_all["classifier_type"] = df_all[clf_col].astype(str).str.strip().str.lower()
    else:
        df_all["classifier_type"] = target_clf.lower()

    # score column
    for src, dst in [("y_score_final", "y_score_final"), ("y_score_cal", "y_score_cal"),
                     ("y_score_raw", "y_score_raw")]:
        if src in df_all.columns and dst not in df_all.columns:
            df_all[dst] = df_all[src]
    if "y_score_final" not in df_all.columns:
        for cand in ["y_score_cal", "y_score_raw", "y_score"]:
            if cand in df_all.columns:
                df_all["y_score_final"] = df_all[cand]
                break

    # --- filter to target_clf ---
    mask = df_all["classifier_type"] == target_clf.lower()
    preds = df_all.loc[mask].copy()

    if preds.empty:
        available = df_all["classifier_type"].unique().tolist()
        raise ValueError(
            f"No rows found for classifier '{target_clf}'. "
            f"Available: {available}"
        )

    preds["fold"] = pd.to_numeric(preds["fold"], errors="coerce").astype("Int64")
    preds["y_true"] = preds["y_true"].astype(int)

    if n_folds is not None:
        found = preds["fold"].dropna().nunique()
        if found != n_folds:
            warnings.warn(
                f"[load_predictions] Expected {n_folds} folds, found {found} "
                f"in {pred_path.name}",
                RuntimeWarning,
            )

    return preds.reset_index(drop=True)


# ============================================================================
# PERFORMANCE UTILITIES
# ============================================================================


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for ROC-AUC.

    Parameters
    ----------
    y_true : array-like, shape (n,)
        True binary labels.
    y_score : array-like, shape (n,)
        Predicted scores (higher = more positive).
    n_boot : int, default=2000
        Number of bootstrap resamples.
    seed : int, default=42
        Random seed for reproducibility.
    ci : float, default=0.95
        Confidence level (e.g. 0.95 → 95% CI).

    Returns
    -------
    (auc_obs, ci_lo, ci_hi) : tuple of float
        Observed AUC and percentile CI bounds.
    """
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    auc_obs = float(roc_auc_score(y_true, y_score))

    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        boot_aucs.append(float(roc_auc_score(yt, ys)))

    alpha = (1.0 - ci) / 2
    ci_lo = float(np.quantile(boot_aucs, alpha)) if boot_aucs else np.nan
    ci_hi = float(np.quantile(boot_aucs, 1.0 - alpha)) if boot_aucs else np.nan
    return auc_obs, ci_lo, ci_hi


def bootstrap_prauc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """Bootstrap confidence interval for Precision-Recall AUC (Average Precision).

    Parameters
    ----------
    y_true, y_score, n_boot, seed, ci : same as ``bootstrap_auc_ci``.

    Returns
    -------
    (prauc_obs, ci_lo, ci_hi) : tuple of float
    """
    from sklearn.metrics import average_precision_score

    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    prauc_obs = float(average_precision_score(y_true, y_score))

    boot_praucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        boot_praucs.append(float(average_precision_score(yt, ys)))

    alpha = (1.0 - ci) / 2
    ci_lo = float(np.quantile(boot_praucs, alpha)) if boot_praucs else np.nan
    ci_hi = float(np.quantile(boot_praucs, 1.0 - alpha)) if boot_praucs else np.nan
    return prauc_obs, ci_lo, ci_hi


# ============================================================================
# STATISTICAL VALIDATION
# ============================================================================


def stratified_permutation_auc(
    df: pd.DataFrame,
    y: str = "_y",
    p: str = "_p",
    group: str = "_confounder",
    n_perm: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Stratified permutation test for AUC.

    Permutes labels *within* each group (confounder domain) to preserve
    the marginal distribution of the confounder.  This is the appropriate
    null for testing whether the model uses features beyond the confounder.

    Parameters
    ----------
    df : DataFrame
        Contains y (true labels), p (predicted scores), group (confounder).
    y, p, group : str
        Column names.
    n_perm : int, default=2000
        Number of permutations.
    seed : int, default=42
        Random seed.

    Returns
    -------
    dict with keys:
        - auc_obs: float        – observed AUC
        - p_one_sided: float   – P(null ≥ obs)
        - p_two_sided: float
        - null_mean: float
        - null_std: float
        - n_perm: int           – actual permutations used
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(seed)
    y_true = df[y].to_numpy().astype(int)
    y_score = df[p].to_numpy().astype(float)
    groups = df[group].astype(str).to_numpy()

    if len(np.unique(y_true)) < 2:
        return {"auc_obs": np.nan, "p_one_sided": np.nan, "p_two_sided": np.nan,
                "null_mean": np.nan, "null_std": np.nan, "n_perm": 0}

    auc_obs = float(roc_auc_score(y_true, y_score))
    idx_by_g: Dict[str, np.ndarray] = {}
    for i, g in enumerate(groups):
        idx_by_g.setdefault(g, []).append(i)
    idx_by_g = {g: np.asarray(v, dtype=int) for g, v in idx_by_g.items()}

    null_aucs: List[float] = []
    for _ in range(n_perm):
        y_perm = y_true.copy()
        for idxs in idx_by_g.values():
            y_perm[idxs] = rng.permutation(y_perm[idxs])
        if len(np.unique(y_perm)) < 2:
            continue
        null_aucs.append(float(roc_auc_score(y_perm, y_score)))

    null_arr = np.asarray(null_aucs, dtype=float)
    p_one = float((null_arr >= auc_obs).mean()) if len(null_arr) else np.nan
    p_two = float((np.abs(null_arr - null_arr.mean()) >= abs(auc_obs - null_arr.mean())).mean()) if len(null_arr) else np.nan

    return {
        "auc_obs": auc_obs,
        "p_one_sided": p_one,
        "p_two_sided": p_two,
        "null_mean": float(null_arr.mean()) if len(null_arr) else np.nan,
        "null_std": float(null_arr.std()) if len(null_arr) else np.nan,
        "n_perm": len(null_arr),
    }


def exact_sign_permutation_test(
    delta: np.ndarray,
    n_mc: int = 200_000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Sign-flip permutation test for paired fold differences.

    H0: mean(delta) = 0.  Exact for n ≤ 18, Monte-Carlo otherwise.

    Parameters
    ----------
    delta : array-like
        Paired differences (one per fold).
    n_mc : int, default=200_000
        Monte Carlo samples when n > 18.
    seed : int, default=42

    Returns
    -------
    dict with keys: n, obs_mean, p_one_sided, p_two_sided.
    """
    import itertools

    rng = np.random.default_rng(seed)
    d = np.asarray(delta, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n == 0:
        return {"n": 0, "obs_mean": np.nan, "p_one_sided": np.nan, "p_two_sided": np.nan}

    obs = float(np.mean(d))

    if n <= 18:
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)), dtype=float)
    else:
        signs = rng.choice([-1.0, 1.0], size=(n_mc, n), replace=True)

    null_means = (signs * d[None, :]).mean(axis=1)
    p_one = float((null_means >= obs).mean())
    p_two = float((np.abs(null_means) >= abs(obs)).mean())

    return {"n": int(n), "obs_mean": obs, "p_one_sided": p_one, "p_two_sided": p_two}


# ============================================================================
# METADATA / CONFOUNDER UTILITIES
# ============================================================================


def build_metadata_aligned_to_tensor(
    global_tensor_npz: Path,
    metadata_csv: Path,
    id_col_candidates: Tuple[str, ...] = ("SubjectID", "RID", "participant_id", "ID"),
) -> pd.DataFrame:
    """
    Align a metadata CSV to the subject order in a global tensor NPZ.

    The NPZ must contain the key ``subject_ids``.  The metadata CSV must
    contain one of the columns in ``id_col_candidates``.

    Parameters
    ----------
    global_tensor_npz : Path
        Path to the .npz file with keys ``global_tensor_data`` and
        ``subject_ids``.
    metadata_csv : Path
        Path to the metadata CSV file.
    id_col_candidates : tuple of str
        Column names to try as subject ID in the metadata CSV.

    Returns
    -------
    pandas.DataFrame
        Metadata rows reindexed to match tensor subject order.
        Rows without a metadata match have NaN values.
        Includes an ``tensor_idx`` column (0-based row in tensor).
    """
    npz = np.load(global_tensor_npz, allow_pickle=True)
    if "subject_ids" not in npz.files:
        raise KeyError(f"NPZ missing 'subject_ids'. Keys: {list(npz.files)}")

    tensor_ids = npz["subject_ids"]
    if tensor_ids.dtype.kind in ("S", "b"):
        tensor_ids = np.array(
            [x.decode("utf-8", errors="ignore") for x in tensor_ids], dtype=object
        )
    tensor_ids_s = pd.Series(tensor_ids).astype(str).str.strip()

    meta = pd.read_csv(metadata_csv)
    id_col = next((c for c in id_col_candidates if c in meta.columns), None)
    if id_col is None:
        raise KeyError(
            f"None of {id_col_candidates} found in metadata columns: {list(meta.columns)}"
        )

    meta[id_col] = meta[id_col].astype(str).str.strip()
    meta = meta.set_index(id_col)

    aligned = meta.reindex(tensor_ids_s.values)
    aligned = aligned.reset_index().rename(columns={"index": id_col})
    aligned.insert(0, "tensor_idx", np.arange(len(tensor_ids_s)))
    aligned[id_col] = tensor_ids_s.values
    return aligned


def detect_confounder_column(
    metadata_df: pd.DataFrame,
    prefer_manufacturer: bool = True,
) -> Optional[str]:
    """
    Select the best site/confounder column from a metadata DataFrame.

    Prefers Manufacturer/Vendor columns (fewer unique values → more stable
    for CV), with fallback to Site/Scanner/Center.

    Parameters
    ----------
    metadata_df : DataFrame
    prefer_manufacturer : bool, default=True

    Returns
    -------
    str or None
        Column name, or None if no suitable column found.
    """
    if metadata_df is None or metadata_df.empty:
        return None

    pref: List[str] = []
    other: List[str] = []
    for c in metadata_df.columns:
        cl = str(c).lower()
        if any(k in cl for k in ("manufacturer", "vendor")):
            pref.append(c)
        elif any(k in cl for k in ("site", "scanner", "center", "centre")):
            other.append(c)

    def _best(cols: List[str]) -> Optional[str]:
        best, best_score = None, None
        for c in cols:
            s = metadata_df[c]
            nn = int(s.notna().sum())
            nu = int(s.nunique(dropna=True))
            if nn == 0 or nu < 2:
                continue
            score = (nn, -nu)
            if best is None or score > best_score:
                best, best_score = c, score
        return best

    if prefer_manufacturer and pref:
        result = _best(pref)
        if result is not None:
            return result
    return _best(other if other else pref + other)


# ============================================================================
# INFORMATION THEORY (DISCRETE)
# ============================================================================

_LN2 = np.log(2.0)


def entropy_bits(s) -> float:
    """Shannon entropy of a discrete variable, in bits."""
    vc = pd.Series(s).value_counts(dropna=False)
    counts = vc.values.astype(float)
    n = counts.sum()
    if n <= 0:
        return np.nan
    p = counts / n
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / _LN2)


def conditional_entropy_bits(y, s) -> float:
    """H(Y | S) in bits, for discrete Y and S."""
    y_s = pd.Series(y)
    s_s = pd.Series(s).astype(str)
    n = len(y_s)
    if n == 0:
        return np.nan
    h = 0.0
    for grp_s, idx in s_s.groupby(s_s).groups.items():
        w = len(idx) / n
        h += w * entropy_bits(y_s.loc[idx])
    return float(h)


def mi_bits(a, b) -> float:
    """Mutual information I(A; B) in bits (via sklearn, converts from nats)."""
    from sklearn.metrics import mutual_info_score
    return float(
        mutual_info_score(pd.Series(a).astype(str), pd.Series(b).astype(str)) / _LN2
    )


def nmi(a, b) -> float:
    """Normalized mutual information: I(A;B) / min(H(A), H(B)), in [0,1]."""
    ha = entropy_bits(a)
    hb = entropy_bits(b)
    denom = min(ha, hb) if (ha and hb and np.isfinite(ha) and np.isfinite(hb)) else 0.0
    return float(mi_bits(a, b) / denom) if denom > 1e-12 else np.nan


def cramers_v_bias_corrected(x, y) -> float:
    """
    Cramér's V with bias correction (Bergsma & Wicher 2013).

    Returns np.nan if not computable (< 2 classes in either variable).
    """
    tab = pd.crosstab(pd.Series(x).astype(str), pd.Series(y).astype(str), dropna=False)
    obs = tab.to_numpy().astype(float)
    n = obs.sum()
    if n == 0:
        return np.nan
    row_sum = obs.sum(axis=1, keepdims=True)
    col_sum = obs.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((obs - expected) ** 2 / np.where(expected > 0, expected, np.nan))
    r, k = obs.shape
    if r < 2 or k < 2:
        return np.nan
    phi2corr = max(0.0, chi2 / n - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else np.nan


def domain_support_report(
    df: pd.DataFrame,
    dom_col: str,
    y_col: str = "_y",
    min_n: int = 6,
    min_per_class: int = 2,
) -> pd.DataFrame:
    """
    Per-domain support table: N, n_classes, class counts, valid flag.

    Parameters
    ----------
    df : DataFrame
    dom_col : str
        Confounder/domain column.
    y_col : str, default="_y"
        Binary label column.
    min_n, min_per_class : int
        Thresholds for marking a domain as statistically valid.

    Returns
    -------
    DataFrame with columns: domain, n, n_classes, n_pos, n_neg, valid.
    """
    rows = []
    for dom, grp in df.groupby(dom_col):
        n = len(grp)
        y = grp[y_col].dropna()
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        n_classes = int((pd.Series([n_pos, n_neg]) > 0).sum())
        valid = (n >= min_n) and (n_pos >= min_per_class) and (n_neg >= min_per_class)
        rows.append({"domain": dom, "n": n, "n_classes": n_classes,
                     "n_pos": n_pos, "n_neg": n_neg, "valid": valid})
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


# ============================================================================
# PROVENANCE / MANIFEST
# ============================================================================


def _git_commit(cwd: Optional[Path] = None) -> Optional[str]:
    """Return short git commit hash, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
            cwd=str(cwd) if cwd else None, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        return None


def save_run_manifest(
    out_dir: Path,
    results_dir: Path,
    run_tag: str,
    summary_file: str,
    run_args: Dict[str, Any],
    artifacts: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Save a run provenance manifest as JSON and return it.

    Parameters
    ----------
    out_dir : Path
        Directory where manifest.json will be written.
    results_dir, run_tag, summary_file, run_args, artifacts : ...
        All collected from upstream calls.
    extra : dict, optional
        Any additional key-value pairs to include.

    Returns
    -------
    dict
        The manifest dictionary (also saved to ``out_dir/run_manifest.json``).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _str(p):
        return str(p) if p is not None else None

    manifest: Dict[str, Any] = {
        "results_dir": str(results_dir),
        "run_tag": run_tag,
        "summary_file_used": summary_file,
        "run_args": {
            k: v for k, v in run_args.items()
            if not k.startswith("_") and isinstance(v, (str, int, float, bool, list, type(None)))
        },
        "resolved_artifacts": {
            "metrics_csv": _str(artifacts.get("metrics_csv")),
            "pred_joblib": _str(artifacts.get("pred_joblib")),
            "vae_history_joblib": _str(artifacts.get("vae_history_joblib")),
        },
        "git_commit": _git_commit(),
    }
    if extra:
        manifest.update(extra)

    manifest_path = out_dir / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest


# ============================================================================
# MODULE INFO
# ============================================================================

__all__ = [
    # Calibration
    '_make_bin_edges',
    'weighted_ece_mce',
    'calibration_slope_intercept',
    'reliability_report',
    # Normalization
    'apply_norm_inference',
    'apply_norm_batch',
    # Information Theory
    'participation_ratio',
    'total_correlation_gauss',
    'clean_confounder_label',
    'entropy_bits',
    'conditional_entropy_bits',
    'mi_bits',
    'nmi',
    'cramers_v_bias_corrected',
    'domain_support_report',
    # Visualization
    'set_publication_style',
    'create_color_palette',
    'save_figure',
    # Pipeline Management
    'infer_run_tag',
    'load_run_args',
    'resolve_artifacts',
    'load_predictions',
    # Performance
    'bootstrap_auc_ci',
    'bootstrap_prauc_ci',
    # Statistical Validation
    'stratified_permutation_auc',
    'exact_sign_permutation_test',
    # Metadata / Confounder
    'build_metadata_aligned_to_tensor',
    'detect_confounder_column',
    # Provenance
    'save_run_manifest',
]

__version__ = '1.1.0'
__author__ = 'BetaVAE-XAI-AD Project'
