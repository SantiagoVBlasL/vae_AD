"""
Interpretability Utilities for BetaVAE-XAI-AD Project.

This module provides comprehensive utilities for analyzing and interpreting
VAE+Classifier models using SHAP, Integrated Gradients, and disentanglement metrics.
Includes statistical rigor (bootstrap CIs, permutation tests, FDR correction),
stability assessment, and network-level aggregation.

Functions are organized into categories:
- Feature naming and data loading
- Visualization helpers
- Statistical testing (bootstrap, permutation, FDR)
- Stability metrics (Jaccard, Dice, ICC)
- Effect sizes (Cohen's d)
- Network aggregation
"""

from pathlib import Path
import re
from typing import Dict, List, Tuple, Optional, Union
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.collections
from scipy import stats
from sklearn.metrics import r2_score
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import LabelEncoder


# ============================================================================
# Section 1: Feature Naming and Data Loading
# ============================================================================

def prettify_feature(name: str) -> str:
    """
    Convert raw feature names to human-readable format.

    Transforms internal feature names (e.g., '__Age', 'latent_42') into
    clean display names for plots and tables.

    Parameters
    ----------
    name : str
        Raw feature name from SHAP values or model output.

    Returns
    -------
    str
        Prettified feature name. Examples:
        - '__Age' → 'Age'
        - 'latent_42' → 'latent 42'
        - 'Sex' → 'Sex'

    Examples
    --------
    >>> prettify_feature('__Age')
    'Age'
    >>> prettify_feature('latent_162')
    'latent 162'
    """
    if re.search(r'(?:^|__)Age\b', name):
        return "Age"
    if re.search(r'(?:^|__)Sex\b', name):
        return "Sex"
    m = re.search(r'latent_(\d+)', name)
    if m:
        return f"latent {m.group(1)}"
    return name


def load_shap_pack(run_dir: Path, fold: int, clf: str, tag: str) -> Dict:
    """
    Load SHAP analysis pack from a specific fold.

    Parameters
    ----------
    run_dir : Path
        Root directory of experimental run (e.g., results/vae_clf_ad_ld256_beta66_manual).
    fold : int
        Fold number (1-5 for 5-fold CV).
    clf : str
        Classifier type ('svm', 'mlp', etc.).
    tag : str
        Analysis tag identifying the SHAP variant:
        - 'frozen': Age/Sex frozen at training stats
        - 'unfrozen': All features vary freely

    Returns
    -------
    dict
        SHAP pack containing:
        - 'shap_values': ndarray (N, F) - SHAP values for test set
        - 'feature_names': list - Feature names
        - 'X_test': DataFrame - Test features
        - 'y_test': array - Test labels
        - 'latent_feature_mask': array - Boolean mask for latent dimensions

    Raises
    ------
    FileNotFoundError
        If the SHAP pack file does not exist.

    Examples
    --------
    >>> pack = load_shap_pack(run_dir, fold=1, clf='svm', tag='frozen')
    >>> shap_vals = pack['shap_values']  # (37, 258)
    """
    path = run_dir / f"fold_{fold}" / "interpretability_shap" / f"shap_pack_{clf}_{tag}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"SHAP pack not found: {path}")
    return joblib.load(path)


def edge_key_df(df: pd.DataFrame) -> pd.Series:
    """
    Create undirected edge keys from dataframe with source/destination ROI names.

    For brain connectivity analysis, edges are undirected (A→B == B→A).
    This function creates canonical edge keys by sorting ROI pairs alphabetically.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'src_AAL3_Name' and 'dst_AAL3_Name'.

    Returns
    -------
    pd.Series
        Series of tuples representing undirected edges: (roi_a, roi_b)
        where roi_a <= roi_b lexicographically.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'src_AAL3_Name': ['Frontal_L', 'Temporal_R'],
    ...     'dst_AAL3_Name': ['Parietal_R', 'Frontal_L']
    ... })
    >>> edge_key_df(df)
    0    (Frontal_L, Parietal_R)
    1    (Frontal_L, Temporal_R)
    dtype: object
    """
    a = df["src_AAL3_Name"].astype(str)
    b = df["dst_AAL3_Name"].astype(str)
    return pd.DataFrame({"a": a, "b": b}).apply(
        lambda r: tuple(sorted((r["a"], r["b"]))),
        axis=1
    )


# ============================================================================
# Section 2: Visualization Helpers
# ============================================================================

def ajustar_opacidad_violin(
    alpha: float = 0.35,
    subir_puntos: bool = True,
    ax: Optional[plt.Axes] = None
) -> None:
    """
    Adjust violin plot opacity and emphasize scatter points in SHAP plots.

    SHAP's default violin plots have opaque violins that obscure the beeswarm.
    This function reduces violin opacity and brings scatter points to the front.

    Parameters
    ----------
    alpha : float, default=0.35
        Opacity level for violin patches (0=transparent, 1=opaque).
    subir_puntos : bool, default=True
        If True, bring scatter points to front layer (zorder=3) and add white edges.
    ax : plt.Axes, optional
        Target axes. If None, uses current axes (plt.gca()).

    Examples
    --------
    >>> shap.summary_plot(shap_values, X_test, show=False)
    >>> ajustar_opacidad_violin(alpha=0.25)
    >>> plt.savefig('shap_violin.png')
    """
    ax = ax or plt.gca()

    # Reduce opacity of violin patches
    for pc in ax.findobj(mpl.collections.PolyCollection):
        pc.set_alpha(alpha)
    for p in ax.patches:
        try:
            p.set_alpha(alpha)
        except Exception:
            pass

    # Bring scatter points to front
    if subir_puntos:
        for sc in ax.findobj(mpl.collections.PathCollection):
            sc.set_zorder(3)
            sc.set_edgecolor("white")
            sc.set_linewidth(0.4)


def save_multi(base_path: Path, dpi: int = 300) -> None:
    """
    Save current matplotlib figure in multiple formats (PNG, SVG, PDF).

    For publication-quality figures, saves in raster (PNG) and vector (SVG, PDF)
    formats simultaneously.

    Parameters
    ----------
    base_path : Path
        Output path without extension (e.g., 'figures/shap_violin').
    dpi : int, default=300
        Resolution for raster formats (300 is publication standard).

    Examples
    --------
    >>> plt.figure()
    >>> plt.plot([1, 2, 3])
    >>> save_multi(Path('output/my_plot'), dpi=300)
    # Creates: output/my_plot.png, output/my_plot.svg, output/my_plot.pdf
    """
    base_path = Path(base_path)
    for ext in (".png", ".svg", ".pdf"):
        out = base_path.with_suffix(ext)
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {base_path.with_suffix('.png')} (+ svg/pdf)")


# ============================================================================
# Section 3: Statistical Testing
# ============================================================================

def bootstrap_shap_importance_ci(
    shap_pack_list: List[Dict],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42
) -> pd.DataFrame:
    """
    Compute bootstrap confidence intervals for SHAP feature importance rankings.

    Pools SHAP values from multiple folds and computes percentile bootstrap CIs
    for mean(|SHAP|) per feature. This quantifies uncertainty in feature rankings.

    Parameters
    ----------
    shap_pack_list : list of dict
        SHAP packs from multiple folds (output of load_shap_pack).
    n_bootstrap : int, default=1000
        Number of bootstrap resamples.
    alpha : float, default=0.05
        Significance level (0.05 → 95% CIs).
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Columns:
        - 'feature': Feature name
        - 'mean_abs_shap': Point estimate of mean(|SHAP|)
        - 'ci_lower_95': Lower bound of 95% CI
        - 'ci_upper_95': Upper bound of 95% CI
        - 'std_bootstrap': Bootstrap standard error
        Sorted by mean_abs_shap descending.

    Notes
    -----
    Bootstrap procedure:
    1. Pool SHAP values from all folds (N_total samples)
    2. Resample with replacement B times
    3. Compute mean(|SHAP|) per bootstrap
    4. Extract percentile CIs

    References
    ----------
    Efron & Tibshirani (1993). An Introduction to the Bootstrap.

    Examples
    --------
    >>> packs = [load_shap_pack(run_dir, f, 'svm', 'frozen') for f in range(1, 6)]
    >>> df_ci = bootstrap_shap_importance_ci(packs, n_bootstrap=1000)
    >>> print(df_ci.head(10))  # Top 10 features with CIs
    """
    # Pool SHAP values from all folds
    shap_matrix = np.vstack([pack['shap_values'] for pack in shap_pack_list])
    feature_names = shap_pack_list[0]['feature_names']
    N, F = shap_matrix.shape

    # Bootstrap resampling
    rng = np.random.default_rng(seed)
    boot_means = np.zeros((n_bootstrap, F))

    for b in range(n_bootstrap):
        idx = rng.choice(N, size=N, replace=True)
        boot_means[b, :] = np.abs(shap_matrix[idx, :]).mean(axis=0)

    # Compute CIs
    lower_pct = (alpha / 2) * 100
    upper_pct = (1 - alpha / 2) * 100
    ci_lower = np.percentile(boot_means, lower_pct, axis=0)
    ci_upper = np.percentile(boot_means, upper_pct, axis=0)
    mean_shap = np.abs(shap_matrix).mean(axis=0)

    df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_shap,
        'ci_lower_95': ci_lower,
        'ci_upper_95': ci_upper,
        'std_bootstrap': boot_means.std(axis=0)
    })

    return df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)


def permutation_test_shap_feature(
    shap_vals: np.ndarray,
    feature_idx: int,
    n_perm: int = 1000,
    seed: int = 42
) -> Dict[str, float]:
    """
    Test if SHAP values for a feature are significantly different from zero.

    Null hypothesis: SHAP values are random noise (symmetric around zero).
    Test statistic: mean(|SHAP|)

    Parameters
    ----------
    shap_vals : np.ndarray, shape (N, F)
        SHAP value matrix for all features.
    feature_idx : int
        Index of feature to test.
    n_perm : int, default=1000
        Number of permutations for null distribution.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict
        - 'observed_stat': Observed mean(|SHAP|)
        - 'p_value': One-tailed p-value (proportion of permuted stats >= observed)

    Notes
    -----
    Permutation strategy: Random sign flipping preserves magnitude distribution
    under the null hypothesis that SHAP values have no directional information.

    Examples
    --------
    >>> result = permutation_test_shap_feature(shap_matrix, feature_idx=0)
    >>> print(f"p-value: {result['p_value']:.4f}")
    """
    obs_stat = np.abs(shap_vals[:, feature_idx]).mean()

    rng = np.random.default_rng(seed)
    perm_stats = np.zeros(n_perm)

    for i in range(n_perm):
        signs = rng.choice([-1, 1], size=shap_vals.shape[0])
        perm_vals = shap_vals[:, feature_idx] * signs
        perm_stats[i] = np.abs(perm_vals).mean()

    p_value = (perm_stats >= obs_stat).sum() / n_perm

    return {
        'observed_stat': float(obs_stat),
        'p_value': float(p_value)
    }


def compute_fdr_correction(
    p_values: np.ndarray,
    alpha: float = 0.05,
    method: str = 'fdr_bh'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply FDR correction to p-values (Benjamini-Hochberg procedure).

    Parameters
    ----------
    p_values : np.ndarray
        Array of uncorrected p-values.
    alpha : float, default=0.05
        FDR threshold.
    method : str, default='fdr_bh'
        Correction method ('fdr_bh' = Benjamini-Hochberg).

    Returns
    -------
    reject : np.ndarray of bool
        Boolean array: True if null hypothesis is rejected at FDR level alpha.
    pvals_corrected : np.ndarray
        FDR-corrected p-values.

    Notes
    -----
    Benjamini-Hochberg controls the False Discovery Rate (FDR), the expected
    proportion of false positives among all rejections. Less conservative than
    Bonferroni for large-scale testing.

    References
    ----------
    Benjamini & Hochberg (1995). Controlling the false discovery rate:
    a practical and powerful approach to multiple testing. JRSS-B 57(1):289-300.

    Examples
    --------
    >>> p_vals = np.array([0.001, 0.05, 0.3, 0.8])
    >>> reject, p_corr = compute_fdr_correction(p_vals, alpha=0.05)
    >>> print(reject)  # [True, True, False, False]
    """
    from statsmodels.stats.multitest import multipletests

    reject, pvals_corrected, _, _ = multipletests(
        p_values,
        alpha=alpha,
        method=method
    )

    return reject, pvals_corrected


# ============================================================================
# Section 4: Stability Metrics
# ============================================================================

def compute_jaccard_stability(edge_sets: List[set]) -> float:
    """
    Compute mean pairwise Jaccard similarity across fold sets.

    Jaccard index: |A ∩ B| / |A ∪ B|
    Measures overlap between top-k feature/edge sets from different folds.

    Parameters
    ----------
    edge_sets : list of set
        List of sets (one per fold) containing top-k features/edges.

    Returns
    -------
    float
        Mean Jaccard similarity across all fold pairs.
        Range: [0, 1], where 1 = perfect agreement, 0 = no overlap.

    Notes
    -----
    - Jaccard > 0.5: Good stability
    - Jaccard < 0.2: Poor reproducibility, investigate further

    Examples
    --------
    >>> set1 = {'feat_1', 'feat_2', 'feat_3'}
    >>> set2 = {'feat_2', 'feat_3', 'feat_4'}
    >>> compute_jaccard_stability([set1, set2])
    0.5  # 2 shared / 4 total
    """
    from itertools import combinations

    jaccard_scores = []
    for set_i, set_j in combinations(edge_sets, 2):
        intersection = len(set_i & set_j)
        union = len(set_i | set_j)
        jaccard = intersection / union if union > 0 else 0.0
        jaccard_scores.append(jaccard)

    return float(np.mean(jaccard_scores)) if jaccard_scores else 0.0


def compute_dice_coefficient(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Compute Dice coefficient between two binary masks.

    Dice = 2|A ∩ B| / (|A| + |B|)
    More lenient than Jaccard, emphasizes overlap over union.

    Parameters
    ----------
    mask1, mask2 : np.ndarray of bool
        Binary masks of same shape.

    Returns
    -------
    float
        Dice coefficient in [0, 1].

    Examples
    --------
    >>> mask1 = np.array([True, True, False, False])
    >>> mask2 = np.array([True, False, True, False])
    >>> compute_dice_coefficient(mask1, mask2)
    0.5  # 2*1 / (2+2)
    """
    intersection = np.logical_and(mask1, mask2).sum()
    dice = 2 * intersection / (mask1.sum() + mask2.sum())
    return float(dice)


def compute_icc_across_folds(
    df_per_fold: pd.DataFrame,
    feature_col: str = 'feature',
    value_col: str = 'mean_abs_shap'
) -> float:
    """
    Compute Intraclass Correlation Coefficient (ICC) for feature rankings.

    ICC(2,1) measures consistency of feature importance rankings across folds.
    Treats folds as random raters and features as subjects.

    Parameters
    ----------
    df_per_fold : pd.DataFrame
        Long-format dataframe with columns:
        - feature_col: Feature identifier
        - 'fold': Fold number
        - value_col: Importance value (e.g., mean_abs_shap)
    feature_col : str, default='feature'
        Column name for feature identifiers.
    value_col : str, default='mean_abs_shap'
        Column name for importance values.

    Returns
    -------
    float
        ICC(2,1) value. Interpretation:
        - > 0.9: Excellent
        - 0.75-0.9: Good
        - 0.5-0.75: Moderate
        - < 0.5: Poor

    Notes
    -----
    Uses pingouin.intraclass_corr with ICC type 2 (random raters,
    absolute agreement).

    References
    ----------
    Shrout & Fleiss (1979). Intraclass correlations: uses in assessing
    rater reliability. Psychological Bulletin 86(2):420-428.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'feature': ['f1', 'f1', 'f2', 'f2'],
    ...     'fold': [1, 2, 1, 2],
    ...     'mean_abs_shap': [0.5, 0.52, 0.3, 0.28]
    ... })
    >>> icc = compute_icc_across_folds(df)
    >>> print(f"ICC: {icc:.3f}")
    """
    try:
        import pingouin as pg
    except ImportError:
        raise ImportError("pingouin required for ICC. Install: pip install pingouin")

    icc_result = pg.intraclass_corr(
        data=df_per_fold,
        targets=feature_col,
        raters='fold',
        ratings=value_col
    )

    # Extract ICC2 (two-way random effects, absolute agreement)
    icc2 = icc_result.loc[icc_result['Type'] == 'ICC2', 'ICC'].values[0]
    return float(icc2)


# ============================================================================
# Section 5: Effect Sizes
# ============================================================================

def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between two groups.

    Cohen's d = (mean1 - mean2) / pooled_std

    Parameters
    ----------
    group1, group2 : np.ndarray
        Sample arrays for two groups (e.g., AD vs CN latent values).

    Returns
    -------
    float
        Cohen's d. Interpretation:
        - |d| > 0.8: Large effect
        - 0.5 < |d| < 0.8: Medium effect
        - 0.2 < |d| < 0.5: Small effect
        - |d| < 0.2: Negligible effect

    Notes
    -----
    Uses pooled standard deviation with Bessel correction (ddof=1).

    References
    ----------
    Cohen (1988). Statistical Power Analysis for the Behavioral Sciences, 2nd ed.

    Examples
    --------
    >>> ad_latent = np.random.randn(50) + 0.5
    >>> cn_latent = np.random.randn(50)
    >>> d = compute_cohens_d(ad_latent, cn_latent)
    >>> print(f"Cohen's d: {d:.2f}")
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    # Cohen's d
    d = (np.mean(group1) - np.mean(group2)) / pooled_std

    return float(d)


def compute_effect_sizes_matrix(
    Z_ad: np.ndarray,
    Z_cn: np.ndarray
) -> np.ndarray:
    """
    Compute Cohen's d for all latent dimensions between AD and CN groups.

    Parameters
    ----------
    Z_ad : np.ndarray, shape (N_ad, D)
        Latent representations for AD subjects.
    Z_cn : np.ndarray, shape (N_cn, D)
        Latent representations for CN subjects.

    Returns
    -------
    np.ndarray, shape (D,)
        Cohen's d for each latent dimension.

    Examples
    --------
    >>> Z_ad = np.random.randn(50, 256) + 0.3  # AD latents
    >>> Z_cn = np.random.randn(50, 256)        # CN latents
    >>> d_per_latent = compute_effect_sizes_matrix(Z_ad, Z_cn)
    >>> large_effects = np.where(np.abs(d_per_latent) > 0.8)[0]
    >>> print(f"Latents with large effect: {large_effects}")
    """
    D = Z_ad.shape[1]
    effect_sizes = np.zeros(D)

    for d in range(D):
        effect_sizes[d] = compute_cohens_d(Z_ad[:, d], Z_cn[:, d])

    return effect_sizes


# ============================================================================
# Section 6: Network Aggregation
# ============================================================================

def aggregate_saliency_by_network(
    saliency_map: np.ndarray,
    roi_info: pd.DataFrame,
    network_col: str = 'Refined_Network'
) -> pd.DataFrame:
    """
    Aggregate ROI-level saliency to functional network-level.

    Converts (R×R) connectivity saliency matrix to (N×N) network-level matrix
    by averaging within network blocks.

    Parameters
    ----------
    saliency_map : np.ndarray, shape (R, R)
        ROI-to-ROI saliency values (e.g., from Integrated Gradients).
    roi_info : pd.DataFrame
        ROI metadata with columns:
        - 'roi_name_in_tensor': ROI identifier
        - network_col: Network assignment (e.g., 'Yeo17_Network')
    network_col : str, default='Refined_Network'
        Column name in roi_info containing network labels.

    Returns
    -------
    pd.DataFrame, shape (N_networks, N_networks)
        Mean saliency between each pair of networks.
        Index and columns are network names.

    Notes
    -----
    Useful for interpreting saliency at the systems neuroscience level
    (e.g., Default Mode Network, Salience Network).

    Examples
    --------
    >>> sal_map = np.load('saliency_map.npy')  # (131, 131)
    >>> roi_info = pd.read_csv('roi_info.csv')
    >>> net_sal = aggregate_saliency_by_network(sal_map, roi_info, 'Yeo17_Network')
    >>> print(net_sal.shape)  # (17, 17) for Yeo17 atlas
    """
    R = saliency_map.shape[0]
    networks = roi_info[network_col].values
    unique_networks = sorted(set(networks))
    N = len(unique_networks)

    network_saliency = np.zeros((N, N))

    for i, net_i in enumerate(unique_networks):
        for j, net_j in enumerate(unique_networks):
            mask_i = networks == net_i
            mask_j = networks == net_j

            # Extract submatrix for this network pair
            submatrix = saliency_map[np.ix_(mask_i, mask_j)]
            network_saliency[i, j] = np.abs(submatrix).mean()

    return pd.DataFrame(
        network_saliency,
        index=unique_networks,
        columns=unique_networks
    )


# ============================================================================
# Section 7: Disentanglement and Factor Analysis
# ============================================================================

def compute_factor_predictability(
    Z: np.ndarray,
    factor_df: pd.DataFrame,
    factor_name: str,
    is_categorical: bool = False
) -> Dict[str, float]:
    """
    Measure how well each latent dimension predicts a ground-truth factor.

    For each latent dimension, fits a Ridge regression to predict the factor.
    Computes SAP score = top_R² - second_R² to measure dimension separability.

    Parameters
    ----------
    Z : np.ndarray, shape (N, D)
        Latent representations.
    factor_df : pd.DataFrame
        DataFrame containing ground-truth factors (Age, Sex, Diagnosis, etc.).
    factor_name : str
        Column name in factor_df to predict.
    is_categorical : bool, default=False
        If True, encodes factor as integer labels before regression.

    Returns
    -------
    dict
        - 'factor': Factor name
        - 'top_latent': Index of most predictive latent dimension
        - 'top_r2': R² score of top latent
        - 'sap_score': SAP score (higher = better separation)

    Notes
    -----
    SAP (Separated Attribute Predictability) measures if a single latent
    dimension captures each factor independently. High SAP indicates good
    disentanglement.

    References
    ----------
    Kumar et al. (2018). Variational Inference of Disentangled Latent Concepts
    from Unlabeled Observations. ICLR.

    Examples
    --------
    >>> Z = test_preds[latent_cols].values  # (185, 256)
    >>> metadata = pd.read_csv('metadata.csv')
    >>> result = compute_factor_predictability(Z, metadata, 'Age')
    >>> print(f"Age best predicted by latent {result['top_latent']}, R²={result['top_r2']:.3f}")
    """
    D = Z.shape[1]
    r2_scores = np.zeros(D)

    y = factor_df[factor_name].values
    if is_categorical:
        y = LabelEncoder().fit_transform(y)

    for d in range(D):
        z_d = Z[:, d].reshape(-1, 1)
        model = RidgeCV(alphas=[0.1, 1.0, 10.0])
        model.fit(z_d, y)
        r2_scores[d] = r2_score(y, model.predict(z_d))

    sorted_r2 = np.sort(r2_scores)[::-1]
    sap_score = sorted_r2[0] - sorted_r2[1] if len(sorted_r2) > 1 else sorted_r2[0]

    return {
        'factor': factor_name,
        'top_latent': int(np.argmax(r2_scores)),
        'top_r2': float(sorted_r2[0]),
        'sap_score': float(sap_score)
    }


# ============================================================================
# Section 8: Age Confounding Analysis
# ============================================================================

def analyze_age_confounding(
    shap_pack_list: List[Dict],
    metadata_df: pd.DataFrame,
    subject_id_col: str = 'Subject_ID',
    age_col: str = 'Age'
) -> pd.DataFrame:
    """
    Analyze correlation between SHAP values and Age to detect confounding.

    For each latent dimension, computes Pearson correlation between SHAP importance
    and subject age. High correlation indicates age-driven rather than disease-driven signal.

    Parameters
    ----------
    shap_pack_list : list of dict
        SHAP packs from multiple folds.
    metadata_df : pd.DataFrame
        Subject metadata with columns: subject_id_col, age_col.
    subject_id_col : str, default='Subject_ID'
        Column name for subject identifiers.
    age_col : str, default='Age'
        Column name for age values.

    Returns
    -------
    pd.DataFrame
        Columns:
        - 'latent': Latent dimension name
        - 'correlation_shap_age': Pearson r
        - 'p_value': Correlation p-value
        - 'p_value_fdr': FDR-corrected p-value
        - 'significant_fdr': Boolean, True if FDR < 0.05

    Notes
    -----
    Age is a known confounder in Alzheimer's disease studies. Latents with
    high SHAP-Age correlation may be capturing age effects rather than
    disease-specific pathology.

    Examples
    --------
    >>> packs = [load_shap_pack(run_dir, f, 'svm', 'frozen') for f in range(1, 6)]
    >>> metadata = pd.read_csv('SubjectsData.csv')
    >>> df_age = analyze_age_confounding(packs, metadata)
    >>> age_driven = df_age[df_age['significant_fdr']]
    >>> print(f"{len(age_driven)} latents significantly correlated with Age")
    """
    from scipy.stats import pearsonr

    # Pool SHAP values
    shap_matrix = np.vstack([pack['shap_values'] for pack in shap_pack_list])
    feature_names = shap_pack_list[0]['feature_names']

    # Extract Age values (requires merging with metadata)
    # This assumes X_test in shap_pack contains Subject_ID or Age directly
    age_values = []
    for pack in shap_pack_list:
        if 'X_test' in pack:
            X_test = pack['X_test']
            if age_col in X_test.columns:
                age_values.extend(X_test[age_col].values)
            else:
                # Try to merge with metadata
                warnings.warn(
                    f"Age column '{age_col}' not found in X_test. "
                    "Attempting to merge with metadata_df."
                )
                # This requires Subject_ID in X_test
                if subject_id_col in X_test.columns:
                    merged = X_test.merge(
                        metadata_df[[subject_id_col, age_col]],
                        on=subject_id_col,
                        how='left'
                    )
                    age_values.extend(merged[age_col].values)
                else:
                    raise ValueError(
                        f"Cannot extract Age: '{age_col}' not in X_test and "
                        f"'{subject_id_col}' not available for merging."
                    )

    if len(age_values) != len(shap_matrix):
        raise ValueError(
            f"Age values length ({len(age_values)}) does not match "
            f"SHAP matrix length ({len(shap_matrix)})"
        )

    age_array = np.array(age_values)

    # Compute correlations for latent features only
    results = []
    latent_cols = [c for c in feature_names if re.search(r'latent_\d+', c)]

    for lat_col in latent_cols:
        feat_idx = feature_names.index(lat_col)
        shap_d = shap_matrix[:, feat_idx]

        r, p = pearsonr(shap_d, age_array)
        results.append({
            'latent': lat_col,
            'correlation_shap_age': r,
            'p_value': p
        })

    df = pd.DataFrame(results)

    # FDR correction
    reject, pvals_fdr = compute_fdr_correction(df['p_value'].values, alpha=0.05)
    df['p_value_fdr'] = pvals_fdr
    df['significant_fdr'] = reject

    return df


# ============================================================================
# End of interpretability_utils.py
# ============================================================================
