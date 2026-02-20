"""
Analysis and Quality Control utilities for BetaVAE-XAI-AD project.

This package provides tools for analyzing VAE training quality,
calibration metrics, and generating publication-ready figures.
"""

from .notebook_utils import (
    # Calibration
    weighted_ece_mce,
    calibration_slope_intercept,
    reliability_report,
    # Normalization
    apply_norm_inference,
    apply_norm_batch,
    # Information Theory
    participation_ratio,
    total_correlation_gauss,
    clean_confounder_label,
    entropy_bits,
    conditional_entropy_bits,
    mi_bits,
    nmi,
    cramers_v_bias_corrected,
    domain_support_report,
    # Visualization
    set_publication_style,
    create_color_palette,
    save_figure,
    # Pipeline Management
    infer_run_tag,
    load_run_args,
    resolve_artifacts,
    load_predictions,
    # Performance
    bootstrap_auc_ci,
    bootstrap_prauc_ci,
    # Statistical Validation
    stratified_permutation_auc,
    exact_sign_permutation_test,
    # Metadata / Confounder
    build_metadata_aligned_to_tensor,
    detect_confounder_column,
    # Provenance
    save_run_manifest,
)

__all__ = [
    'weighted_ece_mce',
    'calibration_slope_intercept',
    'reliability_report',
    'apply_norm_inference',
    'apply_norm_batch',
    'participation_ratio',
    'total_correlation_gauss',
    'clean_confounder_label',
    'entropy_bits',
    'conditional_entropy_bits',
    'mi_bits',
    'nmi',
    'cramers_v_bias_corrected',
    'domain_support_report',
    'set_publication_style',
    'create_color_palette',
    'save_figure',
    'infer_run_tag',
    'load_run_args',
    'resolve_artifacts',
    'load_predictions',
    'bootstrap_auc_ci',
    'bootstrap_prauc_ci',
    'stratified_permutation_auc',
    'exact_sign_permutation_test',
    'build_metadata_aligned_to_tensor',
    'detect_confounder_column',
    'save_run_manifest',
]

__version__ = '1.1.0'
