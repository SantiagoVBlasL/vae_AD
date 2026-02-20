"""
Interpretability Analysis for BetaVAE-XAI-AD Project.

This package provides comprehensive tools for interpreting VAE+Classifier models:
- SHAP and Integrated Gradients analysis
- Statistical rigor (bootstrap CIs, permutation tests, FDR correction)
- Disentanglement metrics (Total Correlation, Participation Ratio, SAP)
- Stability assessment (Jaccard, Dice, ICC)
- Network-level aggregation for neuroscience interpretation
"""

from .interpretability_utils import (
    # Feature naming and data loading
    prettify_feature,
    load_shap_pack,
    edge_key_df,
    # Visualization helpers
    ajustar_opacidad_violin,
    save_multi,
    # Statistical testing
    bootstrap_shap_importance_ci,
    permutation_test_shap_feature,
    compute_fdr_correction,
    # Stability metrics
    compute_jaccard_stability,
    compute_dice_coefficient,
    compute_icc_across_folds,
    # Effect sizes
    compute_cohens_d,
    compute_effect_sizes_matrix,
    # Network aggregation
    aggregate_saliency_by_network,
    # Disentanglement
    compute_factor_predictability,
    # Age confounding
    analyze_age_confounding,
)

from .composite_edge_shap import (
    make_edge_index,
    make_edge_index_offdiag,
    make_edge_feature_names,
    make_edge_mapping_df,
    vectorize_tensor_to_edges,
    reconstruct_tensor_from_edges,
    select_top_edges,
    select_top_edges_per_channel,
    compute_train_edge_median,
    compute_frozen_meta_values,
    make_edge_predict_fn,
    validate_edge_roundtrip,
    extract_logreg_latent_weights,
)

__all__ = [
    'prettify_feature',
    'load_shap_pack',
    'edge_key_df',
    'ajustar_opacidad_violin',
    'save_multi',
    'bootstrap_shap_importance_ci',
    'permutation_test_shap_feature',
    'compute_fdr_correction',
    'compute_jaccard_stability',
    'compute_dice_coefficient',
    'compute_icc_across_folds',
    'compute_cohens_d',
    'compute_effect_sizes_matrix',
    'aggregate_saliency_by_network',
    'compute_factor_predictability',
    'analyze_age_confounding',
    # Composite edge-level SHAP
    'make_edge_index',
    'make_edge_index_offdiag',
    'make_edge_feature_names',
    'make_edge_mapping_df',
    'vectorize_tensor_to_edges',
    'reconstruct_tensor_from_edges',
    'select_top_edges',
    'select_top_edges_per_channel',
    'compute_train_edge_median',
    'compute_frozen_meta_values',
    'make_edge_predict_fn',
    'validate_edge_roundtrip',
    'extract_logreg_latent_weights',
]

__version__ = '1.0.0'
