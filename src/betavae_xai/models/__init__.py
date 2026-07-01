# src/betavae_xai/models/__init__.py

"""
Subpaquete de modelos para betavae_xai.
"""

from .convolutional_vae import (
    BLOCK_ORDER_CHOICES,
    CONDITIONING_MODE_CHOICES,
    ConvolutionalVAE,
    DROPOUT_SCOPE_CHOICES,
    build_vae_dropout_manifest,
    summarize_vae_dropout_manifest,
)
from .classifiers import get_classifier_and_grid, get_available_classifiers

__all__ = [
    "ConvolutionalVAE",
    "BLOCK_ORDER_CHOICES",
    "CONDITIONING_MODE_CHOICES",
    "DROPOUT_SCOPE_CHOICES",
    "build_vae_dropout_manifest",
    "summarize_vae_dropout_manifest",
    "get_classifier_and_grid",
    "get_available_classifiers",
]
