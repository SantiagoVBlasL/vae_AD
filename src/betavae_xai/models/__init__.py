# src/betavae_xai/models/__init__.py

"""
Subpaquete de modelos para betavae_xai.
"""

from .convolutional_vae import ConvolutionalVAE, DROPOUT_SCOPE_CHOICES
from .classifiers import get_classifier_and_grid, get_available_classifiers

__all__ = [
    "ConvolutionalVAE",
    "DROPOUT_SCOPE_CHOICES",
    "get_classifier_and_grid",
    "get_available_classifiers",
]
