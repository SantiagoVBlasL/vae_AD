#!/usr/bin/env python3
"""Synthetic smoke tests for VAE reconstruction loss modes.

No real data is loaded. No training is run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from run_vae_clf_ad_inference import (  # noqa: E402
    RECON_LOSS_MODE_CURRENT,
    RECON_LOSS_MODE_OFFDIAG_CHANNELMEAN,
    vae_loss_function,
    vae_reconstruction_loss,
)
from betavae_xai.models.convolutional_vae import ConvolutionalVAE  # noqa: E402


def assert_close(value: float, expected: float, *, name: str, tol: float = 1e-6) -> None:
    if abs(value - expected) > tol:
        raise AssertionError(f"{name}: got {value}, expected {expected}")


def main() -> int:
    batch_size = 4
    n_rois = 5
    offdiag = n_rois * (n_rois - 1)
    allpix = n_rois * n_rois
    mu = torch.zeros(batch_size, 8)
    logvar = torch.zeros(batch_size, 8)

    current_losses = []
    offdiag_losses = []
    for channels in (1, 2, 3):
        x = torch.zeros(batch_size, channels, n_rois, n_rois)
        recon = torch.ones_like(x)

        current_recon = vae_reconstruction_loss(recon, x, mode=RECON_LOSS_MODE_CURRENT).item()
        offdiag_recon = vae_reconstruction_loss(recon, x, mode=RECON_LOSS_MODE_OFFDIAG_CHANNELMEAN).item()
        current_losses.append(current_recon)
        offdiag_losses.append(offdiag_recon)

        assert_close(current_recon, channels * allpix, name=f"current_C{channels}")
        assert_close(offdiag_recon, offdiag, name=f"offdiag_channelmean_C{channels}")

        total, recon_loss, kld_loss = vae_loss_function(
            recon,
            x,
            mu,
            logvar,
            beta=2.5,
            recon_loss_mode=RECON_LOSS_MODE_OFFDIAG_CHANNELMEAN,
        )
        assert_close(recon_loss.item(), offdiag, name=f"vae_loss_recon_C{channels}")
        assert_close(kld_loss.item(), 0.0, name=f"vae_loss_kld_C{channels}")
        assert_close(total.item(), offdiag, name=f"vae_loss_total_C{channels}")

    assert_close(current_losses[1] / current_losses[0], 2.0, name="current_scales_C2_over_C1")
    assert_close(current_losses[2] / current_losses[0], 3.0, name="current_scales_C3_over_C1")
    assert_close(offdiag_losses[1] / offdiag_losses[0], 1.0, name="offdiag_not_linear_C2_over_C1")
    assert_close(offdiag_losses[2] / offdiag_losses[0], 1.0, name="offdiag_not_linear_C3_over_C1")

    for activation in ("tanh", "linear", "none"):
        model = ConvolutionalVAE(
            input_channels=1,
            latent_dim=8,
            image_size=17,
            final_activation=activation,
            intermediate_fc_dim_config="0",
            num_conv_layers_encoder=3,
            decoder_type="convtranspose",
        )
        last = model.decoder_conv[-1].__class__.__name__
        if activation == "tanh" and last != "Tanh":
            raise AssertionError(f"Expected tanh model to end with Tanh, got {last}")
        if activation in {"linear", "none"} and last == "Tanh":
            raise AssertionError(f"Expected {activation} model not to append Tanh")

    print("Synthetic VAE reconstruction-loss smoke tests passed.")
    print(f"current_losses={current_losses}")
    print(f"offdiag_channelmean_losses={offdiag_losses}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
