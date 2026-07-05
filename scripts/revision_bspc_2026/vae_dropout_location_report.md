# VAE Dropout Location Report

In the locked configuration `channels=[1,0,2]` (`dropout_rate_vae=0.15`), the dropout is applied extensively across both convolutional and fully connected layers, acting as a strong regularizer.

## Dropout Application by Component

- **Encoder Convolutional Layers:**
  - **Type:** `Dropout2d` (Spatial Dropout)
  - **Location:** Applied after the `GroupNorm` and `GELU` activation in every convolutional block.
  - **Effect:** Randomly zeroes out entire channels/feature maps during training, strongly preventing spatial co-adaptation and reducing overfitting to scanner-specific spatial artifacts.

- **Encoder Intermediate Fully Connected (FC) Layer:**
  - **Type:** `Dropout` (Standard 1D Dropout)
  - **Location:** Applied after `BatchNorm1d` and `GELU`.
  - **Effect:** Regularizes the dense bottleneck prior to the $\mu$ and $\log\sigma^2$ projection.

- **Encoder $\mu$ and $\log\sigma^2$ Projections:**
  - **Location:** No dropout is applied directly to the `fc_mu` or `fc_logvar` outputs. This ensures the reparameterization trick operates on stable sufficient statistics.

- **Decoder Intermediate Fully Connected (FC) Layer:**
  - **Type:** `Dropout` (Standard 1D Dropout)
  - **Location:** Applied after `BatchNorm1d` and `GELU` on the initial projection from the latent space $\mathbf{z}$.
  - **Effect:** Prevents the decoder from over-relying on specific latent dimensions to reconstruct the output.

- **Decoder Convolutional Layers (ConvTranspose2d):**
  - **Type:** `Dropout2d` (Spatial Dropout)
  - **Location:** Applied after the `GroupNorm` and `GELU` activation in the first three deconvolutional blocks.
  - **Exception:** The final output layer (producing the 3x131x131 tensor) does not have dropout applied before its `Tanh` activation.

## Summary Conclusion
Dropout is universally applied in all intermediate representations (both 2D spatial maps and 1D dense vectors) prior to the latent bottleneck and during reconstruction. Thus, any adjustment to the `dropout_rate_vae` parameter will symmetrically tighten or loosen the regularization capacity across almost the entirety of the network.
