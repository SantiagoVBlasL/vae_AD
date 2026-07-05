# VAE Capacity Summary

This model utilizes a dense "quarter" bottleneck design (`intermediate_fc_dim_vae=quarter`), leading to a parameter-heavy structure driven by the transition between spatial and flattened dimensions.

## Total Trainable Parameters
**Total Parameters:** ~35.43 Million (35,429,507)

## Parameters by Block

- **Encoder Conv:** 140,800
- **Encoder FC Intermediate:** 16,783,360
- **FC $\mu$ Projection:** 524,544
- **FC $\log\sigma^2$ Projection:** 524,544
- **Decoder FC Intermediate:** 530,432
- **Decoder FC to Conv:** 16,785,408
- **Decoder Conv:** 140,419

## Block-wise Capacity Analysis
The vast majority of the network's capacity (>94%) resides in the dense `Linear` layers transitioning between the final flattened convolutional features (size 8192) and the intermediate bottleneck (size 2048):
1. **`Encoder FC Intermediate`:** ~16.78M
2. **`Decoder FC to Conv`:** ~16.78M

Conversely, the purely convolutional spatial layers account for less than 1% of the total parameter count (~280K parameters total). This implies the network is heavily parameterized at the global abstract level but highly constrained at the local spatial feature extraction level.
