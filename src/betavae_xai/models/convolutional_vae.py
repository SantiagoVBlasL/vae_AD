"""
models/convolutional_vae.py

CNN-based β-VAE used in:
"Explainable Latent Representation Learning for Alzheimer’s Disease:
 A β-VAE and Saliency Map Framework"
"""

from typing import Tuple, Union, List
import torch
import torch.nn as nn

DROPOUT_SCOPE_CHOICES = (
    "legacy_all",
    "encoder_only",
    "no_decoder_dropout",
    "encoder_fc_only",
    "encoder_conv_only",
    "none",
)

BLOCK_ORDER_CHOICES = (
    "legacy_act_norm",
    "norm_act",
)

__all__ = ["ConvolutionalVAE", "DROPOUT_SCOPE_CHOICES", "BLOCK_ORDER_CHOICES"]


class ConvolutionalVAE(nn.Module):
    """CNN-based Variational Autoencoder con GroupNorm."""

    def __init__(
        self,
        input_channels: int = 6,
        latent_dim: int = 128,
        image_size: int = 131,
        final_activation: str = "tanh",
        intermediate_fc_dim_config: Union[int, str] = "0",
        dropout_rate: float = 0.2,
        use_layernorm_fc: bool = False,
        num_conv_layers_encoder: int = 4,
        decoder_type: str = "convtranspose",
        num_groups: int = 16,
        encoder_norm_mode: str = "groupnorm",
        dropout_scope: str = "legacy_all",
        block_order: str = "legacy_act_norm",
    ) -> None:
        super().__init__()

        if num_conv_layers_encoder not in {3, 4}:
            raise ValueError("num_conv_layers_encoder must be 3 or 4.")
        if decoder_type not in {"upsample_conv", "convtranspose"}:
            raise ValueError("decoder_type must be 'upsample_conv' or 'convtranspose'.")

        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.image_size = image_size
        final_activation_norm = "linear" if final_activation is None else str(final_activation).lower()
        if final_activation_norm not in {"sigmoid", "tanh", "linear", "none", "identity"}:
            raise ValueError("final_activation must be one of: sigmoid, tanh, linear, none.")
        encoder_norm_mode = str(encoder_norm_mode or "groupnorm").lower()
        if encoder_norm_mode not in {"groupnorm", "layernorm", "none", "identity"}:
            raise ValueError("encoder_norm_mode must be one of: groupnorm, layernorm, none.")
        dropout_scope = str(dropout_scope or "legacy_all").lower()
        if dropout_scope not in DROPOUT_SCOPE_CHOICES:
            raise ValueError(
                "dropout_scope must be one of: " + ", ".join(DROPOUT_SCOPE_CHOICES)
            )
        block_order = str(block_order or "legacy_act_norm").lower()
        if block_order not in BLOCK_ORDER_CHOICES:
            raise ValueError(
                "block_order must be one of: " + ", ".join(BLOCK_ORDER_CHOICES)
            )

        self.final_activation_name = final_activation_norm
        self.dropout_rate = float(dropout_rate)
        self.dropout_scope = dropout_scope
        self.block_order = block_order
        self.use_layernorm_fc = use_layernorm_fc
        self.num_conv_layers_encoder = num_conv_layers_encoder
        self.decoder_type = decoder_type
        self.num_groups = num_groups
        self.encoder_norm_mode = encoder_norm_mode

        # ------------------------------
        # Encoder (conv → optional FC)
        # ------------------------------
        encoder_layers: List[nn.Module] = []
        curr_ch = input_channels
        base_conv_ch = [
            max(16, input_channels * 2),
            max(32, input_channels * 4),
            max(64, input_channels * 8),
            max(128, input_channels * 16),
        ]
        conv_ch_enc = [min(c, 256) for c in base_conv_ch][: num_conv_layers_encoder]
        kernels = [7, 5, 5, 3][: num_conv_layers_encoder]
        paddings = [1, 1, 1, 1][: num_conv_layers_encoder]
        strides = [2, 2, 2, 2][: num_conv_layers_encoder]

        spatial_dims = [image_size]
        dim = image_size
        for k, p, s, ch_out in zip(kernels, paddings, strides, conv_ch_enc):
            next_dim = ((dim + 2 * p - k) // s) + 1
            encoder_layers.append(
                nn.Conv2d(curr_ch, ch_out, kernel_size=k, stride=s, padding=p)
            )
            norm_layer = self._make_encoder_conv_norm(ch_out)
            if self.block_order == "legacy_act_norm":
                encoder_layers.append(nn.GELU())
                if norm_layer is not None:
                    encoder_layers.append(norm_layer)
            else:
                if norm_layer is not None:
                    encoder_layers.append(norm_layer)
                encoder_layers.append(nn.GELU())
            encoder_layers.append(self._make_dropout("encoder_conv", spatial=True))
            curr_ch = ch_out
            dim = next_dim
            spatial_dims.append(dim)
        self.encoder_conv = nn.Sequential(*encoder_layers)

        self.final_conv_ch = curr_ch
        self.final_spatial_dim = dim
        flat_size = curr_ch * dim * dim

        # FC encoder
        self.intermediate_fc_dim = self._resolve_intermediate_fc(
            intermediate_fc_dim_config, flat_size
        )
        if self.intermediate_fc_dim:
            fc_layers = [nn.Linear(flat_size, self.intermediate_fc_dim)]
            if self.block_order == "legacy_act_norm":
                if use_layernorm_fc:
                    fc_layers.append(nn.LayerNorm(self.intermediate_fc_dim))
                fc_layers += [
                    nn.GELU(),
                    nn.BatchNorm1d(self.intermediate_fc_dim),
                    self._make_dropout("encoder_fc", spatial=False),
                ]
            else:
                fc_layers += [
                    self._make_fc_norm(self.intermediate_fc_dim),
                    nn.GELU(),
                    self._make_dropout("encoder_fc", spatial=False),
                ]
            self.encoder_fc_intermediate = nn.Sequential(*fc_layers)
            mu_logvar_in = self.intermediate_fc_dim
        else:
            self.encoder_fc_intermediate = nn.Identity()
            mu_logvar_in = flat_size

        self.fc_mu = nn.Linear(mu_logvar_in, latent_dim)
        self.fc_logvar = nn.Linear(mu_logvar_in, latent_dim)

        # ------------------------------
        # Decoder
        # ------------------------------
        if self.intermediate_fc_dim:
            dec_fc_layers = [nn.Linear(latent_dim, self.intermediate_fc_dim)]
            if self.block_order == "legacy_act_norm":
                if use_layernorm_fc:
                    dec_fc_layers.append(nn.LayerNorm(self.intermediate_fc_dim))
                dec_fc_layers += [
                    nn.GELU(),
                    nn.BatchNorm1d(self.intermediate_fc_dim),
                    self._make_dropout("decoder_fc", spatial=False),
                ]
            else:
                dec_fc_layers += [
                    self._make_fc_norm(self.intermediate_fc_dim),
                    nn.GELU(),
                    self._make_dropout("decoder_fc", spatial=False),
                ]
            self.decoder_fc_intermediate = nn.Sequential(*dec_fc_layers)
            dec_fc_out = self.intermediate_fc_dim
        else:
            self.decoder_fc_intermediate = nn.Identity()
            dec_fc_out = latent_dim

        self.decoder_fc_to_conv = nn.Linear(dec_fc_out, flat_size)
        # Decoder conv layers
        decoder_layers: List[nn.Module] = []
        if decoder_type == "convtranspose":
            curr_ch_dec = self.final_conv_ch
            target_conv_t_channels = conv_ch_enc[-2 :: -1] + [input_channels]
            decoder_kernels = kernels[::-1]
            decoder_paddings = paddings[::-1]
            decoder_strides = strides[::-1]

            output_paddings: List[int] = []
            tmp_dim = self.final_spatial_dim
            for i in range(num_conv_layers_encoder):
                k, s, p = decoder_kernels[i], decoder_strides[i], decoder_paddings[i]
                target_dim = spatial_dims[num_conv_layers_encoder - 1 - i]
                op = target_dim - ((tmp_dim - 1) * s - 2 * p + k)
                op = max(0, min(s - 1, op))
                output_paddings.append(op)
                tmp_dim = (tmp_dim - 1) * s - 2 * p + k + op

            for i, ch_out in enumerate(target_conv_t_channels):
                decoder_layers.append(
                    nn.ConvTranspose2d(
                        curr_ch_dec,
                        ch_out,
                        kernel_size=decoder_kernels[i],
                        stride=decoder_strides[i],
                        padding=decoder_paddings[i],
                        output_padding=output_paddings[i],
                    )
                )
                if i < len(target_conv_t_channels) - 1:
                    if self.block_order == "legacy_act_norm":
                        decoder_layers += [
                            nn.GELU(),
                            nn.GroupNorm(self.num_groups, ch_out),
                            self._make_dropout("decoder_conv", spatial=True),
                        ]
                    else:
                        decoder_layers += [
                            nn.GroupNorm(self.num_groups, ch_out),
                            nn.GELU(),
                            self._make_dropout("decoder_conv", spatial=True),
                        ]
                else:
                    decoder_layers.append(nn.Identity())
                curr_ch_dec = ch_out

        elif decoder_type == "upsample_conv":
            # Mirror del encoder: subimos a sizes exactos usando spatial_dims
            curr_ch_dec = self.final_conv_ch
            target_channels = conv_ch_enc[-2 :: -1] + [input_channels]   # igual que convtranspose
            decoder_kernels = kernels[::-1]                               # ej [3,5,5,7] si encoder=[7,5,5,3]

            # i=0: upsample a spatial_dims[L-1] (dim anterior a la última conv)
            # ...
            # i=L-1: upsample a spatial_dims[0] (= image_size)
            for i, ch_out in enumerate(target_channels):
                target_dim = spatial_dims[num_conv_layers_encoder - 1 - i]

                k = decoder_kernels[i]
                # padding para preservar tamaño con stride=1
                pad = k // 2

                decoder_layers += [
                    nn.Upsample(size=(target_dim, target_dim), mode="bilinear", align_corners=False),
                    nn.Conv2d(curr_ch_dec, ch_out, kernel_size=k, stride=1, padding=pad),
                ]
                if i < len(target_channels) - 1:
                    if self.block_order == "legacy_act_norm":
                        decoder_layers += [
                            nn.GELU(),
                            nn.GroupNorm(self.num_groups, ch_out),
                            self._make_dropout("decoder_conv", spatial=True),
                        ]
                    else:
                        decoder_layers += [
                            nn.GroupNorm(self.num_groups, ch_out),
                            nn.GELU(),
                            self._make_dropout("decoder_conv", spatial=True),
                        ]
                else:
                    decoder_layers.append(nn.Identity())
                curr_ch_dec = ch_out

        else:
            raise ValueError(f"decoder_type desconocido: {decoder_type}")

        if final_activation_norm == "sigmoid":
            decoder_layers.append(nn.Sigmoid())
        elif final_activation_norm == "tanh":
            decoder_layers.append(nn.Tanh())
        elif final_activation_norm in {"linear", "none", "identity"}:
            pass

        self.decoder_conv = nn.Sequential(*decoder_layers)

    def _make_encoder_conv_norm(self, channels: int) -> Union[nn.Module, None]:
        if self.encoder_norm_mode == "groupnorm":
            return nn.GroupNorm(self.num_groups, channels)
        if self.encoder_norm_mode == "layernorm":
            # GroupNorm(1, C) is a LayerNorm-style option for NCHW conv features.
            return nn.GroupNorm(1, channels)
        return None

    def _make_fc_norm(self, features: int) -> nn.Module:
        if self.use_layernorm_fc:
            return nn.LayerNorm(features)
        return nn.BatchNorm1d(features)

    def _dropout_enabled(self, location: str) -> bool:
        if self.dropout_rate <= 0.0:
            return False
        if self.dropout_scope == "legacy_all":
            return True
        if self.dropout_scope == "none":
            return False
        if self.dropout_scope in {"encoder_only", "no_decoder_dropout"}:
            return location.startswith("encoder_")
        if self.dropout_scope == "encoder_fc_only":
            return location == "encoder_fc"
        if self.dropout_scope == "encoder_conv_only":
            return location == "encoder_conv"
        return False

    def _make_dropout(self, location: str, spatial: bool) -> nn.Module:
        if not self._dropout_enabled(location):
            return nn.Identity()
        dropout_cls = nn.Dropout2d if spatial else nn.Dropout
        return dropout_cls(p=self.dropout_rate)

    def _resolve_intermediate_fc(self, cfg: Union[int, str], flat_size: int) -> int:
        if cfg == "0" or cfg == 0:
            return 0
        if isinstance(cfg, str):
            cfg = cfg.lower()
            if cfg == "half":
                return flat_size // 2
            if cfg == "quarter":
                return flat_size // 4
            try:
                return int(cfg)
            except ValueError:
                return 0
        return int(cfg)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_conv(x)
        h = h.view(h.size(0), -1)
        h = self.encoder_fc_intermediate(h)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc_intermediate(z)
        h = self.decoder_fc_to_conv(h)
        h = h.view(
            h.size(0), self.final_conv_ch, self.final_spatial_dim, self.final_spatial_dim
        )
        return self.decoder_conv(h)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        if recon_x.shape != x.shape:
            recon_x = nn.functional.interpolate(
                recon_x,
                size=(x.shape[2], x.shape[3]),
                mode="bilinear",
                align_corners=False,
            )
        return recon_x, mu, logvar, z
