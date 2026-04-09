import torch
import torch.nn as nn
import torch.nn.functional as F

from .attn import AnomalyAttention, AttentionLayer
from .embed import DataEmbedding


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn, prior, sigma, mixture_weights = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )

        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn, prior, sigma, mixture_weights


class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x: [B, L, D]
        series_list = []
        prior_list = []
        sigma_list = []
        mixture_weight_list = []

        for attn_layer in self.attn_layers:
            x, series, prior, sigma, mixture_weights = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)
            mixture_weight_list.append(mixture_weights)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list, mixture_weight_list


class AnomalyTransformer(nn.Module):
    def __init__(
        self,
        win_size,
        enc_in,
        c_out,
        d_model=512,
        n_heads=8,
        e_layers=3,
        d_ff=512,
        dropout=0.0,
        activation='gelu',
        output_attention=True,
        prior_type="gaussian",
        n_mixtures=1,
        normalize_prior=True,
        sigma_activation="softplus",
        sigma_min=1e-4,
    ):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention
        self.prior_type = prior_type
        self.n_mixtures = n_mixtures

        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        AnomalyAttention(
                            win_size=win_size,
                            mask_flag=False,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                            prior_type=prior_type,
                            n_mixtures=n_mixtures,
                            normalize_prior=normalize_prior,
                            sigma_activation=sigma_activation,
                            sigma_min=sigma_min,
                        ),
                        d_model=d_model,
                        n_heads=n_heads,
                        prior_type=prior_type,
                        n_mixtures=n_mixtures,
                    ),
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model)
        )

        self.projection = nn.Linear(d_model, c_out, bias=True)

    def forward(self, x):
        """
        Returns:
            enc_out:         [B, L, c_out]
            series:          list of [B, H, L, L]
            prior:           list of [B, H, L, L]
            sigmas:          list of [B, H, L, K] (or K=1 for gaussian)
            mixture_weights: list of [B, H, L, K] (or all-ones if gaussian)
        """
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas, mixture_weights = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas, mixture_weights
        else:
            return enc_out  # [B, L, D]