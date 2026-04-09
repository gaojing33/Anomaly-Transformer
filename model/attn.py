import math
from math import sqrt

import numpy as np
import torch
import torch.nn as nn


class TriangularCausalMask:
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(
                torch.ones(mask_shape, dtype=torch.bool, device=device),
                diagonal=1
            )

    @property
    def mask(self):
        return self._mask


class AnomalyAttention(nn.Module):
    """
    Anomaly-Attention with support for:
    1) original single-Gaussian prior
    2) Gaussian Mixture Prior

    Returns:
        V:              attention output
        series:         learned series association, shape [B, H, L, S]
        prior:          prior association, shape [B, H, L, S]
        sigma_out:      positive sigma used for prior construction
                        shape [B, H, L, K] if mixture, else [B, H, L, 1]
        mixture_weights positive mixture weights
                        shape [B, H, L, K] if mixture, else all-ones tensor
    """

    def __init__(
        self,
        win_size,
        mask_flag=True,
        scale=None,
        attention_dropout=0.0,
        output_attention=False,
        prior_type="gaussian",
        n_mixtures=1,
        normalize_prior=True,
        sigma_activation="softplus",
        sigma_min=1e-4,
    ):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        self.win_size = win_size
        self.prior_type = prior_type
        self.n_mixtures = n_mixtures
        self.normalize_prior = normalize_prior
        self.sigma_activation = sigma_activation
        self.sigma_min = sigma_min

        distance = torch.arange(win_size).float()
        distance = torch.abs(distance[:, None] - distance[None, :])  # [L, L]
        self.register_buffer("distances", distance, persistent=False)

    def _positive_sigma(self, sigma_raw):
        """
        Convert raw sigma to positive values.
        Input shape: [B, H, L, K]
        """
        if self.sigma_activation == "softplus":
            sigma = torch.nn.functional.softplus(sigma_raw) + self.sigma_min
        elif self.sigma_activation == "exp":
            sigma = torch.exp(sigma_raw).clamp_min(self.sigma_min)
        else:
            # keep a version closer to the original repo's scaling
            sigma = torch.sigmoid(sigma_raw * 5.0) + 1e-5
            sigma = torch.pow(torch.tensor(3.0, device=sigma.device), sigma) - 1.0
            sigma = sigma.clamp_min(self.sigma_min)
        return sigma

    def _build_gaussian_prior(self, sigma, L):
        """
        Build single-Gaussian prior.
        sigma: [B, H, L, 1]
        return prior: [B, H, L, L]
        """
        distances = self.distances[:L, :L].to(sigma.device)  # [L, L]
        distances = distances.unsqueeze(0).unsqueeze(0)      # [1, 1, L, L]

        sigma_expanded = sigma.expand(-1, -1, -1, L)         # [B, H, L, L]
        coef = 1.0 / (math.sqrt(2.0 * math.pi) * sigma_expanded)
        prior = coef * torch.exp(-(distances ** 2) / (2.0 * sigma_expanded ** 2))
        return prior

    def _build_mixture_prior(self, sigma, mixture_weights, L):
        """
        Build Gaussian mixture prior.
        sigma:           [B, H, L, K]
        mixture_weights: [B, H, L, K]
        return prior:    [B, H, L, L]
        """
        distances = self.distances[:L, :L].to(sigma.device)          # [L, L]
        distances = distances.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, L, L, 1]

        sigma_expanded = sigma.unsqueeze(-2)                         # [B, H, L, 1, K]
        weights_expanded = mixture_weights.unsqueeze(-2)             # [B, H, L, 1, K]

        coef = 1.0 / (math.sqrt(2.0 * math.pi) * sigma_expanded)
        gaussian_terms = coef * torch.exp(-(distances ** 2) / (2.0 * sigma_expanded ** 2))
        prior = (weights_expanded * gaussian_terms).sum(dim=-1)      # [B, H, L, L]
        return prior

    def _normalize_prior(self, prior):
        """
        Row-normalize prior over the last dimension.
        prior: [B, H, L, L]
        """
        denom = prior.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return prior / denom

    def forward(self, queries, keys, values, sigma, attn_mask, mixture_weights=None):
        """
        queries:         [B, L, H, E]
        keys:            [B, S, H, E]
        values:          [B, S, H, D]
        sigma:           [B, L, H, K] or [B, L, H]
        mixture_weights: [B, L, H, K] or None
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or (1.0 / sqrt(E))

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)  # [B, H, L, S]

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = scale * scores

        # series association
        series = self.dropout(torch.softmax(attn, dim=-1))       # [B, H, L, S]
        V = torch.einsum("bhls,bshd->blhd", series, values)       # [B, L, H, D]

        # prepare sigma
        if sigma.dim() == 3:
            sigma = sigma.unsqueeze(-1)  # [B, L, H] -> [B, L, H, 1]

        # [B, L, H, K] -> [B, H, L, K]
        sigma = sigma.permute(0, 2, 1, 3).contiguous()
        sigma_out = self._positive_sigma(sigma)

        # prepare mixture weights
        if mixture_weights is None:
            mixture_weights_out = torch.ones(
                B, H, L, 1,
                device=queries.device,
                dtype=queries.dtype
            )
        else:
            if mixture_weights.dim() == 3:
                mixture_weights = mixture_weights.unsqueeze(-1)
            mixture_weights_out = mixture_weights.permute(0, 2, 1, 3).contiguous()
            mixture_weights_out = torch.softmax(mixture_weights_out, dim=-1)

        # build prior
        if self.prior_type == "mixture":
            prior = self._build_mixture_prior(
                sigma=sigma_out,
                mixture_weights=mixture_weights_out,
                L=L
            )
        else:
            # for gaussian mode only use the first component
            prior = self._build_gaussian_prior(
                sigma=sigma_out[..., :1],
                L=L
            )
            mixture_weights_out = torch.ones(
                B, H, L, 1,
                device=queries.device,
                dtype=queries.dtype
            )
            sigma_out = sigma_out[..., :1]

        if self.normalize_prior:
            prior = self._normalize_prior(prior)

        if self.output_attention:
            return V.contiguous(), series, prior, sigma_out, mixture_weights_out
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(
        self,
        attention,
        d_model,
        n_heads,
        d_keys=None,
        d_values=None,
        prior_type="gaussian",
        n_mixtures=1
    ):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.n_heads = n_heads
        self.prior_type = prior_type
        self.n_mixtures = n_mixtures

        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)

        # sigma projection
        sigma_out_dim = n_heads * n_mixtures
        self.sigma_projection = nn.Linear(d_model, sigma_out_dim)

        # mixture weight projection
        if prior_type == "mixture":
            self.pi_projection = nn.Linear(d_model, sigma_out_dim)
        else:
            self.pi_projection = None

        self.out_projection = nn.Linear(d_values * n_heads, d_model)

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        sigma = self.sigma_projection(x).view(B, L, H, self.n_mixtures)

        if self.pi_projection is not None:
            mixture_weights = self.pi_projection(x).view(B, L, H, self.n_mixtures)
        else:
            mixture_weights = None

        out, series, prior, sigma_out, mixture_weights_out = self.inner_attention(
            queries,
            keys,
            values,
            sigma,
            attn_mask,
            mixture_weights=mixture_weights
        )
        out = out.view(B, L, -1)

        return (
            self.out_projection(out),
            series,
            prior,
            sigma_out,
            mixture_weights_out
        )