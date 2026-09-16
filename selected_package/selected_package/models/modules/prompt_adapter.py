#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prompt_adapter_ablation.py

Drop-in PromptAdapter module with ablation switches:
- route_mode: none/global/hole_only/boundary_only/hole_boundary
- adapter_variant: none/film_only/attn_only/film_attn
- prompt_scale: controlled at forward time

Integration:
    from prompt_adapter_ablation import PromptAdapter, build_prompt_route

Mask convention:
    mask: [B,1,H,W], 1 = known region, 0 = hole/missing region.
"""

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_prompt_route(
    mask: torch.Tensor,
    target_size: Tuple[int, int],
    route_mode: str = "hole_boundary",
    boundary_weight: float = 0.5,
    temperature: float = 1.0,
    kernel_size: int = 5,
) -> torch.Tensor:
    """
    Build spatial routing weights for prompt residual.

    Args:
        mask: [B,1,H,W], 1=known region, 0=hole/missing region.
        target_size: target feature map size, (h,w).
        route_mode:
            - "none" or "global": all-one route.
            - "hole_only": inject only in missing region.
            - "boundary_only": inject only near mask boundary.
            - "hole_boundary": inject in missing region and boundary band.
        boundary_weight: weight for boundary band in hole_boundary mode.
        temperature: optional sharpening value. Use 1.0 to disable sharpening.
        kernel_size: boundary dilation kernel size.

    Returns:
        route: [B,1,h,w]
    """
    if mask is None:
        raise ValueError("mask cannot be None when building prompt route.")

    if mask.ndim != 4 or mask.shape[1] != 1:
        raise ValueError(f"mask should have shape [B,1,H,W], got {tuple(mask.shape)}")

    b = mask.shape[0]
    h, w = target_size

    if route_mode in ["none", "global"]:
        return torch.ones(b, 1, h, w, device=mask.device, dtype=mask.dtype)

    hole = 1.0 - mask.float()
    hole_down = F.interpolate(hole, size=target_size, mode="area").clamp(0.0, 1.0)

    pad = kernel_size // 2
    dilated_hole = F.max_pool2d(
        hole,
        kernel_size=kernel_size,
        stride=1,
        padding=pad,
    )
    boundary = (dilated_hole - hole).clamp(0.0, 1.0)
    boundary_down = F.interpolate(boundary, size=target_size, mode="area").clamp(0.0, 1.0)

    if route_mode == "hole_only":
        route = hole_down
    elif route_mode == "boundary_only":
        route = boundary_down
    elif route_mode == "hole_boundary":
        route = (hole_down + float(boundary_weight) * boundary_down).clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unknown route_mode: {route_mode}")

    if temperature is not None and float(temperature) > 1.0:
        route = torch.sigmoid((route - 0.5) * float(temperature))

    return route


class PromptAdapter(nn.Module):
    """
    Multi-scale prompt adapter with FiLM and cross-attention residuals.

    Expected inputs:
        x_img: [B,C,H,W]
        x_text: [B,T,context_dim] or [1,T,context_dim]
        mask: [B,1,H0,W0], 1=known region, 0=hole/missing region

    Returns:
        residual: [B,C,H,W]
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int = 768,
        heads: int = 8,
        route_boundary_weight: float = 0.5,
        route_temperature: float = 1.0,
        route_mode: str = "hole_boundary",
        route_kernel_size: int = 5,
        adapter_variant: str = "film_attn",
    ):
        super().__init__()

        if query_dim <= 0:
            raise ValueError("query_dim must be positive.")

        if query_dim % heads != 0:
            heads = 4 if query_dim % 4 == 0 else 1

        self.query_dim = query_dim
        self.context_dim = context_dim
        self.heads = heads

        self.route_boundary_weight = route_boundary_weight
        self.route_temperature = route_temperature
        self.route_mode = route_mode
        self.route_kernel_size = route_kernel_size
        self.adapter_variant = adapter_variant

        self.norm = self._make_group_norm(query_dim)
        self.text_proj = nn.Linear(context_dim, query_dim)

        self.to_gamma_beta = nn.Sequential(
            nn.Linear(query_dim, query_dim * 2),
            nn.SiLU(),
            nn.Linear(query_dim * 2, query_dim * 2),
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=query_dim,
            num_heads=heads,
            batch_first=True,
        )

        self.out_proj = nn.Conv2d(query_dim, query_dim, kernel_size=1)

        self.alpha_film = nn.Parameter(torch.tensor(1.0))
        self.alpha_attn = nn.Parameter(torch.tensor(1.0))

        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    @staticmethod
    def _make_group_norm(num_channels: int) -> nn.GroupNorm:
        for groups in (32, 16, 8, 4, 2):
            if num_channels >= groups and num_channels % groups == 0:
                return nn.GroupNorm(groups, num_channels)
        return nn.GroupNorm(1, num_channels)

    def forward(
        self,
        x_img: torch.Tensor,
        x_text: Optional[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        prompt_scale: float = 1.0,
    ) -> torch.Tensor:
        if self.adapter_variant == "none" or prompt_scale == 0 or x_text is None:
            return torch.zeros_like(x_img)

        if x_img.ndim != 4:
            raise ValueError(f"x_img should have shape [B,C,H,W], got {tuple(x_img.shape)}")
        if x_text.ndim != 3:
            raise ValueError(f"x_text should have shape [B,T,C], got {tuple(x_text.shape)}")

        b, c, h, w = x_img.shape

        if x_text.shape[0] == 1 and b > 1:
            x_text = x_text.expand(b, -1, -1)

        if x_text.shape[0] != b:
            raise ValueError(f"Batch size mismatch: x_img B={b}, x_text B={x_text.shape[0]}")

        x_norm = self.norm(x_img)
        text_tokens = self.text_proj(x_text)

        text_pooled = text_tokens.mean(dim=1)
        gamma, beta = self.to_gamma_beta(text_pooled).chunk(2, dim=-1)
        gamma = gamma.view(b, c, 1, 1)
        beta = beta.view(b, c, 1, 1)
        film_residual = x_norm * gamma + beta

        img_tokens = x_norm.flatten(2).transpose(1, 2)
        attn_tokens, _ = self.cross_attn(
            query=img_tokens,
            key=text_tokens,
            value=text_tokens,
            need_weights=False,
        )
        attn_residual = attn_tokens.transpose(1, 2).view(b, c, h, w)

        if self.adapter_variant == "film_only":
            residual = self.alpha_film * film_residual
        elif self.adapter_variant == "attn_only":
            residual = self.alpha_attn * attn_residual
        elif self.adapter_variant == "film_attn":
            residual = self.alpha_film * film_residual + self.alpha_attn * attn_residual
        else:
            raise ValueError(f"Unknown adapter_variant: {self.adapter_variant}")

        residual = self.out_proj(residual)

        if mask is not None:
            route = build_prompt_route(
                mask=mask,
                target_size=(h, w),
                route_mode=self.route_mode,
                boundary_weight=self.route_boundary_weight,
                temperature=self.route_temperature,
                kernel_size=self.route_kernel_size,
            )
            residual = residual * route

        return residual * float(prompt_scale)
