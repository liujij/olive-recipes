# -------------------------------------------------------------------------
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
# -------------------------------------------------------------------------
# Adapter code for Gemma-4-E2B → FLUX.2-Klein-4B text-encoder swap.
#
# Original author: Vahid, 2026-06-18 (Apache 2.0)
# Inlined here to remove the external te_swap_adapter package dependency.
#
# Contents
# --------
#   GemmaToKleinAdapterV2   – maps Gemma hidden states → 7680-d conditioning
#   LoRALinear / inject_lora – standalone cross-attention LoRA for the DiT
# -------------------------------------------------------------------------
from __future__ import annotations

import re

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEMMA_HIDDEN = 1536
CONCAT       = 4608        # 3 slots × 1536
KLEIN_TE_DIM = 7680        # joint_attention_dim
CANDIDATES   = (9, 18, 26, 30)   # Gemma-E2B hidden_states indices
N_SLOTS      = 3


# ---------------------------------------------------------------------------
# Adapter sub-modules
# ---------------------------------------------------------------------------

class TapVarNorm(nn.Module):
    """Per-tap RMSNorm → unit variance, then a small learnable per-channel
    scale (SANA-style, init 0.01).  Tames the 31-175× outlier-dim ratio
    before the taps are combined."""

    def __init__(self, dim: int = GEMMA_HIDDEN, init_scale: float = 0.01,
                 eps: float = 1e-6):
        super().__init__()
        self.eps   = eps
        self.scale = nn.Parameter(torch.full((dim,), float(init_scale)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, T, D]
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * self.scale


class LearnableTaps(nn.Module):
    """3 output slots, each a learnable softmax (convex combination) over the
    candidate layers.  Init = one-hot at 9/18/26 → reproduces the fixed
    3-layer concat at step 0; training can move the taps."""

    def __init__(self, candidates: tuple = CANDIDATES, n_slots: int = N_SLOTS):
        super().__init__()
        self.candidates = list(candidates)
        self.varnorm    = nn.ModuleList([TapVarNorm() for _ in candidates])
        logits = torch.full((n_slots, len(candidates)), -10.0)
        for j in range(n_slots):          # slot j one-hot at candidates[j]
            logits[j, j] = 10.0
        self.logits = nn.Parameter(logits)

    def forward(self, taps: list[torch.Tensor]) -> torch.Tensor:
        # taps: list[len(candidates)] of [B, T, 1536]
        normed = [vn(t) for vn, t in zip(self.varnorm, taps)]
        stack  = torch.stack(normed, dim=0)               # [C, B, T, 1536]
        w      = F.softmax(self.logits, dim=-1)            # [n_slots, C]
        slots  = torch.einsum("sc,cbtd->sbtd", w, stack)  # [n_slots, B, T, 1536]
        return torch.cat([slots[j] for j in range(slots.shape[0])], dim=-1)  # [B,T,4608]


class _RMSNorm(nn.Module):
    """Decomposed RMSNorm compatible with ONNX opset 17 (no aten::rms_norm)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class BidirectionalRefiner(nn.Module):
    """1 bidirectional self-attention block (causal mask DROPPED), QK-norm,
    key_padding_mask over right-pad, ReZero zero-init residual gate → step 0
    is a no-op.  The only cross-token mixer."""

    def __init__(self, dim: int = GEMMA_HIDDEN, heads: int = 12,
                 mlp_ratio: float = 4.0, eps: float = 1e-6):
        super().__init__()
        self.h, self.hd = heads, dim // heads
        self.norm1  = nn.LayerNorm(dim, eps=eps)
        self.qkv    = nn.Linear(dim, dim * 3)
        self.q_norm = _RMSNorm(self.hd, eps=eps)
        self.k_norm = _RMSNorm(self.hd, eps=eps)
        self.proj   = nn.Linear(dim, dim)
        self.norm2  = nn.LayerNorm(dim, eps=eps)
        inner       = int(dim * mlp_ratio)
        self.mlp    = nn.Sequential(
            nn.Linear(dim, inner), nn.GELU(), nn.Linear(inner, dim)
        )
        self.gate_attn = nn.Parameter(torch.zeros(1))   # ReZero: 0 at init
        self.gate_mlp  = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = self.q_norm(q.view(B, T, self.h, self.hd)).transpose(1, 2)  # [B,h,T,hd]
        k = self.k_norm(k.view(B, T, self.h, self.hd)).transpose(1, 2)
        v = v.view(B, T, self.h, self.hd).transpose(1, 2)
        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = torch.zeros(B, 1, 1, T, device=x.device, dtype=q.dtype)
            attn_mask = attn_mask.masked_fill(
                ~key_padding_mask[:, None, None, :], float("-inf")
            )
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        o = o.transpose(1, 2).reshape(B, T, D)
        x = x + self.gate_attn * self.proj(o)
        x = x + self.gate_mlp  * self.mlp(self.norm2(x))
        return x


class GemmaToKleinAdapterV2(nn.Module):
    """Maps Gemma-4-E2B hidden-state taps → FLUX.2-Klein 7680-d text conditioning.

    Chain: per-tap var-norm → learnable softmax taps (3 slots over 4 candidates)
    → concat [B,T,4608] → Linear 4608→1536 → N bidirectional refiners
    → LayerNorm → Linear 1536→7680.
    """

    def __init__(self, candidates: tuple = CANDIDATES, refiner_depth: int = 1,
                 heads: int = 12, out_dim: int = KLEIN_TE_DIM,
                 varnorm: bool = True, learnable_taps: bool = True):
        super().__init__()
        self.candidates = list(candidates)
        self.taps       = LearnableTaps(candidates)
        if not varnorm:
            self.taps.varnorm = nn.ModuleList(
                [nn.Identity() for _ in candidates]
            )
        if not learnable_taps:
            self.taps.logits.requires_grad_(False)
        self.project  = nn.Linear(CONCAT, GEMMA_HIDDEN)
        self.refiner  = nn.ModuleList([
            BidirectionalRefiner(GEMMA_HIDDEN, heads)
            for _ in range(refiner_depth)
        ])
        self.out_norm = nn.LayerNorm(GEMMA_HIDDEN, eps=1e-6)
        self.head     = nn.Linear(GEMMA_HIDDEN, out_dim)

    def trunk(self, taps: list[torch.Tensor]) -> torch.Tensor:
        """Warm-startable refiner-free path."""
        x = self.taps(taps)
        return self.head(self.out_norm(self.project(x)))

    def forward(self, taps: list[torch.Tensor],
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x   = self.project(self.taps(taps))                      # [B, T, 1536]
        kpm = attention_mask.bool() if attention_mask is not None else None
        for blk in self.refiner:
            x = blk(x, key_padding_mask=kpm)
        return self.head(self.out_norm(x))                        # [B, T, 7680]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# LoRA for the FLUX.2-Klein DiT cross-attention projections
# ---------------------------------------------------------------------------

_DEFAULT_LORA_TARGET = r"\.attn\.(to_k|to_v|add_k_proj|add_v_proj)$"


class LoRALinear(nn.Module):
    """Frozen base Linear + trainable low-rank update  (W_out = W_base + B·A·scale)."""

    def __init__(self, base: nn.Linear, rank: int = 64, alpha: int = 64):
        super().__init__()
        self.base  = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.zeros(rank, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + (x @ self.A.t() @ self.B.t()) * self.scale


def inject_lora(transformer: nn.Module, target_re: str = _DEFAULT_LORA_TARGET,
                rank: int = 64, alpha: int = 64) -> tuple[list, int]:
    """Wrap matching Linear modules of *transformer* in LoRALinear, in place.

    Returns ``(param_list, n_wrapped)``.  ``param_list`` is ``[A0, B0, A1, B1, …]``
    in module-discovery order, which matches the checkpoint's ``'lora'`` tensor
    list exactly (they zip 1-to-1).
    """
    pat     = re.compile(target_re)
    wrapped = 0
    params: list = []
    for name, mod in list(transformer.named_modules()):
        for child_name, child in list(mod.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Linear) and pat.search(full):
                lora = LoRALinear(child, rank, alpha).to(
                    child.weight.device, child.weight.dtype
                )
                setattr(mod, child_name, lora)
                params += [lora.A, lora.B]
                wrapped += 1
    return params, wrapped
