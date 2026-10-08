"""Encoder building blocks: RMSNorm, rotary MHSA, SwiGLU, SE convolution, Macaron block."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


class Rotary(nn.Module):
    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self._len = 0
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None

    def forward(self, T: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cos is None or T > self._len or self._cos.device != device:
            n = max(T, 2 * self._len, 1024)
            t = torch.arange(n, device=device, dtype=torch.float32)
            freqs = torch.outer(t, self.inv_freq.to(device))
            emb = torch.cat([freqs, freqs], dim=-1)
            self._cos, self._sin, self._len = emb.cos(), emb.sin(), n
        return self._cos[:T].to(dtype), self._sin[:T].to(dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return torch.cat([-b, a], dim=-1)


class RotaryMHSA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.h = n_heads
        self.dh = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.rotary = Rotary(self.dh)
        self.p = dropout

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor | None) -> torch.Tensor:
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.k(x).view(B, T, self.h, self.dh).transpose(1, 2)
        v = self.v(x).view(B, T, self.h, self.dh).transpose(1, 2)
        cos, sin = self.rotary(T, x.device, q.dtype)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        attn_mask = key_mask[:, None, None, :] if key_mask is not None else None
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.p if self.training else 0.0)
        return self.o(out.transpose(1, 2).reshape(B, T, D))


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, multiplier: float, dropout: float):
        super().__init__()
        inner = int(d_model * multiplier)
        self.norm = RMSNorm(d_model)
        self.gate = nn.Linear(d_model, inner, bias=False)
        self.up = nn.Linear(d_model, inner, bias=False)
        self.down = nn.Linear(inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.down(self.drop(F.silu(self.gate(h)) * self.up(h)))
        return self.drop(h)


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, ratio: int):
        super().__init__()
        mid = max(1, channels // ratio)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        # x: (B, T, C). Mean over valid frames only.
        if mask is None:
            s = x.mean(dim=1, keepdim=True)
        else:
            m = mask.unsqueeze(-1).to(x.dtype)
            s = (x * m).sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True).clamp(min=1.0)
        return x * torch.sigmoid(self.fc2(F.silu(self.fc1(s))))


class ConvModule(nn.Module):
    def __init__(self, d_model: int, kernel: int, expansion: int, se_ratio: int, dropout: float, norm: str = "batch"):
        super().__init__()
        inner = d_model * expansion
        self.norm = RMSNorm(d_model)
        self.pw1 = nn.Linear(d_model, 2 * inner)
        self.dw = nn.Conv1d(inner, inner, kernel, padding=kernel // 2, groups=inner)
        if norm == "batch":
            self.conv_norm: nn.Module = nn.BatchNorm1d(inner)
        elif norm == "layer":
            self.conv_norm = nn.GroupNorm(1, inner)
        else:
            raise ValueError(f"conv_norm must be batch|layer, got {norm!r}")
        self.se = SqueezeExcite(inner, se_ratio)
        self.pw2 = nn.Linear(inner, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        h = F.glu(self.pw1(self.norm(x)), dim=-1)
        if mask is not None:
            h = h.masked_fill(~mask.unsqueeze(-1), 0.0)
        h = self.dw(h.transpose(1, 2))
        h = F.silu(self.conv_norm(h)).transpose(1, 2)
        h = self.se(h, mask)
        return self.drop(self.pw2(h))


class MacaronBlock(nn.Module):
    """½FFN → MHSA → Conv → ½FFN → RMSNorm, each with a residual connection."""

    def __init__(self, d_model: int, n_heads: int, ff_multiplier: float, conv_kernel: int, conv_expansion: int,
                 se_ratio: int, dropout: float, drop_rate: float, conv_norm: str = "batch"):
        super().__init__()
        self.ff1 = SwiGLUFFN(d_model, ff_multiplier, dropout)
        self.attn_norm = RMSNorm(d_model)
        self.attn = RotaryMHSA(d_model, n_heads, dropout)
        self.attn_drop = nn.Dropout(dropout)
        self.conv = ConvModule(d_model, conv_kernel, conv_expansion, se_ratio, dropout, conv_norm)
        self.ff2 = SwiGLUFFN(d_model, ff_multiplier, dropout)
        self.final_norm = RMSNorm(d_model)
        self.drop_rate = drop_rate

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if self.training and self.drop_rate > 0 and torch.rand(()) < self.drop_rate:
            return x
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn_drop(self.attn(self.attn_norm(x), mask))
        x = x + self.conv(x, mask)
        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)


class ConvSubsampling4x(nn.Module):
    """Two 3×3 stride-2 Conv2d layers (4× in time and frequency) → Linear → RMSNorm."""

    def __init__(self, n_mels: int, d_model: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv2d(1, d_model, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(d_model, d_model, 3, stride=2, padding=1)
        f = (n_mels + 1) // 2
        f = (f + 1) // 2
        self.linear = nn.Linear(d_model * f, d_model)
        self.norm = RMSNorm(d_model)
        self.drop = nn.Dropout(dropout)

    @staticmethod
    def output_lengths(lengths: torch.Tensor) -> torch.Tensor:
        lengths = torch.div(lengths - 1, 2, rounding_mode="floor") + 1
        lengths = torch.div(lengths - 1, 2, rounding_mode="floor") + 1
        return lengths.clamp(min=1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.silu(self.conv1(x.unsqueeze(1)))
        x = F.silu(self.conv2(x))
        B, C, T, Fq = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * Fq)
        x = self.drop(self.norm(self.linear(x)))
        return x, self.output_lengths(lengths).clamp(max=T)
