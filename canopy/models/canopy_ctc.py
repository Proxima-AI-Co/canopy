from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from canopy.frontend.features import LogMelFrontend
from canopy.models.config import ModelConfig
from canopy.models.modules import ConvSubsampling4x, MacaronBlock


def lengths_to_mask(lengths: torch.Tensor, T: int) -> torch.Tensor:
    return torch.arange(T, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)


class CanopyCTC(nn.Module):
    """Conformer-style CTC encoder with intermediate CTC, optional language
    conditioning, self-conditioning and a language-ID head."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.vocab_size <= 1:
            raise ValueError("cfg.vocab_size must be set from the tokenizer")
        for k in cfg.aux_ctc_layers:
            if not 1 <= k < cfg.n_layers:
                raise ValueError(f"aux_ctc_layers entries must be in [1, n_layers), got {k}")
        self.cfg = cfg
        d = cfg.d_model
        self.frontend = LogMelFrontend(n_mels=cfg.n_mels, normalize=cfg.feature_normalize)
        self.subsampling = ConvSubsampling4x(cfg.n_mels, d, cfg.dropout)
        self.lang_embed = nn.Embedding(cfg.num_languages + 1, d) if cfg.language_embedding else None
        if self.lang_embed is not None:
            nn.init.zeros_(self.lang_embed.weight)
        n = cfg.n_layers
        self.blocks = nn.ModuleList([
            MacaronBlock(d, cfg.n_heads, cfg.ff_multiplier, cfg.conv_kernel, cfg.conv_expansion, cfg.se_ratio,
                         cfg.dropout, cfg.stochastic_depth * i / max(n - 1, 1), cfg.conv_norm)
            for i in range(n)
        ])
        self.ctc_head = nn.Linear(d, cfg.vocab_size)
        self.aux_heads = nn.ModuleDict({str(k): nn.Linear(d, cfg.vocab_size) for k in cfg.aux_ctc_layers})
        self.self_cond = (nn.ModuleDict({str(k): nn.Linear(cfg.vocab_size, d, bias=False) for k in cfg.aux_ctc_layers})
                          if cfg.self_conditioning else None)
        self.lid_head = nn.Linear(d, cfg.num_languages + 1) if cfg.lid_head else None
        with torch.no_grad():
            for head in [self.ctc_head, *self.aux_heads.values()]:
                head.bias[0] = cfg.blank_bias_init

    # ------------------------------------------------------------------
    def encode(self, feats: torch.Tensor, feat_lengths: torch.Tensor, lang_ids: torch.Tensor | None = None) -> dict:
        x, lengths = self.subsampling(feats, feat_lengths)
        mask = lengths_to_mask(lengths, x.shape[1])
        if self.lang_embed is not None:
            if lang_ids is None:
                lang_ids = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            x = x + self.lang_embed(lang_ids).unsqueeze(1)
        aux_logits: dict[int, torch.Tensor] = {}
        for i, block in enumerate(self.blocks, start=1):
            if self.cfg.gradient_checkpointing and self.training:
                x = checkpoint(block, x, mask, use_reentrant=False)
            else:
                x = block(x, mask)
            key = str(i)
            if key in self.aux_heads:
                logits = self.aux_heads[key](x)
                aux_logits[i] = logits
                if self.self_cond is not None:
                    x = x + self.self_cond[key](logits.float().softmax(-1).to(x.dtype))
        out = {"hidden": x, "lengths": lengths, "mask": mask, "logits": self.ctc_head(x), "aux_logits": aux_logits}
        if self.lid_head is not None:
            m = mask.unsqueeze(-1).to(x.dtype)
            pooled = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
            out["lid_logits"] = self.lid_head(pooled)
        return out

    def forward(self, wav: torch.Tensor, wav_lengths: torch.Tensor, lang_ids: torch.Tensor | None = None,
                spec_augment: nn.Module | None = None) -> dict:
        feats, feat_lengths = self.frontend(wav, wav_lengths)
        if spec_augment is not None and self.training:
            feats = spec_augment(feats, feat_lengths)
        out = self.encode(feats, feat_lengths, lang_ids)
        out["log_probs"] = F.log_softmax(out["logits"].float(), dim=-1)
        out["aux_log_probs"] = {k: F.log_softmax(v.float(), dim=-1) for k, v in out["aux_logits"].items()}
        return out

    # ------------------------------------------------------------------
    def num_parameters(self, include_aux: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if not include_aux:
            n -= sum(p.numel() for p in self.aux_heads.parameters())
        return n

    def parameter_breakdown(self) -> dict[str, int]:
        def count(m):
            return sum(p.numel() for p in m.parameters()) if m is not None else 0

        return {
            "subsampling": count(self.subsampling),
            "blocks": count(self.blocks),
            "ctc_head": count(self.ctc_head),
            "aux_heads": count(self.aux_heads),
            "self_conditioning": count(self.self_cond),
            "language_embedding": count(self.lang_embed),
            "lid_head": count(self.lid_head),
            "total": count(self),
        }
