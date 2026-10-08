from __future__ import annotations

from collections.abc import Sequence

import torch

from canopy.tokenizer.base import BLANK_ID


def ctc_collapse(ids: Sequence[int], blank: int = BLANK_ID) -> list[int]:
    """Merge repeats, then remove blanks."""
    out, prev = [], None
    for i in ids:
        if i != prev and i != blank:
            out.append(int(i))
        prev = i
    return out


def language_mask_bias(tokenizer, languages: Sequence[str | None], vocab_size: int, device=None) -> torch.Tensor | None:
    """(B, V) additive bias: 0 for allowed tokens, -inf for tokens never seen in that
    language's training text (output design O2). Rows for ``None`` or unknown
    languages are all zeros (unrestricted)."""
    if not any(languages):
        return None
    bias = torch.zeros(len(languages), vocab_size, device=device)
    for b, lang in enumerate(languages):
        if not lang:
            continue
        mask = tokenizer.language_mask(lang)
        if mask is None:
            continue
        m = torch.tensor(mask, dtype=torch.bool, device=device)
        bias[b, ~m] = float("-inf")
    return bias


@torch.no_grad()
def greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor, mask_bias: torch.Tensor | None = None) -> list[list[int]]:
    if mask_bias is not None:
        log_probs = log_probs + mask_bias.unsqueeze(1)
    best = log_probs.argmax(dim=-1).cpu()
    return [ctc_collapse(best[b, : int(lengths[b])].tolist()) for b in range(best.shape[0])]
