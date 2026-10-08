"""CTC prefix beam search with optional word-level n-gram shallow fusion.

score(prefix) = log P_ctc(prefix) + alpha * log P_lm(words) + beta * |words|

A word is scored by the LM when it is completed, i.e. when the next emitted
token string contains a space (character tokenizers emit " "; SentencePiece
pieces that start a word begin with "▁", rendered as " ").
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

NEG_INF = -math.inf


def _lse(a: float, b: float) -> float:
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


class _LMState:
    __slots__ = ("raw", "history", "lm", "words")

    def __init__(self, raw: str, history: tuple[str, ...], lm: float, words: int):
        self.raw, self.history, self.lm, self.words = raw, history, lm, words


def _pending_word(raw: str) -> str | None:
    if not raw or raw.endswith(" "):
        return None
    parts = raw.split()
    return parts[-1] if parts else None


def prefix_beam_search(log_probs: np.ndarray, token_strings: Sequence[str], *, beam: int = 10, lm=None,
                       alpha: float = 0.5, beta: float = 1.0, token_prune: int = 20, prune_logp: float = -12.0,
                       allowed: np.ndarray | None = None, blank: int = 0) -> list[int]:
    """Return the best token-id sequence for one utterance.

    ``log_probs``: (T, V) natural-log posteriors. ``token_strings[i]``: surface
    string of token i ("" for blank). ``lm``: object with
    ``score(history, word) -> ln prob``, ``score_end(history) -> ln prob`` and
    ``order``. ``allowed``: optional bool (V,) output mask.
    """
    T, V = log_probs.shape
    lp = log_probs.copy()
    if allowed is not None:
        lp[:, ~allowed] = NEG_INF
    use_lm = lm is not None and alpha != 0.0
    hist_len = max(0, (lm.order - 1)) if use_lm else 0

    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, NEG_INF)}
    lm_states: dict[tuple[int, ...], _LMState] = {(): _LMState("", (), 0.0, 0)}

    def lm_state(prefix: tuple[int, ...]) -> _LMState:
        st = lm_states.get(prefix)
        if st is not None:
            return st
        parent = lm_state(prefix[:-1])
        s = token_strings[prefix[-1]]
        raw = parent.raw + s
        history, lm_score, words = parent.history, parent.lm, parent.words
        if " " in s:
            w = _pending_word(parent.raw)
            if w is not None:
                words += 1
                if use_lm:
                    lm_score += lm.score(history, w)
                    history = (history + (w,))[-hist_len:] if hist_len else ()
        st = _LMState(raw, history, lm_score, words)
        lm_states[prefix] = st
        return st

    def total(prefix, pb, pnb):
        st = lm_state(prefix)
        return _lse(pb, pnb) + alpha * st.lm + beta * st.words

    for t in range(T):
        row = lp[t]
        cand = np.argsort(row)[::-1][:token_prune]
        cand = [int(c) for c in cand if row[c] > prune_logp or c == blank]
        nxt: dict[tuple[int, ...], list[float]] = {}

        def add(p, b, nb, nxt=nxt):
            cur = nxt.get(p)
            if cur is None:
                nxt[p] = [b, nb]
            else:
                cur[0] = _lse(cur[0], b)
                cur[1] = _lse(cur[1], nb)

        for prefix, (pb, pnb) in beams.items():
            for c in cand:
                p = float(row[c])
                if c == blank:
                    add(prefix, _lse(pb, pnb) + p, NEG_INF)
                    continue
                last = prefix[-1] if prefix else None
                ext = prefix + (c,)
                if c == last:
                    add(ext, NEG_INF, pb + p)
                    add(prefix, NEG_INF, pnb + p)
                else:
                    add(ext, NEG_INF, _lse(pb, pnb) + p)
        scored = sorted(nxt.items(), key=lambda kv: total(kv[0], kv[1][0], kv[1][1]), reverse=True)[:beam]
        beams = {k: (v[0], v[1]) for k, v in scored}

    best, best_score = (), NEG_INF
    for prefix, (pb, pnb) in beams.items():
        st = lm_state(prefix)
        lm_score, words = st.lm, st.words
        w = _pending_word(st.raw)
        history = st.history
        if w is not None:
            words += 1
            if use_lm:
                lm_score += lm.score(history, w)
                history = (history + (w,))[-hist_len:] if hist_len else ()
        if use_lm:
            lm_score += lm.score_end(history)
        s = _lse(pb, pnb) + alpha * lm_score + beta * words
        if s > best_score:
            best, best_score = prefix, s
    return list(best)
