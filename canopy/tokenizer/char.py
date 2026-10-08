"""Character tokenizers.

``unit="codepoint"``: one token per Unicode code point (NFC).
``unit="grapheme"``: a base character plus its following combining marks form
one token, so vowel signs and other diacritics stay attached to their letter.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from collections.abc import Iterable

from canopy.tokenizer.base import BLANK, UNK, UNK_ID, CTCTokenizer

SPACE = " "


def split_units(text: str, unit: str) -> list[str]:
    if unit == "codepoint":
        return list(text)
    if unit != "grapheme":
        raise ValueError(f"unit must be codepoint|grapheme, got {unit!r}")
    out: list[str] = []
    for ch in text:
        if out and out[-1] != SPACE and unicodedata.category(ch) in ("Mn", "Me", "Mc"):
            out[-1] += ch
        else:
            out.append(ch)
    return out


class CharTokenizer(CTCTokenizer):
    type = "char"

    def __init__(self, tokens, language_token_ids=None, meta=None):
        super().__init__(tokens, language_token_ids, meta)
        self.unit = self.meta.get("unit", "codepoint")

    @classmethod
    def train(cls, texts_by_language: dict[str, Iterable[str]], unit: str = "codepoint", min_count: int = 1) -> CharTokenizer:
        counts: Counter = Counter()
        per_lang: dict[str, Counter] = {}
        for lang, texts in texts_by_language.items():
            c: Counter = Counter()
            for t in texts:
                c.update(split_units(t, unit))
            per_lang[lang] = c
            counts.update(c)
        units = sorted(u for u, n in counts.items() if n >= min_count and u != SPACE)
        tokens = [BLANK, UNK, SPACE, *units]
        idx = {t: i for i, t in enumerate(tokens)}
        lang_ids = {lang: [idx[u] for u in c if u in idx] for lang, c in per_lang.items()}
        meta = {"unit": unit, "min_count": min_count, "unit_counts": {u: counts[u] for u in units}}
        return cls(tokens, lang_ids, meta)

    @classmethod
    def from_payload(cls, payload: dict) -> CharTokenizer:
        return cls(payload["tokens"], payload.get("language_token_ids"), payload.get("meta"))

    def always_allowed_ids(self) -> list[int]:
        return [0, self.token_to_id[SPACE]]

    def encode(self, text: str, language: str | None = None) -> list[int]:
        return [self.token_to_id.get(u, UNK_ID) for u in split_units(text, self.unit)]

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            if i <= 0:
                continue
            out.append("\u2047" if i == UNK_ID else self.tokens[i])
        return " ".join("".join(out).split())

    def labels_for_pyctcdecode(self) -> list[str]:
        return ["", "\u2047", *self.tokens[2:]]
