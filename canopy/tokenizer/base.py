"""Common CTC tokenizer interface. Id 0 is always the CTC blank, id 1 is <unk>."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()

BLANK = "<blank>"
UNK = "<unk>"
BLANK_ID = 0
UNK_ID = 1


class CTCTokenizer:
    type: str = "base"

    def __init__(self, tokens: list[str], language_token_ids: dict[str, list[int]] | None = None, meta: dict | None = None):
        if tokens[BLANK_ID] != BLANK or tokens[UNK_ID] != UNK:
            raise ValueError("token 0 must be <blank> and token 1 must be <unk>")
        self.tokens = tokens
        self.token_to_id = {t: i for i, t in enumerate(tokens)}
        self.language_token_ids = {k: sorted(set(v)) for k, v in (language_token_ids or {}).items()}
        self.meta = meta or {}
        self.version: str | None = None

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def blank_id(self) -> int:
        return BLANK_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    def encode(self, text: str, language: str | None = None) -> list[int]:
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        raise NotImplementedError

    def always_allowed_ids(self) -> list[int]:
        return [BLANK_ID]

    def language_mask(self, language: str) -> list[bool] | None:
        """Allowed output ids for ``language`` (O2 output masking), or None if unknown."""
        ids = self.language_token_ids.get(language)
        if ids is None:
            return None
        mask = [False] * self.vocab_size
        for i in ids:
            mask[i] = True
        for i in self.always_allowed_ids():
            mask[i] = True
        return mask

    def labels_for_pyctcdecode(self) -> list[str]:
        raise NotImplementedError

    # ------------------------------------------------------------------ io
    def _payload(self) -> dict:
        return {"type": self.type, "tokens": self.tokens, "language_token_ids": self.language_token_ids, "meta": self.meta}

    def save(self, directory: str | os.PathLike) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        path = d / "tokenizer.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._payload(), f, ensure_ascii=False, indent=1)
        self.version = sha256_file(path)[:16]
        return path


def load_tokenizer(directory: str | os.PathLike) -> CTCTokenizer:
    d = Path(directory)
    path = d / "tokenizer.json" if d.is_dir() else d
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    t = payload["type"]
    if t == "char":
        from canopy.tokenizer.char import CharTokenizer

        tok = CharTokenizer.from_payload(payload)
    elif t == "sentencepiece":
        from canopy.tokenizer.spm import SentencePieceTokenizer

        tok = SentencePieceTokenizer.from_payload(payload, path.parent)
    else:
        raise ValueError(f"unknown tokenizer type {t!r}")
    tok.version = sha256_file(path)[:16]
    return tok
