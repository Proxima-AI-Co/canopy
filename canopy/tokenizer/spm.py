"""SentencePiece (unigram / BPE) tokenizer with CTC blank at id 0.

SentencePiece's own ids are shifted by +1, so SentencePiece <unk> (0) becomes
Canopy id 1. Normalization is disabled inside SentencePiece
(``normalization_rule_name=identity``): Canopy's per-language normalizer has
already produced ``text_train``, and SentencePiece's default NFKC would fold
characters that are distinct in some Pakistani orthographies.
"""

from __future__ import annotations

import io
import math
import random
from collections.abc import Iterable
from pathlib import Path

from canopy.tokenizer.base import BLANK, UNK, CTCTokenizer

SPM_SPACE = "\u2581"


def balanced_sample(texts_by_language: dict[str, list[str]], temperature: float, seed: int = 0,
                    total: int | None = None) -> tuple[list[str], dict[str, int]]:
    """Resample sentences so language ``l`` gets a share ∝ n_l ** (1/temperature)."""
    rng = random.Random(seed)
    n = {lang: len(t) for lang, t in texts_by_language.items() if t}
    total = total or sum(n.values())
    if math.isinf(temperature):
        w = {lang: 1.0 for lang in n}
    else:
        w = {lang: c ** (1.0 / temperature) for lang, c in n.items()}
    z = sum(w.values())
    out, counts = [], {}
    for lang, texts in texts_by_language.items():
        if not texts:
            continue
        k = max(1, round(total * w[lang] / z))
        if k <= len(texts):
            picked = rng.sample(texts, k)
        else:
            picked = texts * (k // len(texts)) + rng.sample(texts, k % len(texts))
        out.extend(picked)
        counts[lang] = k
    rng.shuffle(out)
    return out, counts


class SentencePieceTokenizer(CTCTokenizer):
    type = "sentencepiece"

    def __init__(self, tokens, language_token_ids=None, meta=None, model_proto: bytes | None = None):
        super().__init__(tokens, language_token_ids, meta)
        import sentencepiece as spm

        self._proto = model_proto
        self.sp = spm.SentencePieceProcessor(model_proto=model_proto)

    @classmethod
    def train(cls, texts_by_language: dict[str, list[str]], vocab_size: int, model_type: str = "unigram",
              temperature: float = 5.0, seed: int = 0, character_coverage: float = 1.0,
              max_sentences: int | None = 2_000_000) -> SentencePieceTokenizer:
        import sentencepiece as spm

        sample, counts = balanced_sample(texts_by_language, temperature, seed,
                                         total=min(max_sentences or 10**12, sum(len(v) for v in texts_by_language.values())))
        buf = io.BytesIO()
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(sample), model_writer=buf, vocab_size=vocab_size, model_type=model_type,
            character_coverage=character_coverage, byte_fallback=False, split_digits=True,
            normalization_rule_name="identity", add_dummy_prefix=True, remove_extra_whitespaces=True,
            unk_id=0, bos_id=-1, eos_id=-1, pad_id=-1, hard_vocab_limit=False,
            input_sentence_size=0, shuffle_input_sentence=False, num_threads=4, minloglevel=2,
        )
        proto = buf.getvalue()
        sp = spm.SentencePieceProcessor(model_proto=proto)
        tokens = [BLANK, UNK] + [sp.id_to_piece(i) for i in range(1, sp.get_piece_size())]
        meta = {"model_type": model_type, "requested_vocab_size": vocab_size, "temperature": temperature,
                "seed": seed, "character_coverage": character_coverage, "training_sentences_per_language": counts}
        tok = cls(tokens, None, meta, proto)
        tok.language_token_ids = {
            lang: sorted({i for t in texts for i in tok.encode(t)}) for lang, texts in texts_by_language.items()
        }
        return tok

    @classmethod
    def from_payload(cls, payload: dict, directory: Path) -> SentencePieceTokenizer:
        proto = (directory / "spm.model").read_bytes()
        return cls(payload["tokens"], payload.get("language_token_ids"), payload.get("meta"), proto)

    def save(self, directory):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / "spm.model").write_bytes(self._proto)
        return super().save(d)

    def encode(self, text: str, language: str | None = None) -> list[int]:
        return [i + 1 for i in self.sp.encode(text, out_type=int)]

    def decode(self, ids: Iterable[int]) -> str:
        return self.sp.decode([i - 1 for i in ids if i > 0])

    def always_allowed_ids(self) -> list[int]:
        ids = [0]
        sp_space = self.token_to_id.get(SPM_SPACE)
        if sp_space is not None:
            ids.append(sp_space)
        return ids

    def labels_for_pyctcdecode(self) -> list[str]:
        return ["", "\u2047", *[t.replace(SPM_SPACE, " ") if t.startswith(SPM_SPACE) else t for t in self.tokens[2:]]]
