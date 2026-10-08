"""User-facing loader for every Canopy checkpoint."""

from __future__ import annotations

from pathlib import Path

import torch

from canopy.inference.hub import resolve_repo
from canopy.inference.transcriber import Transcriber

LANGUAGE_ALIASES = {"ur": "urd", "urdu": "urd", "sd": "snd", "sindhi": "snd"}


def resolve_language(language: str | None) -> str | None:
    if language is None:
        return None
    return LANGUAGE_ALIASES.get(language.strip().lower(), language.strip().lower())


class Canopy:
    """One interface for every Canopy variant.

    ``model`` is a short name (``canopy-m``) or a Hub id (``ProximaAI/Canopy-M``).
    A path to a local ``.pt`` file loads that file and does not contact the Hub.
    """

    def __init__(self, transcriber: Transcriber, model: str):
        self.transcriber = transcriber
        self.model = model
        self.languages = list(transcriber.languages)

    @classmethod
    def from_pretrained(cls, model: str = "canopy-m", *, filename: str | None = None,
                        device: str | None = None, token: str | None = None,
                        revision: str | None = None) -> Canopy:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        path = Path(model)
        if path.is_file():
            asr = Transcriber.from_checkpoint(path, device=device)
            return cls(asr, str(path))
        repo_id = resolve_repo(model)
        asr = Transcriber.from_pretrained(
            repo_id, filename=filename, device=device, token=token, revision=revision,
        )
        return cls(asr, repo_id)

    def transcribe(self, audio, language: str | None = "urd", decoder: str = "greedy", **kw) -> str:
        code = resolve_language(language)
        if code is not None and self.languages and code not in self.languages:
            known = ", ".join(self.languages)
            raise ValueError(f"{self.model} was trained on {known}, not {language!r}")
        return self.transcriber.transcribe(audio, language=code, decoder=decoder, **kw)
