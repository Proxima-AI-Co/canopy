from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from canopy import SAMPLE_RATE
from canopy.decoding.beam import prefix_beam_search
from canopy.decoding.greedy import greedy_decode, language_mask_bias
from canopy.frontend.audio import load_audio
from canopy.tokenizer.base import load_tokenizer
from canopy.checkpoint import load_checkpoint, model_from_checkpoint


def _tokenizer_from_checkpoint(ck: dict, fallback_dir: str | None = None):
    if "tokenizer_payload" in ck:
        tmp = Path(tempfile.mkdtemp(prefix="canopy_tok_"))
        (tmp / "tokenizer.json").write_text(json.dumps(ck["tokenizer_payload"], ensure_ascii=False), encoding="utf-8")
        if ck.get("spm_model"):
            (tmp / "spm.model").write_bytes(ck["spm_model"])
        return load_tokenizer(tmp)
    return load_tokenizer(fallback_dir or ck["tokenizer_dir"])


class Transcriber:
    """Load a checkpoint and transcribe audio.

    ``language``: ISO 639-3 code. When given, the model is conditioned on it
    (if trained with a language embedding), the output is restricted to that
    language's tokens, and that language's LM is used. ``None`` runs
    language-agnostic decoding.
    """

    def __init__(self, model, tokenizer, device: str = "cpu", max_chunk_seconds: float = 30.0):
        self.model = model.to(device).eval()
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.languages = list(model.cfg.languages)
        self.lang2id = {lang: i + 1 for i, lang in enumerate(self.languages)}
        self.max_chunk = int(max_chunk_seconds * SAMPLE_RATE)
        self.lms: dict[str, object] = {}

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu", tokenizer_dir: str | None = None, **kw) -> Transcriber:
        ck = load_checkpoint(path)
        model = model_from_checkpoint(ck)
        return cls(model, _tokenizer_from_checkpoint(ck, tokenizer_dir), device=device, **kw)

    @classmethod
    def from_pretrained(cls, repo_id: str = "ProximaAI/Canopy-M", *, filename: str | None = None,
                        device: str = "cpu", revision: str | None = None, token: str | None = None, **kw) -> Transcriber:
        """Download an inference checkpoint from the Hugging Face Hub and load it.

        ``repo_id`` is an ``org/name`` id. ``filename`` defaults to the single
        ``.pt`` at the repo root, preferring ``<repo-name>.pt``.
        """
        try:
            from huggingface_hub import hf_hub_download, list_repo_files
        except ImportError as exc:
            raise ImportError("pip install huggingface_hub") from exc
        from canopy.inference.hub import pick_checkpoint

        files = list_repo_files(repo_id, revision=revision, token=token)
        chosen = pick_checkpoint(repo_id, files, filename)
        path = hf_hub_download(repo_id=repo_id, filename=chosen, revision=revision, token=token)
        return cls.from_checkpoint(path, device=device, **kw)

    def attach_lm(self, language: str, lm) -> None:
        """``lm``: ArpaLM or KenLMScorer. Use language "*" for a language-agnostic LM."""
        self.lms[language] = lm

    @torch.no_grad()
    def log_probs(self, wav: torch.Tensor, language: str | None = None) -> torch.Tensor:
        lid = None
        if language is not None and self.model.lang_embed is not None:
            lid = torch.tensor([self.lang2id.get(language, 0)], device=self.device)
        chunks = [wav[i : i + self.max_chunk] for i in range(0, max(1, wav.numel()), self.max_chunk)]
        outs = []
        for c in chunks:
            if c.numel() < 400:
                continue
            x = c.unsqueeze(0).to(self.device)
            res = self.model(x, torch.tensor([c.numel()], device=self.device), lid)
            outs.append(res["log_probs"][0, : int(res["lengths"][0])])
        if not outs:
            return torch.zeros(0, self.tokenizer.vocab_size)
        return torch.cat(outs, dim=0)

    def transcribe(self, audio, language: str | None = None, decoder: str = "greedy", beam: int = 10,
                   alpha: float = 0.5, beta: float = 1.0) -> str:
        wav = load_audio(audio) if isinstance(audio, (str, Path)) else torch.as_tensor(audio, dtype=torch.float32)
        lp = self.log_probs(wav, language)
        if lp.shape[0] == 0:
            return ""
        bias = language_mask_bias(self.tokenizer, [language], lp.shape[-1], device=lp.device) if language else None
        if decoder == "greedy":
            ids = greedy_decode(lp.unsqueeze(0), torch.tensor([lp.shape[0]]), bias)[0]
        elif decoder == "beam":
            labels = self.tokenizer.labels_for_pyctcdecode()
            allowed = np.isfinite(bias[0].cpu().numpy()) if bias is not None else None
            lm = self.lms.get(language or "*") or self.lms.get("*")
            ids = prefix_beam_search(lp.float().cpu().numpy(), [""] + labels[1:], beam=beam, lm=lm, alpha=alpha,
                                     beta=beta, allowed=allowed)
        else:
            raise ValueError(f"unknown decoder {decoder!r}")
        return self.tokenizer.decode(ids)
