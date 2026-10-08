"""Transcribe with any published Canopy checkpoint.

GitHub install (one package, every variant):

    pip install torch torchaudio soundfile numpy pyyaml huggingface_hub
    pip install "git+https://github.com/mwzkhalil/canopy.git@danish#subdirectory=canopy"
    hf auth login
    canopy-transcribe clip.wav --model canopy-m --language urd
    canopy-transcribe clip.wav --model ProximaAI/Canopy-M --language snd
"""

from __future__ import annotations

import argparse

from canopy.inference.api import LANGUAGE_ALIASES, Canopy
from canopy.inference.hub import VARIANTS


def main() -> None:
    names = ", ".join(sorted(VARIANTS))
    ap = argparse.ArgumentParser(description="Transcribe with a Canopy checkpoint")
    ap.add_argument("audio", nargs="+", help="wav, flac, or mp3 paths")
    ap.add_argument("--model", default="canopy-m", help=f"variant name ({names}) or org/repo id")
    ap.add_argument("--filename", default=None, help="checkpoint filename when a repo has several")
    ap.add_argument("--language", required=True, help="ISO code, for example urd or snd")
    ap.add_argument("--device", default=None, help="cpu, cuda, or omit for auto")
    ap.add_argument("--decoder", choices=["greedy", "beam"], default="greedy")
    args = ap.parse_args()
    asr = Canopy.from_pretrained(args.model, filename=args.filename, device=args.device)
    for path in args.audio:
        text = asr.transcribe(path, language=args.language, decoder=args.decoder)
        print(f"{path}\t{text}")


__all__ = ["LANGUAGE_ALIASES", "main"]
