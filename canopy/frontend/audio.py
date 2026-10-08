"""Audio loading and resampling. All model inputs are 16 kHz mono float32."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch

from canopy import SAMPLE_RATE


@dataclass
class AudioInfo:
    sample_rate: int
    channels: int
    frames: int

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate if self.sample_rate else 0.0


def audio_info(path: str | os.PathLike) -> AudioInfo:
    import soundfile as sf

    i = sf.info(str(path))
    return AudioInfo(sample_rate=int(i.samplerate), channels=int(i.channels), frames=int(i.frames))


def _read(path: str) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf

        data, sr = sf.read(path, dtype="float32", always_2d=True)
        return data, int(sr)
    except Exception as sf_err:
        try:
            import torchaudio

            wav, sr = torchaudio.load(path)
            return wav.numpy().T.astype(np.float32), int(sr)
        except Exception as ta_err:
            raise RuntimeError(
                f"could not decode {path!r} (soundfile: {sf_err}; torchaudio: {ta_err}). "
                "For MP3 input use libsndfile >= 1.1 (soundfile >= 0.12) or convert with scripts/prepare_data.py --convert."
            ) from ta_err


def resample(wav: torch.Tensor, orig_sr: int, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    if orig_sr == target_sr:
        return wav
    import torchaudio.functional as taf

    return taf.resample(wav, orig_sr, target_sr)


def load_audio(path: str | os.PathLike, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Load an audio file as a 1-D float32 tensor at ``target_sr``, downmixed to mono."""
    data, sr = _read(str(path))
    wav = torch.from_numpy(np.ascontiguousarray(data.mean(axis=1)))
    return resample(wav, sr, target_sr)


def save_audio(path: str | os.PathLike, wav: torch.Tensor, sr: int = SAMPLE_RATE) -> None:
    import soundfile as sf

    sf.write(str(path), wav.detach().cpu().numpy().astype(np.float32), sr)
