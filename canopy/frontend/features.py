"""Log-mel frontend implemented in plain PyTorch (ONNX-exportable, no torchaudio needed).

Defaults: 25 ms Hann window, 10 ms hop, 512-point FFT, 80 Slaney mel bins, 100 fps.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from canopy import SAMPLE_RATE


def _hz_to_mel_slaney(f: torch.Tensor) -> torch.Tensor:
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    mel = f / f_sp
    return torch.where(f >= min_log_hz, min_log_mel + torch.log(f.clamp(min=1e-10) / min_log_hz) / logstep, mel)


def _mel_to_hz_slaney(m: torch.Tensor) -> torch.Tensor:
    f_sp = 200.0 / 3
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    return torch.where(m >= min_log_mel, min_log_hz * torch.exp(logstep * (m - min_log_mel)), f_sp * m)


def mel_filterbank(n_freqs: int, n_mels: int, sample_rate: int, f_min: float = 0.0, f_max: float | None = None) -> torch.Tensor:
    """Slaney-scale, Slaney-normalised triangular filterbank of shape (n_freqs, n_mels)."""
    f_max = f_max if f_max is not None else sample_rate / 2
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs, dtype=torch.float64)
    m_pts = torch.linspace(_hz_to_mel_slaney(torch.tensor(f_min, dtype=torch.float64)).item(),
                           _hz_to_mel_slaney(torch.tensor(f_max, dtype=torch.float64)).item(), n_mels + 2,
                           dtype=torch.float64)
    f_pts = _mel_to_hz_slaney(m_pts)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
    down = -slopes[:, :-2] / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    fb = torch.clamp(torch.minimum(down, up), min=0.0)
    enorm = 2.0 / (f_pts[2 : n_mels + 2] - f_pts[:n_mels])
    return (fb * enorm.unsqueeze(0)).float()


def feature_lengths(num_samples: torch.Tensor, hop_length: int = 160) -> torch.Tensor:
    """Frame count for a centred STFT."""
    return torch.div(num_samples, hop_length, rounding_mode="floor") + 1


class LogMelFrontend(nn.Module):
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        n_fft: int = 512,
        win_length: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        normalize: str = "utterance",
        eps: float = 1e-10,
    ):
        super().__init__()
        if normalize not in ("utterance", "per_bin", "none"):
            raise ValueError(f"normalize must be utterance|per_bin|none, got {normalize!r}")
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.normalize = normalize
        self.eps = eps
        self.register_buffer("window", torch.hann_window(win_length), persistent=False)
        self.register_buffer("fbank", mel_filterbank(n_fft // 2 + 1, n_mels, sample_rate), persistent=False)

    @torch.no_grad()
    def forward(self, wav: torch.Tensor, lengths: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """wav (B, S) float -> features (B, T, n_mels), lengths (B,)."""
        with torch.autocast(device_type=wav.device.type, enabled=False):
            return self._forward(wav, lengths)

    def _forward(self, wav: torch.Tensor, lengths: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav = wav.float()
        B, S = wav.shape
        if lengths is None:
            lengths = torch.full((B,), S, dtype=torch.long, device=wav.device)
        spec = torch.stft(wav, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                          window=self.window, center=True, pad_mode="reflect", return_complex=True)
        power = spec.real.pow(2) + spec.imag.pow(2)  # (B, F, T)
        mel = torch.matmul(power.transpose(1, 2), self.fbank)  # (B, T, M)
        logmel = torch.log(mel.clamp(min=self.eps))
        feat_len = feature_lengths(lengths, self.hop_length).clamp(max=logmel.shape[1])
        mask = torch.arange(logmel.shape[1], device=wav.device).unsqueeze(0) < feat_len.unsqueeze(1)
        m = mask.unsqueeze(-1).float()
        if self.normalize == "utterance":
            n = (feat_len.float() * self.n_mels).view(B, 1, 1)
            mean = (logmel * m).sum(dim=(1, 2), keepdim=True) / n
            var = (((logmel - mean) * m) ** 2).sum(dim=(1, 2), keepdim=True) / n
            logmel = (logmel - mean) / (var.sqrt() + 1e-6)
        elif self.normalize == "per_bin":
            n = feat_len.float().view(B, 1, 1)
            mean = (logmel * m).sum(dim=1, keepdim=True) / n
            var = (((logmel - mean) * m) ** 2).sum(dim=1, keepdim=True) / n
            logmel = (logmel - mean) / (var.sqrt() + 1e-6)
        logmel = logmel * m
        return logmel, feat_len
