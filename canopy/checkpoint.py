"""Checkpoint format.

Training checkpoints hold model, optimizer, scheduler, scaler and loop state.
Inference checkpoints (``export_inference_checkpoint``) hold only the model
config, weights, language list and the tokenizer payload, so one file is
enough to transcribe.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from canopy.models.canopy_ctc import CanopyCTC
from canopy.models.config import ModelConfig

FORMAT_VERSION = 1


def save_training_checkpoint(path: Path, *, model, optimizer, scheduler, scaler, state: dict, model_config: dict,
                             tokenizer_dir: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save({
        "format_version": FORMAT_VERSION,
        "kind": "training",
        "model_config": model_config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "loop_state": state,
        "tokenizer_dir": tokenizer_dir,
        "rng": {"torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
    }, tmp)
    tmp.replace(path)


def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    return torch.load(str(path), map_location=map_location, weights_only=False)


def export_inference_checkpoint(train_ckpt: str | Path, out: str | Path, tokenizer_dir: str | Path | None = None,
                                extra_meta: dict | None = None) -> Path:
    ck = load_checkpoint(train_ckpt)
    tok_dir = Path(tokenizer_dir or ck["tokenizer_dir"])
    payload = json.loads((tok_dir / "tokenizer.json").read_text(encoding="utf-8"))
    spm_model = (tok_dir / "spm.model").read_bytes() if (tok_dir / "spm.model").exists() else None
    state = {k: v for k, v in ck["model_state"].items() if not k.startswith("aux_heads.")}
    cfg = dict(ck["model_config"])
    if not cfg.get("self_conditioning"):
        cfg["aux_ctc_layers"] = []
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": FORMAT_VERSION,
        "kind": "inference",
        "model_config": cfg,
        "model_state": state,
        "tokenizer_payload": payload,
        "spm_model": spm_model,
        "meta": {"source_checkpoint": str(train_ckpt), "loop_state": ck.get("loop_state"), **(extra_meta or {})},
    }, out)
    return out


def model_from_checkpoint(ck: dict) -> CanopyCTC:
    cfg = ModelConfig.from_dict(ck["model_config"])
    model = CanopyCTC(cfg)
    missing, unexpected = model.load_state_dict(ck["model_state"], strict=False)
    unexpected = [k for k in unexpected if not k.startswith("aux_heads.")]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    return model
