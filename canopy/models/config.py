from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields


@dataclass
class ModelConfig:
    # Frontend
    n_mels: int = 80
    feature_normalize: str = "utterance"  # utterance | per_bin | none
    # Encoder
    d_model: int = 384
    n_layers: int = 12
    n_heads: int = 6
    ff_multiplier: float = 3.0
    conv_kernel: int = 31
    conv_expansion: int = 2
    conv_norm: str = "batch"  # batch | layer
    se_ratio: int = 4
    dropout: float = 0.1
    stochastic_depth: float = 0.1
    # Output
    vocab_size: int = 0
    blank_bias_init: float = 0.0
    # Intermediate CTC heads after these blocks (1-indexed: 6 = output of the 6th block).
    aux_ctc_layers: list[int] = field(default_factory=lambda: [6])
    self_conditioning: bool = False
    # Language conditioning. `languages` fixes the id order; id 0 means "no language".
    languages: list[str] = field(default_factory=list)
    language_embedding: bool = False
    lid_head: bool = False
    # Memory
    gradient_checkpointing: bool = False

    @property
    def num_languages(self) -> int:
        return len(self.languages)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown model config keys: {sorted(unknown)}")
        return cls(**d)


PRESETS: dict[str, dict] = {
    "canopy-s": {"d_model": 320, "n_layers": 12, "n_heads": 5},
    "canopy-m": {"d_model": 384, "n_layers": 12, "n_heads": 6},
    "canopy-m16": {"d_model": 384, "n_layers": 16, "n_heads": 6},
    "canopy-l": {"d_model": 512, "n_layers": 16, "n_heads": 8, "aux_ctc_layers": [8]},
    # For tests and smoke runs only.
    "canopy-tiny": {"d_model": 64, "n_layers": 2, "n_heads": 2, "aux_ctc_layers": [1], "conv_kernel": 7},
}
