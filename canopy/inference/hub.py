"""Resolve a Canopy variant to a Hugging Face repo and checkpoint file.

A variant is either a short name from ``VARIANTS`` or a full ``org/repo`` id.
Add a line to ``VARIANTS`` when a new checkpoint is published. Passing
``org/repo`` directly works before that line exists.
"""

from __future__ import annotations

from pathlib import Path

# Short name -> Hub repo. Weights stay on the Hub; this table is only the address.
VARIANTS: dict[str, str] = {
    "canopy-m": "ProximaAI/Canopy-M",
}


def resolve_repo(model: str) -> str:
    """Return an ``org/repo`` id. Local checkpoint paths are not repos."""
    key = model.strip()
    if not key:
        raise ValueError("model name is empty")
    if key.lower() in VARIANTS:
        return VARIANTS[key.lower()]
    if "/" in key and not Path(key).exists():
        return key
    known = ", ".join(sorted(VARIANTS))
    raise ValueError(
        f"unknown Canopy variant {model!r}. Use one of: {known}. "
        "Or pass a Hub id such as ProximaAI/Canopy-M."
    )


def checkpoint_name(repo_id: str) -> str:
    """Default weight file: the repo name, lowercased, with a .pt suffix."""
    name = repo_id.split("/")[-1].strip().lower()
    return name if name.endswith(".pt") else f"{name}.pt"


def pick_checkpoint(repo_id: str, files: list[str], filename: str | None = None) -> str:
    """Choose one ``.pt`` at the repo root. ``filename`` wins when it is present."""
    root = [f for f in files if "/" not in f and f.endswith(".pt")]
    if filename:
        if filename not in files and filename not in root:
            raise FileNotFoundError(f"{repo_id} has no file {filename!r}. Root checkpoints: {root or 'none'}")
        return filename
    preferred = checkpoint_name(repo_id)
    if preferred in root:
        return preferred
    if len(root) == 1:
        return root[0]
    raise FileNotFoundError(
        f"{repo_id} has {len(root)} checkpoint files {root}. Pass filename= to choose one."
    )
