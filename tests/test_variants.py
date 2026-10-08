import pytest

from canopy.inference.hub import checkpoint_name, pick_checkpoint, resolve_repo


def test_short_name_and_repo_id():
    assert resolve_repo("canopy-m") == "ProximaAI/Canopy-M"
    assert resolve_repo("CANOPY-M") == "ProximaAI/Canopy-M"
    assert resolve_repo("ProximaAI/Canopy-S") == "ProximaAI/Canopy-S"


def test_unknown_short_name():
    with pytest.raises(ValueError, match="canopy-m"):
        resolve_repo("canopy-s")


def test_checkpoint_choice():
    files = ["README.md", "canopy-m.pt", "figures/overview_cer_wer.png", "tokenizer.json"]
    assert checkpoint_name("ProximaAI/Canopy-M") == "canopy-m.pt"
    assert pick_checkpoint("ProximaAI/Canopy-M", files) == "canopy-m.pt"
    assert pick_checkpoint("ProximaAI/Canopy-M", files, filename="canopy-m.pt") == "canopy-m.pt"
    assert pick_checkpoint("ProximaAI/Other", ["only.pt"]) == "only.pt"
    with pytest.raises(FileNotFoundError):
        pick_checkpoint("ProximaAI/Other", ["a.pt", "b.pt"])
