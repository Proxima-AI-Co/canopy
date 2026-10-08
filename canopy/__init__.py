"""Canopy: load any published Canopy speech-recognition variant."""

__version__ = "0.1.0"

BLANK_ID = 0
SAMPLE_RATE = 16000

from canopy.inference.api import Canopy

__all__ = ["BLANK_ID", "Canopy", "SAMPLE_RATE", "__version__"]
