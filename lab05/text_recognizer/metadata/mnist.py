"""Metadata for the MNIST dataset."""
from __future__ import annotations

from text_recognizer.metadata import shared

DOWNLOADED_DATA_DIRNAME = shared.DOWNLOADED_DATA_DIRNAME

DIMS = (1, 28, 28)
OUTPUT_DIMS = (1,)
MAPPING = list(range(10))

TRAIN_SIZE = 55000
VAL_SIZE = 5000
