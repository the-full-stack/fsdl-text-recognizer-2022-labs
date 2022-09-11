"""Module containing submodules for each dataset.

Each dataset is defined as a class in that submodule.

The datasets should have a .config method that returns
any configuration information needed by the model.

Most datasets define their constants in a submodule
of the metadata module that is parallel to this one in the
hierarchy.
"""
from __future__ import annotations

from .base_data_module import BaseDataModule
from .emnist import EMNIST
from .emnist_lines import EMNISTLines
from .fake_images import FakeImageData
from .iam_lines import IAMLines
from .iam_original_and_synthetic_paragraphs import IAMOriginalAndSyntheticParagraphs
from .iam_paragraphs import IAMParagraphs
from .iam_synthetic_paragraphs import IAMSyntheticParagraphs
from .mnist import MNIST
from .util import BaseDataset
