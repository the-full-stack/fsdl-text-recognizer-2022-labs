"""EMNIST dataset. Downloads from NIST website and saves as .npz file if not already present."""
import json
import os
from pathlib import Path
import shutil
from typing import Sequence
import zipfile

import h5py
import numpy as np
import toml

from text_recognizer.data.base_data_module import _download_raw_dataset, BaseDataModule, load_and_print_info
from text_recognizer.data.util import BaseDataset, split_dataset
import text_recognizer.metadata.emnist as metadata
from text_recognizer.stems.image import ImageStem
from text_recognizer.util import temporary_working_directory

NUM_SPECIAL_TOKENS = metadata.NUM_SPECIAL_TOKENS

RAW_DATA_DIRNAME = metadata.RAW_DATA_DIRNAME
METADATA_FILENAME = metadata.METADATA_FILENAME
DL_DATA_DIRNAME = metadata.DL_DATA_DIRNAME
PROCESSED_DATA_DIRNAME = metadata.PROCESSED_DATA_DIRNAME
PROCESSED_DATA_FILENAME = metadata.PROCESSED_DATA_FILENAME
ESSENTIALS_FILENAME = metadata.ESSENTIALS_FILENAME

SAMPLE_TO_BALANCE = True  # If true, take at most the mean number of instances per class.
TRAIN_FRAC = 0.8


class EMNIST(BaseDataModule):
    """EMNIST dataset of handwritten characters and digits.

    "The EMNIST dataset is a set of handwritten character digits derived from the NIST Special Database 19
    and converted to a 28x28 pixel image format and dataset structure that directly matches the MNIST dataset."
    From https://www.nist.gov/itl/iad/image-group/emnist-dataset

    The data split we will use is
    EMNIST ByClass: 814,255 characters. 62 unbalanced classes.
    """

    def __init__(self, args=None):
        super().__init__(args)

        self.mapping = metadata.MAPPING
        self.inverse_mapping = {v: k for k, v in enumerate(self.mapping)}
        self.transform = ImageStem()
        self.input_dims = metadata.DIMS
        self.output_dims = metadata.OUTPUT_DIMS

    def prepare_data(self, *args, **kwargs) -> None:
        if not os.path.exists(PROCESSED_DATA_FILENAME):
            _download_and_process_emnist()

    def setup(self, stage: str = None) -> None:
        if stage == "fit" or stage is None:
            with h5py.File(PROCESSED_DATA_FILENAME, "r") as f:
                self.x_trainval = f["x_train"][:]
                self.y_trainval = f["y_train"][:].squeeze().astype(int)

            data_trainval = BaseDataset(self.x_trainval, self.y_trainval, transform=self.transform)
            self.data_train, self.data_val = split_dataset(base_dataset=data_trainval, fraction=TRAIN_FRAC, seed=42)

        if stage == "test" or stage is None:
            with h5py.File(PROCESSED_DATA_FILENAME, "r") as f:
                self.x_test = f["x_test"][:]
                self.y_test = f["y_test"][:].squeeze().astype(int)
            self.data_test = BaseDataset(self.x_test, self.y_test, transform=self.transform)

    def __repr__(self):
        basic = f"EMNIST Dataset\nNum classes: {len(self.mapping)}\nMapping: {self.mapping}\nDims: {self.input_dims}\n"
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic

        x, y = next(iter(self.train_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
        )
        return basic + data


def _download_and_process_emnist():
    metadata = toml.load(METADATA_FILENAME)
    _download_raw_dataset(metadata, DL_DATA_DIRNAME)
    _process_raw_dataset(metadata["filename"], DL_DATA_DIRNAME)


def _process_raw_dataset(filename: str, dirname: Path):
    print("Unzipping EMNIST...")
    with temporary_working_directory(dirname):
        with zipfile.ZipFile(filename, "r") as zf:
            zf.extract("matlab/emnist-byclass.mat")

        from scipy.io import loadmat

        # NOTE: If importing at the top of module, would need to list scipy as prod dependency.

        print("Loading training data from .mat file")
        data = loadmat("matlab/emnist-byclass.mat")
        x_train = data["dataset"]["train"][0, 0]["images"][0, 0].reshape(-1, 28, 28).swapaxes(1, 2)
        y_train = data["dataset"]["train"][0, 0]["labels"][0, 0] + NUM_SPECIAL_TOKENS
        x_test = data["dataset"]["test"][0, 0]["images"][0, 0].reshape(-1, 28, 28).swapaxes(1, 2)
        y_test = data["dataset"]["test"][0, 0]["labels"][0, 0] + NUM_SPECIAL_TOKENS
        # NOTE that we add NUM_SPECIAL_TOKENS to targets, since these tokens are the first class indices

        if SAMPLE_TO_BALANCE:
            print("Balancing classes to reduce amount of data")
            x_train, y_train = _sample_to_balance(x_train, y_train)
            x_test, y_test = _sample_to_balance(x_test, y_test)

        print("Saving to HDF5 in a compressed format...")
        PROCESSED_DATA_DIRNAME.mkdir(parents=True, exist_ok=True)
        with h5py.File(PROCESSED_DATA_FILENAME, "w") as f:
            f.create_dataset("x_train", data=x_train, dtype="u1", compression="lzf")
            f.create_dataset("y_train", data=y_train, dtype="u1", compression="lzf")
            f.create_dataset("x_test", data=x_test, dtype="u1", compression="lzf")
            f.create_dataset("y_test", data=y_test, dtype="u1", compression="lzf")

        print("Saving essential dataset parameters to text_recognizer/data...")
        mapping = {int(k): chr(v) for k, v in data["dataset"]["mapping"][0, 0]}
        characters = _augment_emnist_characters(list(mapping.values()))
        essentials = {"characters": characters, "input_shape": list(x_train.shape[1:])}
        with open(ESSENTIALS_FILENAME, "w") as f:
            json.dump(essentials, f)

        print("Cleaning up...")
        shutil.rmtree("matlab")


def _sample_to_balance(x, y):
    """Because the dataset is not balanced, we take at most the mean number of instances per class."""
    np.random.seed(42)
    num_to_sample = int(np.bincount(y.flatten()).mean())
    all_sampled_inds = []
    for label in np.unique(y.flatten()):
        inds = np.where(y == label)[0]
        sampled_inds = np.unique(np.random.choice(inds, num_to_sample))
        all_sampled_inds.append(sampled_inds)
    ind = np.concatenate(all_sampled_inds)
    x_sampled = x[ind]
    y_sampled = y[ind]
    return x_sampled, y_sampled


def _augment_emnist_characters(characters: Sequence[str]) -> Sequence[str]:
    """Augment the mapping with extra symbols."""
    # Extra characters from the IAM dataset
    iam_characters = [
        " ",
        "!",
        '"',
        "#",
        "&",
        "'",
        "(",
        ")",
        "*",
        "+",
        ",",
        "-",
        ".",
        "/",
        ":",
        ";",
        "?",
    ]

    # Also add special tokens:
    # - CTC blank token at index 0
    # - Start token at index 1
    # - End token at index 2
    # - Padding token at index 3
    # NOTE: Don't forget to update NUM_SPECIAL_TOKENS if changing this!
    return ["<B>", "<S>", "<E>", "<P>", *characters, *iam_characters]


if __name__ == "__main__":
    load_and_print_info(EMNIST)
