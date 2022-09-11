"""Base DataModule class."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Collection

import pytorch_lightning as pl
import torch
from loguru import logger as log
from torch.utils.data import ConcatDataset, DataLoader

import text_recognizer.metadata.shared as metadata
from text_recognizer import util
from text_recognizer.data.util import BaseDataset


def load_and_print_info(data_module_class) -> None:
    """Load EMNISTLines and print info."""
    parser = argparse.ArgumentParser()
    data_module_class.add_to_argparse(parser)
    args = parser.parse_args()
    dataset = data_module_class(args)
    dataset.prepare_data()
    dataset.setup()
    log.info(dataset)


def _download_raw_dataset(metadata: dict, dl_dirname: Path) -> Path:
    dl_dirname.mkdir(parents=True, exist_ok=True)
    filename = dl_dirname / metadata["filename"]
    if filename.exists():
        return filename
    log.info(f"Downloading raw dataset from {metadata['url']} to {filename}...")
    util.download_url(metadata["url"], filename)
    log.info("Computing SHA-256...")
    sha256 = util.compute_sha256(filename)
    if sha256 != metadata["sha256"]:
        raise ValueError(
            "Downloaded data file SHA-256 does not match that listed in metadata document.",
        )
    return filename


BATCH_SIZE = 128
NUM_AVAIL_CPUS = len(os.sched_getaffinity(0))
NUM_AVAIL_GPUS = torch.cuda.device_count()

# sensible multiprocessing defaults: at most one worker per CPU
DEFAULT_NUM_WORKERS = NUM_AVAIL_CPUS
# but in distributed data parallel mode, we launch a training on each GPU, so must divide out to
# keep total at one worker per CPU
DEFAULT_NUM_WORKERS = NUM_AVAIL_CPUS // NUM_AVAIL_GPUS if NUM_AVAIL_GPUS else DEFAULT_NUM_WORKERS


class BaseDataModule(pl.LightningDataModule):
    """Base for all of our LightningDataModules.

    Learn more at about LDMs at
    https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
    """

    def __init__(self, args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.batch_size = self.args.get("batch_size", BATCH_SIZE)
        self.num_workers = self.args.get("num_workers", DEFAULT_NUM_WORKERS)

        self.on_gpu = isinstance(self.args.get("gpus", None), (str, int))

        # Make sure to set the variables below in subclasses
        self.input_dims: tuple[int, ...]
        self.output_dims: tuple[int, ...]
        self.mapping: Collection
        self.data_train: BaseDataset | ConcatDataset
        self.data_val: BaseDataset | ConcatDataset
        self.data_test: BaseDataset | ConcatDataset

    @classmethod
    def data_dirname(cls):
        return metadata.DATA_DIRNAME

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument(
            "--batch_size",
            type=int,
            default=BATCH_SIZE,
            help=f"Number of examples to operate on per forward step. Default is {BATCH_SIZE}.",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=DEFAULT_NUM_WORKERS,
            help=f"Number of additional processes to load data. Default is {DEFAULT_NUM_WORKERS}.",
        )
        return parser

    def config(self):
        """Return important settings of the dataset, which will be passed to instantiate models."""
        return {
            "input_dims": self.input_dims,
            "output_dims": self.output_dims,
            "mapping": self.mapping,
        }

    def prepare_data(self, *args, **kwargs) -> None:
        """Take the first steps to prepare data for use.

        Use this method to do things that might write to disk or that need to be done only from a
        single GPU in distributed settings (so don't set state `self.x = y`).
        """

    def setup(self, stage: str | None = None) -> None:
        """Perform final setup to prepare data for consumption by DataLoader.

        Here is where we typically split into train, validation, and test. This is done once per GPU
        in a DDP setting. Should assign `torch Dataset` objects to self.data_train, self.data_val,
        and optionally self.data_test.
        """

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.on_gpu,
        )
