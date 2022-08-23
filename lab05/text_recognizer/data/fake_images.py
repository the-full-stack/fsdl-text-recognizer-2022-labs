"""A fake image dataset for testing."""
import argparse

import torch
import torchvision

from text_recognizer.data.base_data_module import BaseDataModule


_NUM_SAMPLES = 512
_IMAGE_LEN = 28
_NUM_CLASSES = 10


class FakeImageData(BaseDataModule):
    """Fake images dataset."""

    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)
        self.num_samples = self.args.get("num_samples", _NUM_SAMPLES)
        self.input_dims = (1, self.args.get("image_height", _IMAGE_LEN), self.args.get("image_width", _IMAGE_LEN))

        self.num_classes = self.args.get("num_classes", _NUM_CLASSES)
        self.output_dims = (self.num_classes, 1)
        self.mapping = list(range(0, self.num_classes))

    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--num_samples", type=int, default=_NUM_SAMPLES)
        parser.add_argument("--num_classes", type=int, default=_NUM_CLASSES)
        parser.add_argument("--image_height", type=int, default=_IMAGE_LEN)
        parser.add_argument("--image_width", type=int, default=_IMAGE_LEN)
        return parser

    def setup(self, stage: str = None) -> None:
        fake_dataset = torchvision.datasets.FakeData(
            size=self.num_samples,
            image_size=self.input_dims,
            num_classes=self.output_dims[0],
            transform=torchvision.transforms.ToTensor(),
        )
        val_size = int(self.num_samples * 0.25)
        self.data_train, self.data_val, self.data_test = torch.utils.data.random_split(  # type: ignore
            dataset=fake_dataset, lengths=[self.num_samples - 2 * val_size, val_size, val_size]
        )
