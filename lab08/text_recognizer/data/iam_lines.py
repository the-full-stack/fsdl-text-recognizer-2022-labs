"""A dataset of lines of handwritten text derived from the IAM dataset."""
import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageFile

from text_recognizer import util
from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info
from text_recognizer.data.iam import IAM
from text_recognizer.data.util import BaseDataset, convert_strings_to_labels, resize_image
import text_recognizer.metadata.iam_lines as metadata
from text_recognizer.stems.line import IAMLineStem

ImageFile.LOAD_TRUNCATED_IMAGES = True

PROCESSED_DATA_DIRNAME = metadata.PROCESSED_DATA_DIRNAME

IMAGE_SCALE_FACTOR = metadata.IMAGE_SCALE_FACTOR


class IAMLines(BaseDataModule):
    """Lines of text pulled from the IAM Handwriting database."""

    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)
        self.augment = self.args.get("augment_data", "true") == "true"
        self.mapping = metadata.MAPPING
        self.inverse_mapping = {v: k for k, v in enumerate(self.mapping)}
        self.input_dims = metadata.DIMS  # We assert that this is correct in setup()
        self.output_dims = metadata.OUTPUT_DIMS  # We assert that this is correct in setup()
        self.transform = IAMLineStem()
        self.trainval_transform = IAMLineStem(augment=self.augment)

    @staticmethod
    def add_to_argparse(parser):
        BaseDataModule.add_to_argparse(parser)
        parser.add_argument("--augment_data", type=str, default="true")
        return parser

    def prepare_data(self, *args, **kwargs) -> None:
        if PROCESSED_DATA_DIRNAME.exists():
            return

        print("Cropping IAM line regions...")
        iam = IAM()
        iam.prepare_data()
        crops_train, labels_train = generate_line_crops_and_labels(iam, "train")
        crops_val, labels_val = generate_line_crops_and_labels(iam, "val")
        crops_test, labels_test = generate_line_crops_and_labels(iam, "test")

        shapes = np.array([crop.size for crop in crops_train + crops_val + crops_test])
        aspect_ratios = shapes[:, 0] / shapes[:, 1]

        print("Saving images, labels, and statistics...")
        save_images_and_labels(crops_train, labels_train, "train", PROCESSED_DATA_DIRNAME)
        save_images_and_labels(crops_val, labels_val, "val", PROCESSED_DATA_DIRNAME)
        save_images_and_labels(crops_test, labels_test, "test", PROCESSED_DATA_DIRNAME)
        with open(PROCESSED_DATA_DIRNAME / "_max_aspect_ratio.txt", "w") as file:
            file.write(str(aspect_ratios.max()))

    def setup(self, stage: str = None) -> None:
        with open(PROCESSED_DATA_DIRNAME / "_max_aspect_ratio.txt") as file:
            max_aspect_ratio = float(file.read())
            image_width = int(metadata.IMAGE_HEIGHT * max_aspect_ratio)
            assert image_width <= metadata.IMAGE_WIDTH

        if stage == "fit" or stage is None:
            x_train, labels_train = load_processed_crops_and_labels("train", PROCESSED_DATA_DIRNAME)
            y_train = convert_strings_to_labels(labels_train, self.inverse_mapping, length=self.output_dims[0])
            self.data_train = BaseDataset(x_train, y_train, transform=self.trainval_transform)

            x_val, labels_val = load_processed_crops_and_labels("val", PROCESSED_DATA_DIRNAME)
            y_val = convert_strings_to_labels(labels_val, self.inverse_mapping, length=self.output_dims[0])
            self.data_val = BaseDataset(x_val, y_val, transform=self.trainval_transform)

            # quick check: do we have the right sequence lengths?
            assert self.output_dims[0] >= max([len(_) for _ in labels_train]) + 2  # Add 2 for start/end tokens.
            assert self.output_dims[0] >= max([len(_) for _ in labels_val]) + 2  # Add 2 for start/end tokens.

        if stage == "test" or stage is None:
            x_test, labels_test = load_processed_crops_and_labels("test", PROCESSED_DATA_DIRNAME)

            y_test = convert_strings_to_labels(labels_test, self.inverse_mapping, length=self.output_dims[0])
            self.data_test = BaseDataset(x_test, y_test, transform=self.transform)

            assert self.output_dims[0] >= max([len(_) for _ in labels_test]) + 2

    def __repr__(self) -> str:
        """Print info about the dataset."""
        basic = (
            "IAM Lines Dataset\n"
            f"Num classes: {len(self.mapping)}\n"
            f"Dims: {self.input_dims}\n"
            f"Output dims: {self.output_dims}\n"
        )
        if self.data_train is None and self.data_val is None and self.data_test is None:
            return basic

        x, y = next(iter(self.train_dataloader()))
        xt, yt = next(iter(self.test_dataloader()))
        data = (
            f"Train/val/test sizes: {len(self.data_train)}, {len(self.data_val)}, {len(self.data_test)}\n"
            f"Train Batch x stats: {(x.shape, x.dtype, x.min(), x.mean(), x.std(), x.max())}\n"
            f"Train Batch y stats: {(y.shape, y.dtype, y.min(), y.max())}\n"
            f"Test Batch x stats: {(xt.shape, xt.dtype, xt.min(), xt.mean(), xt.std(), xt.max())}\n"
            f"Test Batch y stats: {(yt.shape, yt.dtype, yt.min(), yt.max())}\n"
        )
        return basic + data


def generate_line_crops_and_labels(iam: IAM, split: str, scale_factor=IMAGE_SCALE_FACTOR):
    """Create both cropped lines and associated labels from IAM, with resizing by default"""
    crops, labels = [], []
    for iam_id in iam.ids_by_split[split]:
        labels += iam.line_strings_by_id[iam_id]

        image = iam.load_image(iam_id)
        for line in iam.line_regions_by_id[iam_id]:
            coords = [line[point] for point in ["x1", "y1", "x2", "y2"]]
            crop = image.crop(coords)
            crop = resize_image(crop, scale_factor=scale_factor)
            crops.append(crop)

    assert len(crops) == len(labels)
    return crops, labels


def save_images_and_labels(crops: Sequence[Image.Image], labels: Sequence[str], split: str, data_dirname: Path):
    (data_dirname / split).mkdir(parents=True, exist_ok=True)

    with open(data_dirname / split / "_labels.json", "w") as f:
        json.dump(labels, f)
    for ind, crop in enumerate(crops):
        crop.save(data_dirname / split / f"{ind}.png")


def load_processed_crops_and_labels(split: str, data_dirname: Path):
    """Load line crops and labels for given split from processed directory."""
    crops = load_processed_line_crops(split, data_dirname)
    labels = load_processed_line_labels(split, data_dirname)
    assert len(crops) == len(labels)
    return crops, labels


def load_processed_line_crops(split: str, data_dirname: Path):
    """Load line crops for given split from processed directory."""
    crop_filenames = sorted((data_dirname / split).glob("*.png"), key=lambda filename: int(Path(filename).stem))
    crops = [util.read_image_pil(filename, grayscale=True) for filename in crop_filenames]
    return crops


def load_processed_line_labels(split: str, data_dirname: Path):
    """Load line labels for given split from processed directory."""
    with open(data_dirname / split / "_labels.json") as file:
        labels = json.load(file)
    return labels


if __name__ == "__main__":
    load_and_print_info(IAMLines)
