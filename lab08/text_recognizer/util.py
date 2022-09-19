"""Utility functions for text_recognizer module."""
import base64
import contextlib
import hashlib
from io import BytesIO
import os
from pathlib import Path
from typing import Union
from urllib.request import urlretrieve

import numpy as np
from PIL import Image
import smart_open
from tqdm import tqdm


def to_categorical(y, num_classes):
    """1-hot encode a tensor."""
    return np.eye(num_classes, dtype="uint8")[y]


def read_image_pil(image_uri: Union[Path, str], grayscale=False) -> Image:
    with smart_open.open(image_uri, "rb") as image_file:
        return read_image_pil_file(image_file, grayscale)


def read_image_pil_file(image_file, grayscale=False) -> Image:
    with Image.open(image_file) as image:
        if grayscale:
            image = image.convert(mode="L")
        else:
            image = image.convert(mode=image.mode)
        return image


@contextlib.contextmanager
def temporary_working_directory(working_dir: Union[str, Path]):
    """Temporarily switches to a directory, then returns to the original directory on exit."""
    curdir = os.getcwd()
    os.chdir(working_dir)
    try:
        yield
    finally:
        os.chdir(curdir)


def read_b64_image(b64_string, grayscale=False):
    """Load base64-encoded images."""
    try:
        image_file = read_b64_string(b64_string)
        return read_image_pil_file(image_file, grayscale)
    except Exception as exception:
        raise ValueError("Could not load image from b64 {}: {}".format(b64_string, exception)) from exception


def read_b64_string(b64_string, return_data_type=False):
    """Read a base64-encoded string into an in-memory file-like object."""
    data_header, b64_data = split_and_validate_b64_string(b64_string)
    b64_buffer = BytesIO(base64.b64decode(b64_data))
    if return_data_type:
        return get_b64_filetype(data_header), b64_buffer
    else:
        return b64_buffer


def get_b64_filetype(data_header):
    """Retrieves the filetype information from the data type header of a base64-encoded object."""
    _, file_type = data_header.split("/")
    return file_type


def split_and_validate_b64_string(b64_string):
    """Return the data_type and data of a b64 string, with validation."""
    header, data = b64_string.split(",", 1)
    assert header.startswith("data:")
    assert header.endswith(";base64")
    data_type = header.split(";")[0].split(":")[1]
    return data_type, data


def encode_b64_image(image, format="png"):
    """Encode a PIL image as a base64 string."""
    _buffer = BytesIO()  # bytes that live in memory
    image.save(_buffer, format=format)  # but which we write to like a file
    encoded_image = base64.b64encode(_buffer.getvalue()).decode("utf8")
    return encoded_image


def compute_sha256(filename: Union[Path, str]):
    """Return SHA256 checksum of a file."""
    with open(filename, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class TqdmUpTo(tqdm):
    """From https://github.com/tqdm/tqdm/blob/master/examples/tqdm_wget.py"""

    def update_to(self, blocks=1, bsize=1, tsize=None):
        """
        Parameters
        ----------
        blocks: int, optional
            Number of blocks transferred so far [default: 1].
        bsize: int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(blocks * bsize - self.n)  # will also set self.n = b * bsize


def download_url(url, filename):
    """Download a file from url to filename, with a progress bar."""
    with TqdmUpTo(unit="B", unit_scale=True, unit_divisor=1024, miniters=1) as t:
        urlretrieve(url, filename, reporthook=t.update_to, data=None)  # noqa: S310
