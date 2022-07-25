"""Class for loading the IAM dataset, which encompasses both paragraphs and lines, with associated utilities."""
import argparse
import os
from pathlib import Path
from typing import Dict, List
import zipfile

from boltons.cacheutils import cachedproperty
from defusedxml import ElementTree
import toml

from text_recognizer.data.base_data_module import _download_raw_dataset, BaseDataModule, load_and_print_info
import text_recognizer.metadata.iam as metadata


METADATA_FILENAME = metadata.METADATA_FILENAME
DL_DATA_DIRNAME = metadata.DL_DATA_DIRNAME
EXTRACTED_DATASET_DIRNAME = metadata.EXTRACTED_DATASET_DIRNAME


class IAM(BaseDataModule):
    """
    "The IAM Lines dataset, first published at the ICDAR 1999, contains forms of unconstrained handwritten text,
    which were scanned at a resolution of 300dpi and saved as PNG images with 256 gray levels.
    From http://www.fki.inf.unibe.ch/databases/iam-handwriting-database

    The data split we will use is
    IAM lines Large Writer Independent Text Line Recognition Task (lwitlrt): 9,862 text lines.
        The validation set has been merged into the train set.
        The train set has 7,101 lines from 326 writers.
        The test set has 1,861 lines from 128 writers.
        The text lines of all data sets are mutually exclusive, thus each writer has contributed to one set only.
    """

    def __init__(self, args: argparse.Namespace = None):
        super().__init__(args)
        self.metadata = toml.load(METADATA_FILENAME)

    def prepare_data(self, *args, **kwargs) -> None:
        if self.xml_filenames:
            return
        filename = _download_raw_dataset(self.metadata, DL_DATA_DIRNAME)  # type: ignore
        _extract_raw_dataset(filename, DL_DATA_DIRNAME)

    @property
    def xml_filenames(self) -> List[Path]:
        return list((EXTRACTED_DATASET_DIRNAME / "xml").glob("*.xml"))

    @property
    def form_filenames(self) -> List[Path]:
        return list((EXTRACTED_DATASET_DIRNAME / "forms").glob("*.jpg"))

    @property
    def form_filenames_by_id(self):
        return {filename.stem: filename for filename in self.form_filenames}

    @cachedproperty
    def all_ids(self):
        """Return a list of all ids."""
        return sorted([f.stem for f in self.xml_filenames])

    @cachedproperty
    def test_ids(self):
        """Return a list of IAM lines Large Writer Independent Text Line Recognition Task test ids."""
        return _get_ids_from_lwitlrt_split_file(EXTRACTED_DATASET_DIRNAME / "task/testset.txt")

    @cachedproperty
    def validation_ids(self):
        """Return a list of IAM lines Large Writer Independent Text Line Recognition Task validation (set 1 and set 2) ids."""
        val_ids = _get_ids_from_lwitlrt_split_file(EXTRACTED_DATASET_DIRNAME / "task/validationset1.txt")
        val_ids.extend(_get_ids_from_lwitlrt_split_file(EXTRACTED_DATASET_DIRNAME / "task/validationset2.txt"))
        return val_ids

    @cachedproperty
    def train_ids(self):
        """Return a list of train ids - all ids which aren't test or validation ids."""
        return list(set(self.all_ids) - (set(self.test_ids) | set(self.validation_ids)))

    @cachedproperty
    def split_by_id(self):
        split_by_id = {id_: "train" for id_ in self.train_ids}
        split_by_id.update({id_: "val" for id_ in self.validation_ids})
        split_by_id.update({id_: "test" for id_ in self.test_ids})
        return split_by_id

    @cachedproperty
    def line_strings_by_id(self):
        """Return a dict from name of IAM form to a list of line texts in it."""
        return {filename.stem: _get_line_strings_from_xml_file(filename) for filename in self.xml_filenames}

    @cachedproperty
    def line_regions_by_id(self):
        """Return a dict from name of IAM form to a list of (x1, x2, y1, y2) coordinates of all lines in it."""
        return {filename.stem: _get_line_regions_from_xml_file(filename) for filename in self.xml_filenames}

    def __repr__(self):
        """Print info about the dataset."""
        return "IAM Dataset\n" f"Num total: {len(self.xml_filenames)}\nNum test: {len(self.metadata['test_ids'])}\n"


def _extract_raw_dataset(filename: Path, dirname: Path) -> None:
    print("Extracting IAM data")
    curdir = os.getcwd()
    os.chdir(dirname)
    with zipfile.ZipFile(filename, "r") as zip_file:
        zip_file.extractall()
    os.chdir(curdir)


def _get_ids_from_lwitlrt_split_file(filename: str) -> List[str]:
    """Get the ids from Large Writer Independent Text Line Recognition Task (LWITLRT) data split file."""
    with open(filename, "r") as f:
        line_ids_str = f.read()
    line_ids = line_ids_str.split("\n")
    page_ids = list({"-".join(line_id.split("-")[:2]) for line_id in line_ids if line_id})
    return page_ids


def _get_line_strings_from_xml_file(filename: str) -> List[str]:
    """Get the text content of each line. Note that we replace &quot; with "."""
    xml_root_element = ElementTree.parse(filename).getroot()  # nosec
    xml_line_elements = xml_root_element.findall("handwritten-part/line")
    return [el.attrib["text"].replace("&quot;", '"') for el in xml_line_elements]


def _get_line_regions_from_xml_file(filename: str) -> List[Dict[str, int]]:
    """Get the line region dict for each line."""
    xml_root_element = ElementTree.parse(filename).getroot()  # nosec
    xml_line_elements = xml_root_element.findall("handwritten-part/line")
    return [_get_line_region_from_xml_element(el) for el in xml_line_elements]


def _get_line_region_from_xml_element(xml_line) -> Dict[str, int]:
    """
    Parameters
    ----------
    xml_line
        xml element that has x, y, width, and height attributes
    """
    word_elements = xml_line.findall("word/cmp")
    x1s = [int(el.attrib["x"]) for el in word_elements]
    y1s = [int(el.attrib["y"]) for el in word_elements]
    x2s = [int(el.attrib["x"]) + int(el.attrib["width"]) for el in word_elements]
    y2s = [int(el.attrib["y"]) + int(el.attrib["height"]) for el in word_elements]
    return {
        "x1": min(x1s) // metadata.DOWNSAMPLE_FACTOR - metadata.LINE_REGION_PADDING,
        "y1": min(y1s) // metadata.DOWNSAMPLE_FACTOR - metadata.LINE_REGION_PADDING,
        "x2": max(x2s) // metadata.DOWNSAMPLE_FACTOR + metadata.LINE_REGION_PADDING,
        "y2": max(y2s) // metadata.DOWNSAMPLE_FACTOR + metadata.LINE_REGION_PADDING,
    }


if __name__ == "__main__":
    load_and_print_info(IAM)
