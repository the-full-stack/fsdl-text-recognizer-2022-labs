"""Class for loading the IAM handwritten text dataset, which encompasses both paragraphs and lines, plus utilities."""
from pathlib import Path
from typing import Any, cast, Dict, List, Optional
import zipfile

from boltons.cacheutils import cachedproperty
from defusedxml import ElementTree
from PIL import Image, ImageOps
import toml

from text_recognizer import util
from text_recognizer.data.base_data_module import _download_raw_dataset, load_and_print_info
import text_recognizer.metadata.iam as metadata
from text_recognizer.metadata.iam_paragraphs import NEW_LINE_TOKEN


METADATA_FILENAME = metadata.METADATA_FILENAME
DL_DATA_DIRNAME = metadata.DL_DATA_DIRNAME
EXTRACTED_DATASET_DIRNAME = metadata.EXTRACTED_DATASET_DIRNAME


class IAM:
    """A dataset of images of handwritten text written on a form underneath a typewritten prompt.

    "The IAM Lines dataset, first published at the ICDAR 1999, contains forms of unconstrained handwritten text,
    which were scanned at a resolution of 300dpi and saved as PNG images with 256 gray levels."
    From http://www.fki.inf.unibe.ch/databases/iam-handwriting-database

    Images are identified by their "form ID". These IDs are used to separate train, validation and test splits,
    as keys for dictonaries returning label and image crop region data, and more.

    The data split we will use is
    IAM lines Large Writer Independent Text Line Recognition Task (LWITLRT): 9,862 text lines.
        The validation set has been merged into the train set.
        The train set has 7,101 lines from 326 writers.
        The test set has 1,861 lines from 128 writers.
        The text lines of all data sets are mutually exclusive, thus each writer has contributed to one set only.
    """

    def __init__(self):
        self.metadata = toml.load(METADATA_FILENAME)

    def prepare_data(self):
        if self.xml_filenames:
            return
        filename = _download_raw_dataset(self.metadata, DL_DATA_DIRNAME)  # type: ignore
        _extract_raw_dataset(filename, DL_DATA_DIRNAME)

    def load_image(self, id: str) -> Image.Image:
        """Load and return an image of an entire IAM form.

        The image is grayscale with white text on black background.

        This image will have the printed prompt text at the top, above the handwritten text.
        Images of individual words or lines and of whole paragraphs can be cropped out using the
        relevant crop region data.
        """
        image = util.read_image_pil(self.form_filenames_by_id[id], grayscale=True)
        image = ImageOps.invert(image)
        return image

    def __repr__(self):
        """Print info about the dataset."""
        info = ["IAM Dataset"]
        info.append(f"Total Images: {len(self.xml_filenames)}")
        info.append(f"Total Test Images: {len(self.test_ids)}")
        info.append(f"Total Paragraphs: {len(self.paragraph_string_by_id)}")
        num_lines = sum(len(line_regions) for line_regions in self.line_regions_by_id.items())
        info.append(f"Total Lines: {num_lines}")

        return "\n\t".join(info)

    @cachedproperty
    def all_ids(self):
        """A list of all form IDs."""
        return sorted([f.stem for f in self.xml_filenames])

    @cachedproperty
    def ids_by_split(self):
        return {"train": self.train_ids, "val": self.validation_ids, "test": self.test_ids}

    @cachedproperty
    def split_by_id(self):
        """A dictionary mapping form IDs to their split according to IAM Lines LWITLRT."""
        split_by_id = {id_: "train" for id_ in self.train_ids}
        split_by_id.update({id_: "val" for id_ in self.validation_ids})
        split_by_id.update({id_: "test" for id_ in self.test_ids})
        return split_by_id

    @cachedproperty
    def train_ids(self):
        """A list of form IDs which are in the IAM Lines LWITLRT training set."""
        return list(set(self.all_ids) - (set(self.test_ids) | set(self.validation_ids)))

    @cachedproperty
    def test_ids(self):
        """A list of form IDs from the IAM Lines LWITLRT test set."""
        return _get_ids_from_lwitlrt_split_file(EXTRACTED_DATASET_DIRNAME / "task/testset.txt")

    @property
    def xml_filenames(self) -> List[Path]:
        """A list of the filenames of all .xml files, which contain label information."""
        return list((EXTRACTED_DATASET_DIRNAME / "xml").glob("*.xml"))

    @cachedproperty
    def validation_ids(self):
        """A list of form IDs from IAM Lines LWITLRT validation sets 1 and 2."""
        val_ids = _get_ids_from_lwitlrt_split_file(EXTRACTED_DATASET_DIRNAME / "task/validationset1.txt")
        val_ids.extend(_get_ids_from_lwitlrt_split_file(EXTRACTED_DATASET_DIRNAME / "task/validationset2.txt"))
        return val_ids

    @property
    def form_filenames(self) -> List[Path]:
        """A list of the filenames of all .jpg files, which contain images of IAM forms."""
        return list((EXTRACTED_DATASET_DIRNAME / "forms").glob("*.jpg"))

    @property
    def xml_filenames_by_id(self):
        """A dictionary mapping form IDs to their XML label information files."""
        return {filename.stem: filename for filename in self.xml_filenames}

    @property
    def form_filenames_by_id(self):
        """A dictionary mapping form IDs to their JPEG images."""
        return {filename.stem: filename for filename in self.form_filenames}

    @cachedproperty
    def line_strings_by_id(self):
        """A dict mapping an IAM form id to its list of line texts."""
        return {filename.stem: _get_line_strings_from_xml_file(filename) for filename in self.xml_filenames}

    @cachedproperty
    def line_regions_by_id(self):
        """A dict mapping an IAM form id to its list of line image crop regions."""
        return {filename.stem: _get_line_regions_from_xml_file(filename) for filename in self.xml_filenames}

    @cachedproperty
    def paragraph_string_by_id(self):
        """A dict mapping an IAM form id to its paragraph text."""
        return {id: NEW_LINE_TOKEN.join(line_strings) for id, line_strings in self.line_strings_by_id.items()}

    @cachedproperty
    def paragraph_region_by_id(self):
        """A dict mapping an IAM form id to its paragraph image crop region."""
        return {
            id: {
                "x1": min(region["x1"] for region in line_regions),
                "y1": min(region["y1"] for region in line_regions),
                "x2": max(region["x2"] for region in line_regions),
                "y2": max(region["y2"] for region in line_regions),
            }
            for id, line_regions in self.line_regions_by_id.items()
        }


def _extract_raw_dataset(filename: Path, dirname: Path) -> None:
    print("Extracting IAM data")
    with util.temporary_working_directory(dirname):
        with zipfile.ZipFile(filename, "r") as zip_file:
            zip_file.extractall()


def _get_ids_from_lwitlrt_split_file(filename: str) -> List[str]:
    """Get the ids from Large Writer Independent Text Line Recognition Task (LWITLRT) data split file."""
    with open(filename, "r") as f:
        line_ids_str = f.read()
    line_ids = line_ids_str.split("\n")
    page_ids = list({"-".join(line_id.split("-")[:2]) for line_id in line_ids if line_id})
    return page_ids


def _get_line_strings_from_xml_file(filename: str) -> List[str]:
    """Get the text content of each line. Note that we replace &quot; with "."""
    xml_line_elements = _get_line_elements_from_xml_file(filename)
    return [_get_text_from_xml_element(el) for el in xml_line_elements]


def _get_text_from_xml_element(xml_element: Any) -> str:
    """Extract text from any XML element."""
    return xml_element.attrib["text"].replace("&quot;", '"')


def _get_line_regions_from_xml_file(filename: str) -> List[Dict[str, int]]:
    """Get the line region dict for each line."""
    xml_line_elements = _get_line_elements_from_xml_file(filename)
    line_regions = [
        cast(Dict[str, int], _get_region_from_xml_element(xml_elem=el, xml_path="word/cmp")) for el in xml_line_elements
    ]
    assert any(region is not None for region in line_regions), "Line regions cannot be None"

    # next_line_region["y1"] - prev_line_region["y2"] can be negative due to overlapping characters
    line_gaps_y = [
        max(next_line_region["y1"] - prev_line_region["y2"], 0)
        for next_line_region, prev_line_region in zip(line_regions[1:], line_regions[:-1])
    ]
    post_line_gaps_y = line_gaps_y + [2 * metadata.LINE_REGION_PADDING]
    pre_line_gaps_y = [2 * metadata.LINE_REGION_PADDING] + line_gaps_y

    return [
        {
            "x1": region["x1"] - metadata.LINE_REGION_PADDING,
            "x2": region["x2"] + metadata.LINE_REGION_PADDING,
            "y1": region["y1"] - min(metadata.LINE_REGION_PADDING, pre_line_gaps_y[i] // 2),
            "y2": region["y2"] + min(metadata.LINE_REGION_PADDING, post_line_gaps_y[i] // 2),
        }
        for i, region in enumerate(line_regions)
    ]


def _get_line_elements_from_xml_file(filename: str) -> List[Any]:
    """Get all line xml elements from xml file."""
    xml_root_element = ElementTree.parse(filename).getroot()  # nosec
    return xml_root_element.findall("handwritten-part/line")


def _get_region_from_xml_element(xml_elem: Any, xml_path: str) -> Optional[Dict[str, int]]:
    """
    Get region from input xml element. The region is downsampled because the stored images are also downsampled.

    Parameters
    ----------
    xml_elem
        xml element can be a line or word element with x, y, width, and height attributes
    xml_path
        should be "word/cmp" if xml_elem is a line element, else "cmp"
    """
    unit_elements = xml_elem.findall(xml_path)
    if not unit_elements:
        return None
    return {
        "x1": min(int(el.attrib["x"]) for el in unit_elements) // metadata.DOWNSAMPLE_FACTOR,
        "y1": min(int(el.attrib["y"]) for el in unit_elements) // metadata.DOWNSAMPLE_FACTOR,
        "x2": max(int(el.attrib["x"]) + int(el.attrib["width"]) for el in unit_elements) // metadata.DOWNSAMPLE_FACTOR,
        "y2": max(int(el.attrib["y"]) + int(el.attrib["height"]) for el in unit_elements) // metadata.DOWNSAMPLE_FACTOR,
    }


if __name__ == "__main__":
    load_and_print_info(IAM)
