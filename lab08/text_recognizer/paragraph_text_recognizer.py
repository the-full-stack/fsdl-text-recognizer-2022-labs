"""Detects a paragraph of text in an input image.

Example usage as a script:

  python text_recognizer/paragraph_text_recognizer.py \
    text_recognizer/tests/support/paragraphs/a01-077.png

  python text_recognizer/paragraph_text_recognizer.py \
    https://fsdl-public-assets.s3-us-west-2.amazonaws.com/paragraphs/a01-077.png
"""
import argparse
from pathlib import Path
from typing import Sequence, Union

from PIL import Image
import torch

from text_recognizer import util
from text_recognizer.stems.paragraph import ParagraphStem


STAGED_MODEL_DIRNAME = Path(__file__).resolve().parent / "artifacts" / "paragraph-text-recognizer"
MODEL_FILE = "model.pt"


class ParagraphTextRecognizer:
    """Recognizes a paragraph of text in an image."""

    def __init__(self, model_path=None):
        if model_path is None:
            model_path = STAGED_MODEL_DIRNAME / MODEL_FILE
        self.model = torch.jit.load(model_path)
        self.mapping = self.model.mapping
        self.ignore_tokens = self.model.ignore_tokens
        self.stem = ParagraphStem()

    @torch.no_grad()
    def predict(self, image: Union[str, Path, Image.Image]) -> str:
        """Predict/infer text in input image (which can be a file path or url)."""
        image_pil = image
        if not isinstance(image, Image.Image):
            image_pil = util.read_image_pil(image, grayscale=True)

        image_tensor = self.stem(image_pil).unsqueeze(axis=0)
        y_pred = self.model(image_tensor)[0]
        pred_str = convert_y_label_to_string(y=y_pred, mapping=self.mapping, ignore_tokens=self.ignore_tokens)

        return pred_str


def convert_y_label_to_string(y: torch.Tensor, mapping: Sequence[str], ignore_tokens: Sequence[int]) -> str:
    return "".join([mapping[i] for i in y if i not in ignore_tokens])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "filename",
        type=str,
        help="Name for an image file. This can be a local path, a URL, a URI from AWS/GCP/Azure storage, an HDFS path, or any other resource locator supported by the smart_open library.",
    )
    args = parser.parse_args()

    text_recognizer = ParagraphTextRecognizer()
    pred_str = text_recognizer.predict(args.filename)
    print(pred_str)


if __name__ == "__main__":
    main()
