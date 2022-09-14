"""Provide an image of handwritten text and get back out a string!"""
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Callable

import warnings

import gradio as gr
from PIL import ImageStat
from PIL.Image import Image
import requests

from app_gradio.flagging import GantryImageToTextLogger, get_api_key
from app_gradio.s3_util import make_unique_bucket_name

from text_recognizer.paragraph_text_recognizer import ParagraphTextRecognizer
import text_recognizer.util as util

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # do not use GPU

logging.basicConfig(level=logging.INFO)
DEFAULT_APPLICATION_NAME = "fsdl-text-recognizer"

APP_DIR = Path(__file__).resolve().parent  # what is the directory for this application?
FAVICON = APP_DIR / "1f95e.png"  # path to a small image for display in browser tab and social media
README = APP_DIR / "README.md"  # path to an app readme file in HTML/markdown

DEFAULT_PORT = 11700


def main(args):
    predictor = PredictorBackend(url=args.model_url)
    frontend = make_frontend(predictor.run, flagging=args.flagging, gantry=args.gantry, app_name=args.application)
    frontend.launch(
        server_name="0.0.0.0",  # make server accessible, binding all interfaces  # noqa: S104
        server_port=args.port,  # set a port to bind to, failing if unavailable
        share=True,  # should we create a (temporary) public link on https://gradio.app?
        favicon_path=FAVICON,  # what icon should we display in the address bar?
    )


def make_frontend(
    fn: Callable[[Image], str], flagging: bool = False, gantry: bool = False, app_name: str = "fsdl-text-recognizer"
):
    """Creates a gradio.Interface frontend for an image to text function."""
    examples_dir = Path("text_recognizer") / "tests" / "support" / "paragraphs"
    example_fnames = [elem for elem in os.listdir(examples_dir) if elem.endswith(".png")]
    example_paths = [examples_dir / fname for fname in example_fnames]

    examples = [[str(path)] for path in example_paths]

    allow_flagging = "never"
    if flagging:
        allow_flagging = "manual"
        api_key = get_api_key()
        if gantry and api_key:  # if we're logging user feedback to Gantry and we have an API key
            allow_flagging = "manual"  # turn on Gradio flagging features
            # callback for logging input images, output text, and feedback to Gantry
            flagging_callback = GantryImageToTextLogger(application=app_name, api_key=api_key)
            # that sends images to S3
            flagging_dir = make_unique_bucket_name(prefix=app_name, seed=api_key)
        else:  # otherwise, log to a local CSV file
            if gantry and api_key is None:
                warnings.warn("No Gantry API key found, logging to local directory instead.")
            flagging_callback = gr.CSVLogger()
            flagging_dir = "flagged"
    else:
        flagging_callback, flagging_dir = None, None

    readme = _load_readme(with_logging=allow_flagging == "manual")

    # build a basic browser interface to a Python function
    frontend = gr.Interface(
        fn=fn,  # which Python function are we interacting with?
        outputs=gr.components.Textbox(),  # what output widgets does it need? the default text widget
        # what input widgets does it need? we configure an image widget
        inputs=gr.components.Image(type="pil", label="Handwritten Text"),
        title="📝 Text Recognizer",  # what should we display at the top of the page?
        thumbnail=FAVICON,  # what should we display when the link is shared, e.g. on social media?
        description=__doc__,  # what should we display just above the interface?
        article=readme,  # what long-form content should we display below the interface?
        examples=examples,  # which potential inputs should we provide?
        cache_examples=False,  # should we cache those inputs for faster inference? slows down start
        allow_flagging=allow_flagging,  # should we show users the option to "flag" outputs?
        flagging_options=["incorrect", "offensive", "other"],  # what options do users have for feedback?
        flagging_callback=flagging_callback,
        flagging_dir=flagging_dir,
    )

    return frontend


class PredictorBackend:
    """Interface to a backend that serves predictions.

    To communicate with a backend accessible via a URL, provide the url kwarg.

    Otherwise, runs a predictor locally.
    """

    def __init__(self, url=None):
        if url is not None:
            self.url = url
            self._predict = self._predict_from_endpoint
        else:
            model = ParagraphTextRecognizer()
            self._predict = model.predict

    def run(self, image):
        pred, metrics = self._predict_with_metrics(image)
        self._log_inference(pred, metrics)
        return pred

    def _predict_with_metrics(self, image):
        pred = self._predict(image)

        stats = ImageStat.Stat(image)
        metrics = {
            "image_mean_intensity": stats.mean,
            "image_median": stats.median,
            "image_extrema": stats.extrema,
            "image_area": image.size[0] * image.size[1],
            "pred_length": len(pred),
        }
        return pred, metrics

    def _predict_from_endpoint(self, image):
        """Send an image to an endpoint that accepts JSON and return the predicted text.

        The endpoint should expect a base64 representation of the image, encoded as a string,
        under the key "image". It should return the predicted text under the key "pred".

        Parameters
        ----------
        image
            A PIL image of handwritten text to be converted into a string.

        Returns
        -------
        pred
            A string containing the predictor's guess of the text in the image.
        """
        encoded_image = util.encode_b64_image(image)

        headers = {"Content-type": "application/json"}
        payload = json.dumps({"image": "data:image/png;base64," + encoded_image})

        response = requests.post(self.url, data=payload, headers=headers)
        pred = response.json()["pred"]

        return pred

    def _log_inference(self, pred, metrics):
        for key, value in metrics.items():
            logging.info(f"METRIC {key} {value}")
        logging.info(f"PRED >begin\n{pred}\nPRED >end")


def _load_readme(with_logging=False):
    with open(README) as f:
        lines = f.readlines()
        if not with_logging:
            lines = lines[: lines.index("<!-- logging content below -->\n")]

        readme = "".join(lines)
    return readme


def _make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_url",
        default=None,
        type=str,
        help="Identifies a URL to which to send image data. Data is base64-encoded, converted to a utf-8 string, and then set via a POST request as JSON with the key 'image'. Default is None, which instead sends the data to a model running locally.",
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        type=int,
        help=f"Port on which to expose this server. Default is {DEFAULT_PORT}.",
    )
    parser.add_argument(
        "--flagging",
        action="store_true",
        help="Pass this flag to allow users to 'flag' model behavior and provide feedback.",
    )
    parser.add_argument(
        "--gantry",
        action="store_true",
        help="Pass --flagging and this flag to log user feedback to Gantry. Requires GANTRY_API_KEY to be defined as an environment variable.",
    )
    parser.add_argument(
        "--application",
        default=DEFAULT_APPLICATION_NAME,
        type=str,
        help=f"Name of the Gantry application to which feedback should be logged, if --gantry and --flagging are passed. Default is {DEFAULT_APPLICATION_NAME}.",
    )

    return parser


if __name__ == "__main__":
    parser = _make_parser()
    args = parser.parse_args()
    main(args)
