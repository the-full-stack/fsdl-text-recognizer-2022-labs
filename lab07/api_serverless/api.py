"""AWS Lambda function serving text_recognizer predictions."""
from __future__ import annotations

import json

from PIL import ImageStat

import text_recognizer.util as util
from text_recognizer.paragraph_text_recognizer import ParagraphTextRecognizer

model = ParagraphTextRecognizer()


def handler(event, _context):
    """Provide main prediction API."""
    print("INFO loading image")
    image = _load_image(event)
    if image is None:
        return {"statusCode": 400, "message": "neither image_url nor image found in event"}
    print("INFO image loaded")
    print("INFO starting inference")
    pred = model.predict(image)
    print("INFO inference complete")
    image_stat = ImageStat.Stat(image)
    print(f"METRIC image_mean_intensity {image_stat.mean[0]}")
    print(f"METRIC image_area {image.size[0] * image.size[1]}")
    print(f"METRIC pred_length {len(pred)}")
    print(f"INFO pred {pred}")
    return {"pred": str(pred)}


def _load_image(event):
    event = _from_string(event)
    event = _from_string(event.get("body", event))
    image_url = event.get("image_url")
    if image_url is not None:
        print(f"INFO url {image_url}")
        return util.read_image_pil(image_url, grayscale=True)
    else:
        image = event.get("image")
        if image is not None:
            print("INFO reading image from event")
            return util.read_b64_image(image, grayscale=True)
        else:
            return None


def _from_string(event):
    if isinstance(event, str):
        return json.loads(event)
    else:
        return event
