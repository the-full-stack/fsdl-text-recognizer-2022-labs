import json
import os

import requests

from app_gradio import app
from text_recognizer import util


os.environ["CUDA_VISIBLE_DEVICES"] = ""


TEST_IMAGE = "text_recognizer/tests/support/paragraphs/a01-077.png"


def test_local_run():
    """A quick test to make sure we can build the app and ping the API locally."""
    backend = app.PredictorBackend()
    frontend = app.make_frontend(fn=backend.run)

    # run the UI without blocking
    frontend.launch(share=False, prevent_thread_lock=True)
    local_url = frontend.local_url
    get_response = requests.get(local_url)
    assert get_response.status_code == 200

    image_b64 = util.encode_b64_image(util.read_image_pil(TEST_IMAGE))

    local_api = f"{local_url}api/predict"
    headers = {"Content-Type": "application/json"}
    payload = json.dumps({"data": ["data:image/png;base64," + image_b64]})
    post_response = requests.post(local_api, data=payload, headers=headers)
    assert "error" not in post_response.json()
    assert "data" in post_response.json()
