"""IAMParagraphs Stem class."""
from PIL import Image
import torchvision.transforms as transforms

import text_recognizer.metadata.iam_paragraphs as metadata
from text_recognizer.stems.image import ImageStem


IMAGE_SCALE_FACTOR = metadata.IMAGE_SCALE_FACTOR
IMAGE_HEIGHT, IMAGE_WIDTH = metadata.IMAGE_HEIGHT, metadata.IMAGE_WIDTH
IMAGE_SHAPE = metadata.IMAGE_SHAPE

MAX_LABEL_LENGTH = metadata.MAX_LABEL_LENGTH


class ParagraphStem(ImageStem):
    """A stem for handling images that contain a paragraph of text."""

    scale_factor = IMAGE_SCALE_FACTOR

    def __init__(self, augment=False, color_jitter_kwargs=None, random_affine_kwargs=None):
        super().__init__()

        if not augment:
            self.pil_transforms = transforms.Compose([transforms.CenterCrop(IMAGE_SHAPE)])
        else:
            if color_jitter_kwargs is None:
                color_jitter_kwargs = {"brightness": (0.8, 1.6)}
            if random_affine_kwargs is None:
                random_affine_kwargs = {
                    "degrees": 1,
                    "shear": (-10, 10),
                    "interpolation": transforms.InterpolationMode.BILINEAR,
                }

            self.pil_transforms = transforms.Compose(
                [
                    transforms.RandomCrop(
                        size=IMAGE_SHAPE, padding=None, pad_if_needed=True, fill=0, padding_mode="constant"
                    ),
                    transforms.ColorJitter(**color_jitter_kwargs),
                    transforms.RandomAffine(**random_affine_kwargs),
                ]
            )

    def __call__(self, img):
        img = self.resize(img)
        return super().__call__(img)

    def resize(self, img):
        if self.scale_factor == 1:
            return img
        else:
            out_shape = (img.width // self.scale_factor, img.height // self.scale_factor)
            return img.resize(out_shape, resample=Image.BILINEAR)
