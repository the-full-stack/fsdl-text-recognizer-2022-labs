import random

from PIL import Image
from torchvision import transforms


import text_recognizer.metadata.iam_lines as metadata
from text_recognizer.stems.image import ImageStem


class LineStem(ImageStem):
    """A stem for handling images containing a line of text."""

    def __init__(self, augment=False, color_jitter_kwargs=None, random_affine_kwargs=None):
        super().__init__()
        if color_jitter_kwargs is None:
            color_jitter_kwargs = {"brightness": (0.5, 1)}
        if random_affine_kwargs is None:
            random_affine_kwargs = {
                "degrees": 3,
                "translate": (0, 0.05),
                "scale": (0.4, 1.1),
                "shear": (-40, 50),
                "interpolation": transforms.InterpolationMode.BILINEAR,
                "fill": 0,
            }

        if augment:
            self.pil_transforms = transforms.Compose(
                [
                    transforms.ColorJitter(**color_jitter_kwargs),
                    transforms.RandomAffine(**random_affine_kwargs),
                ]
            )


class IAMLineStem(ImageStem):
    """A stem for handling images containing lines of text from the IAMLines dataset."""

    def __init__(self, augment=False, color_jitter_kwargs=None, random_affine_kwargs=None):
        super().__init__()

        def embed_crop(crop, augment=augment):
            # crop is PIL.image of dtype="L" (so values range from 0 -> 255)
            image = Image.new("L", (metadata.IMAGE_WIDTH, metadata.IMAGE_HEIGHT))

            # Resize crop
            crop_width, crop_height = crop.size
            new_crop_height = metadata.IMAGE_HEIGHT
            new_crop_width = int(new_crop_height * (crop_width / crop_height))
            if augment:
                # Add random stretching
                new_crop_width = int(new_crop_width * random.uniform(0.9, 1.1))
                new_crop_width = min(new_crop_width, metadata.IMAGE_WIDTH)
            crop_resized = crop.resize((new_crop_width, new_crop_height), resample=Image.BILINEAR)

            # Embed in the image
            x = min(metadata.CHAR_WIDTH, metadata.IMAGE_WIDTH - new_crop_width)
            y = metadata.IMAGE_HEIGHT - new_crop_height

            image.paste(crop_resized, (x, y))

            return image

        if color_jitter_kwargs is None:
            color_jitter_kwargs = {"brightness": (0.8, 1.6)}
        if random_affine_kwargs is None:
            random_affine_kwargs = {
                "degrees": 1,
                "shear": (-30, 20),
                "interpolation": transforms.InterpolationMode.BILINEAR,
                "fill": 0,
            }

        pil_transforms_list = [transforms.Lambda(embed_crop)]
        if augment:
            pil_transforms_list += [
                transforms.ColorJitter(**color_jitter_kwargs),
                transforms.RandomAffine(**random_affine_kwargs),
            ]
        self.pil_transforms = transforms.Compose(pil_transforms_list)
