import torch
from torchvision import transforms

from text_recognizer.stems.base import ImageStem


class MNISTStem(ImageStem):
    """A stem for handling images from the MNIST dataset."""

    def __init__(self):
        super().__init__()
        self.torch_transforms = torch.nn.Sequential(transforms.Normalize((0.1307,), (0.3081,)))
