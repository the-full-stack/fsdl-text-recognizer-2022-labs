"""Simplest version of LineCNN that works on cleanly-separated characters."""
import argparse
import math
from typing import Any, Dict

import torch
from torch import nn

from .cnn import CNN

IMAGE_SIZE = 28
WINDOW_WIDTH = IMAGE_SIZE
WINDOW_STRIDE = IMAGE_SIZE


class LineCNNSimple(nn.Module):
    """LeNet based model that takes a line of width that is a multiple of CHAR_WIDTH."""

    def __init__(
        self,
        data_config: Dict[str, Any],
        args: argparse.Namespace = None,
    ) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}
        self.data_config = data_config

        self.WW = self.args.get("window_width", WINDOW_WIDTH)
        self.WS = self.args.get("window_stride", WINDOW_STRIDE)
        self.limit_output_length = self.args.get("limit_output_length", False)

        self.num_classes = len(data_config["mapping"])
        self.output_length = data_config["output_dims"][0]
        cnn_input_dims = (data_config["input_dims"][0], self.WW, self.WW)
        cnn_data_config = {**data_config, **{"input_dims": cnn_input_dims}}
        self.cnn = CNN(data_config=cnn_data_config, args=args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the LineCNN to an input image and return logits.

        Parameters
        ----------
        x
            (B, C, H, W) input image with H equal to IMAGE_SIZE

        Returns
        -------
        torch.Tensor
            (B, C, S) logits, where S is the length of the sequence and C is the number of classes
            S can be computed from W and CHAR_WIDTH
            C is self.num_classes
        """
        B, _C, H, W = x.shape
        assert H == IMAGE_SIZE  # Make sure we can use our CNN class

        # Compute number of windows
        S = math.floor((W - self.WW) / self.WS + 1)

        # NOTE: type_as properly sets device
        activations = torch.zeros((B, self.num_classes, S)).type_as(x)
        for s in range(S):
            start_w = self.WS * s
            end_w = start_w + self.WW
            window = x[:, :, :, start_w:end_w]  # -> (B, C, H, self.WW)
            activations[:, :, s] = self.cnn(window)

        if self.limit_output_length:
            # S might not match ground truth, so let's only take enough activations as are expected
            activations = activations[:, :, : self.output_length]
        return activations

    @staticmethod
    def add_to_argparse(parser):
        CNN.add_to_argparse(parser)
        parser.add_argument(
            "--window_width",
            type=int,
            default=WINDOW_WIDTH,
            help="Width of the window that will slide over the input image.",
        )
        parser.add_argument(
            "--window_stride",
            type=int,
            default=WINDOW_STRIDE,
            help="Stride of the window that will slide over the input image.",
        )
        parser.add_argument("--limit_output_length", action="store_true", default=False)
        return parser
