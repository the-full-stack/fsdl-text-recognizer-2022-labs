"""Model combining a ResNet with a Transformer for image-to-sequence tasks."""
import argparse
import math
from typing import Any, Dict

import torch
from torch import nn
import torchvision

from .transformer_util import generate_square_subsequent_mask, PositionalEncoding, PositionalEncodingImage


TF_DIM = 256
TF_FC_DIM = 1024
TF_DROPOUT = 0.4
TF_LAYERS = 4
TF_NHEAD = 4
RESNET_DIM = 512  # hard-coded


class ResnetTransformer(nn.Module):
    """Pass an image through a Resnet and decode the resulting embedding with a Transformer."""

    def __init__(
        self,
        data_config: Dict[str, Any],
        args: argparse.Namespace = None,
    ) -> None:
        super().__init__()
        self.data_config = data_config
        self.input_dims = data_config["input_dims"]
        self.num_classes = len(data_config["mapping"])

        self.mapping = data_config["mapping"]
        inverse_mapping = {val: ind for ind, val in enumerate(data_config["mapping"])}
        self.start_token = inverse_mapping["<S>"]
        self.end_token = inverse_mapping["<E>"]
        self.padding_token = inverse_mapping["<P>"]
        self.max_output_length = data_config["output_dims"][0]

        self.args = vars(args) if args is not None else {}
        self.dim = self.args.get("tf_dim", TF_DIM)
        tf_fc_dim = self.args.get("tf_fc_dim", TF_FC_DIM)
        tf_nhead = self.args.get("tf_nhead", TF_NHEAD)
        tf_dropout = self.args.get("tf_dropout", TF_DROPOUT)
        tf_layers = self.args.get("tf_layers", TF_LAYERS)

        # ## Encoder part - should output  vector sequence of length self.dim per sample
        resnet = torchvision.models.resnet18(weights=None)
        self.resnet = torch.nn.Sequential(*(list(resnet.children())[:-2]))  # Exclude AvgPool and Linear layers
        # Resnet will output (B, RESNET_DIM, _H, _W) logits where _H = input_H // 32, _W = input_W // 32

        self.encoder_projection = nn.Conv2d(RESNET_DIM, self.dim, kernel_size=1)
        # encoder_projection will output (B, dim, _H, _W) logits

        self.enc_pos_encoder = PositionalEncodingImage(
            d_model=self.dim, max_h=self.input_dims[1], max_w=self.input_dims[2]
        )  # Max (Ho, Wo)

        # ## Decoder part
        self.embedding = nn.Embedding(self.num_classes, self.dim)
        self.fc = nn.Linear(self.dim, self.num_classes)

        self.dec_pos_encoder = PositionalEncoding(d_model=self.dim, max_len=self.max_output_length)

        self.y_mask = generate_square_subsequent_mask(self.max_output_length)

        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.dim, nhead=tf_nhead, dim_feedforward=tf_fc_dim, dropout=tf_dropout),
            num_layers=tf_layers,
        )

        self.init_weights()  # This is empirically important

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Autoregressively produce sequences of labels from input images.

        Parameters
        ----------
        x
            (B, Ch, H, W) image, where Ch == 1 or Ch == 3

        Returns
        -------
        output_tokens
            (B, Sy) with elements in [0, C-1] where C is num_classes
        """
        B = x.shape[0]
        S = self.max_output_length
        x = self.encode(x)  # (Sx, B, E)

        output_tokens = (torch.ones((B, S)) * self.padding_token).type_as(x).long()  # (B, Sy)
        output_tokens[:, 0] = self.start_token  # Set start token
        for Sy in range(1, S):
            y = output_tokens[:, :Sy]  # (B, Sy)
            output = self.decode(x, y)  # (Sy, B, C)
            output = torch.argmax(output, dim=-1)  # (Sy, B)
            output_tokens[:, Sy] = output[-1]  # Set the last output token

            # Early stopping of prediction loop to speed up prediction
            if ((output_tokens[:, Sy] == self.end_token) | (output_tokens[:, Sy] == self.padding_token)).all():
                break

        # Set all tokens after end or padding token to be padding
        for Sy in range(1, S):
            ind = (output_tokens[:, Sy - 1] == self.end_token) | (output_tokens[:, Sy - 1] == self.padding_token)
            output_tokens[ind, Sy] = self.padding_token

        return output_tokens  # (B, Sy)

    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()
        self.fc.weight.data.uniform_(-initrange, initrange)

        nn.init.kaiming_normal_(self.encoder_projection.weight.data, a=0, mode="fan_out", nonlinearity="relu")
        if self.encoder_projection.bias is not None:
            _fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(self.encoder_projection.weight.data)
            bound = 1 / math.sqrt(fan_out)
            nn.init.normal_(self.encoder_projection.bias, -bound, bound)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode each image tensor in a batch into a sequence of embeddings.

        Parameters
        ----------
        x
            (B, Ch, H, W) image, where Ch == 1 or Ch == 3

        Returns
        -------
            (Sx, B, E) sequence of embeddings, going left-to-right, top-to-bottom from final ResNet feature maps
        """
        _B, C, _H, _W = x.shape
        if C == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.resnet(x)  # (B, RESNET_DIM, _H // 32, _W // 32),   (B, 512, 18, 20) in the case of IAMParagraphs
        x = self.encoder_projection(x)  # (B, E, _H // 32, _W // 32),   (B, 256, 18, 20) in the case of IAMParagraphs

        # x = x * math.sqrt(self.dim)  # (B, E, _H // 32, _W // 32)  # This prevented any learning
        x = self.enc_pos_encoder(x)  # (B, E, Ho, Wo);     Ho = _H // 32, Wo = _W // 32
        x = torch.flatten(x, start_dim=2)  # (B, E, Ho * Wo)
        x = x.permute(2, 0, 1)  # (Sx, B, E);    Sx = Ho * Wo
        return x

    def decode(self, x, y):
        """Decode a batch of encoded images x with guiding sequences y.

        During autoregressive inference, the guiding sequence will be previous predictions.

        During training, the guiding sequence will be the ground truth.

        Parameters
        ----------
        x
            (Sx, B, E) images encoded as sequences of embeddings
        y
            (B, Sy) guiding sequences with elements in [0, C-1] where C is num_classes

        Returns
        -------
        torch.Tensor
            (Sy, B, C) batch of logit sequences
        """
        y_padding_mask = y == self.padding_token
        y = y.permute(1, 0)  # (Sy, B)
        y = self.embedding(y) * math.sqrt(self.dim)  # (Sy, B, E)
        y = self.dec_pos_encoder(y)  # (Sy, B, E)
        Sy = y.shape[0]
        y_mask = self.y_mask[:Sy, :Sy].type_as(x)
        output = self.transformer_decoder(
            tgt=y, memory=x, tgt_mask=y_mask, tgt_key_padding_mask=y_padding_mask
        )  # (Sy, B, E)
        output = self.fc(output)  # (Sy, B, C)
        return output

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--tf_dim", type=int, default=TF_DIM)
        parser.add_argument("--tf_fc_dim", type=int, default=TF_DIM)
        parser.add_argument("--tf_dropout", type=float, default=TF_DROPOUT)
        parser.add_argument("--tf_layers", type=int, default=TF_LAYERS)
        parser.add_argument("--tf_nhead", type=int, default=TF_NHEAD)
        return parser
