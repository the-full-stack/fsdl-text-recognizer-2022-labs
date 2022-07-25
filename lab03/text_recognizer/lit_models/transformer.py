"""An encoder-decoder Transformer model"""
from typing import List, Sequence

import torch

from .base import BaseImageToTextLitModel
from .util import replace_after


class TransformerLitModel(BaseImageToTextLitModel):
    """
    Generic image to text PyTorch-Lightning module that must be initialized with a PyTorch module.

    The module must implement an encode and decode method, and the forward method
    should be the forward pass during production inference.
    """

    def __init__(self, model, args=None):
        super().__init__(model, args)
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=self.padding_index)

    def forward(self, x):
        return self.model(x)

    def teacher_forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Uses provided sequence y as guide for non-autoregressive encoding-decoding of x.

        Parameters
        ----------
        x
            Batch of images to be encoded. See self.model.encode for shape information.
        y
            Batch of ground truth output sequences.

        Returns
        -------
        torch.Tensor
            (B, C, Sy) logits
        """
        x = self.model.encode(x)
        output = self.model.decode(x, y)  # (Sy, B, C)
        return output.permute(1, 2, 0)  # (B, C, Sy)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.teacher_forward(x, y[:, :-1])
        loss = self.loss_fn(logits, y[:, 1:])

        self.log("train/loss", loss)

        outputs = {"loss": loss}

        return outputs

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # compute loss as in training, for comparison
        logits = self.teacher_forward(x, y[:, :-1])
        loss = self.loss_fn(logits, y[:, 1:])

        self.log("validation/loss", loss, prog_bar=True, sync_dist=True)

        outputs = {"loss": loss}

        # compute predictions as in production, for comparison
        preds = self(x)
        self.val_cer(preds, y)
        self.log("validation/cer", self.val_cer, prog_bar=True, sync_dist=True)

        return outputs

    def test_step(self, batch, batch_idx):
        x, y = batch
        # compute loss as in training, for comparison
        logits = self.teacher_forward(x, y[:, :-1])
        loss = self.loss_fn(logits, y[:, 1:])

        self.log("test/loss", loss, prog_bar=True, sync_dist=True)

        outputs = {"loss": loss}

        # compute predictions as in production, for comparison
        preds = self(x)
        self.val_cer(preds, y)
        self.log("test/cer", self.val_cer, prog_bar=True, sync_dist=True)

        return outputs

    def map(self, ks: Sequence[int], ignore: bool = True) -> str:
        """Maps an iterable of integers to a string using the lit model's mapping."""
        if ignore:
            return "".join([self.mapping[k] for k in ks if k not in self.ignore_tokens])
        else:
            return "".join([self.mapping[k] for k in ks])

    def batchmap(self, ks: Sequence[Sequence[int]], ignore=True) -> List[str]:
        """Maps a list of lists of integers to a list of strings using the lit model's mapping."""
        return [self.map(k, ignore) for k in ks]

    def get_preds(self, logitlikes: torch.Tensor, replace_after_end: bool = True) -> torch.Tensor:
        """Converts logit-like Tensors into prediction indices, optionally overwritten after end token index.

        Parameters
        ----------
        logitlikes
            (B, C, Sy) Tensor with classes as second dimension. The largest value is the one
            whose index we will return. Logits, logprobs, and probs are all acceptable.
        replace_after_end
            Whether to replace values after the first appearance of the end token with the padding token.

        Returns
        -------
        torch.Tensor
            (B, Sy) Tensor of integers in [0, C-1] representing predictions.
        """
        raw = torch.argmax(logitlikes, dim=1)  # (B, C, Sy) -> (B, Sy)
        if replace_after_end:
            return replace_after(raw, self.end_index, self.padding_index)  # (B, Sy)
        else:
            return raw  # (B, Sy)
