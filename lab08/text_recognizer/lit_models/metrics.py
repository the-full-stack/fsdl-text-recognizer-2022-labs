"""Special-purpose metrics for tracking our model performance."""
from typing import Sequence

import torch
import torchmetrics


class CharacterErrorRate(torchmetrics.CharErrorRate):
    """Character error rate metric, allowing for tokens to be ignored."""

    def __init__(self, ignore_tokens: Sequence[int], *args):
        super().__init__(*args)
        self.ignore_tokens = set(ignore_tokens)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):  # type: ignore
        preds_l = [[t for t in pred if t not in self.ignore_tokens] for pred in preds.tolist()]
        targets_l = [[t for t in target if t not in self.ignore_tokens] for target in targets.tolist()]
        super().update(preds_l, targets_l)


def test_character_error_rate():
    metric = CharacterErrorRate([0, 1])
    X = torch.tensor(
        [
            [0, 2, 2, 3, 3, 1],  # error will be 0
            [0, 2, 1, 1, 1, 1],  # error will be .75
            [0, 2, 2, 4, 4, 1],  # error will be .5
        ]
    )
    Y = torch.tensor(
        [
            [0, 2, 2, 3, 3, 1],
            [0, 2, 2, 3, 3, 1],
            [0, 2, 2, 3, 3, 1],
        ]
    )
    metric(X, Y)
    assert metric.compute() == sum([0, 0.75, 0.5]) / 3


if __name__ == "__main__":
    test_character_error_rate()
