import pytorch_lightning as pl

KEY = "optimizer"


class LearningRateMonitor(pl.callbacks.LearningRateMonitor):
    """Extends Lightning's LearningRateMonitor with a prefix.

    Logs the learning rate during training. See the docs for
    pl.callbacks.LearningRateMonitor for details.
    """

    def _add_prefix(self, *args, **kwargs) -> str:
        return f"{KEY}/" + super()._add_prefix(*args, **kwargs)
