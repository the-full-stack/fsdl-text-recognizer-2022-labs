import os
from pathlib import Path
import tempfile

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import torch

from .util import check_and_warn, logging

try:
    import torchviz

    has_torchviz = True
except ImportError:
    has_torchviz = False


class ModelSizeLogger(pl.Callback):
    """Logs information about model size (in parameters and on disk)."""

    def __init__(self, print_size=True):
        super().__init__()
        self.print_size = print_size

    @rank_zero_only
    def on_fit_start(self, trainer, module):
        self._run(trainer, module)

    def _run(self, trainer, module):
        metrics = {}
        metrics["mb_disk"] = self.get_model_disksize(module)
        metrics["nparams"] = count_params(module)

        if self.print_size:
            print(f"Model State Dict Disk Size: {round(metrics['mb_disk'], 2)} MB")

        metrics = {f"size/{key}": value for key, value in metrics.items()}

        trainer.logger.log_metrics(metrics, step=-1)

    @staticmethod
    def get_model_disksize(module):
        """Determine the model's size on disk by saving it to disk."""
        with tempfile.NamedTemporaryFile() as f:
            torch.save(module.state_dict(), f)
            size_mb = os.path.getsize(f.name) / 1e6
        return size_mb


class GraphLogger(pl.Callback):
    """Logs a compute graph as an image."""

    def __init__(self, output_key="logits"):
        super().__init__()
        self.graph_logged = False
        self.output_key = output_key
        if not has_torchviz:
            raise ImportError("GraphLogCallback requires torchviz." "")

    @rank_zero_only
    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx, dataloader_idx):
        if not self.graph_logged:
            try:
                outputs = outputs[0][0]["extra"]
                self.log_graph(trainer, module, outputs[self.output_key])
            except KeyError:
                logging.warning(f"Unable to log graph: outputs not found at key {self.output_key}")
            self.graph_logged = True

    @staticmethod
    def log_graph(trainer, module, outputs):
        if check_and_warn(trainer.logger, "log_image", "graph"):
            return
        params_dict = dict(list(module.named_parameters()))
        graph = torchviz.make_dot(outputs, params=params_dict)
        graph.format = "png"
        fname = Path(trainer.logger.experiment.dir) / "graph"
        graph.render(fname)
        fname = str(fname.with_suffix("." + graph.format))
        trainer.logger.log_image(key="graph", images=[fname])


def count_params(module):
    """Counts the number of parameters in a Torch Module."""
    return sum(p.numel() for p in module.parameters())
