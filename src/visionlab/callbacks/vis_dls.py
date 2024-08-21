from pathlib import Path

from lightning.pytorch.callbacks import Callback
from object_detection_impl.utils.vis import gridify


class VisualizeDlsCallback(Callback):
    def __init__(
        self,
        num_samples=8,
        dirpath="dls",
        scale=None,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.dirpath = Path(dirpath)
        self.scale = scale

    def on_fit_start(self, trainer, pl_module):
        dm = trainer.datamodule
        train_dls = dm.train_dataloader()
        val_dls = dm.val_dataloader()

        train_batch = next(iter(train_dls))
        val_batch = next(iter(val_dls))

        train_vis = dm.visualize_batch(
            train_batch,
            num_samples=self.num_samples,
        )
        val_vis = dm.visualize_batch(
            val_batch,
            num_samples=self.num_samples,
        )

        self.dirpath.mkdir(parents=True, exist_ok=True)
        gridify(
            train_vis,
            self.dirpath.joinpath("train.jpg"),
            scale=self.scale,
            nrow=4,
        )
        gridify(
            val_vis,
            self.dirpath.joinpath("val.jpg"),
            scale=self.scale,
            nrow=4,
        )
