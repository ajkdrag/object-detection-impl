from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback
from PIL import Image
from torchvision.utils import make_grid


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

    def _make_grid(self, visualized, fp, save=True, **kwargs):
        grid = make_grid(visualized, **kwargs)
        ndarr = (
            grid.mul(255)
            .add_(0.5)
            .clamp_(0, 255)
            .permute(1, 2, 0)
            .to("cpu", torch.uint8)
            .numpy()
        )
        im = Image.fromarray(ndarr)
        if self.scale is not None:
            og_size = im.size
            size = tuple([int(x * self.scale) for x in og_size])
            im = im.resize(size)
        if save:
            im.save(fp)
        else:
            return im

    def on_fit_start(self, trainer, pl_module):
        dm = trainer.datamodule
        train_dls = dm.train_dataloader()
        val_dls = dm.val_dataloader()

        train_batch = next(iter(train_dls))
        val_batch = next(iter(val_dls))

        train_vis = dm.visualize_batch(train_batch, self.num_samples)
        val_vis = dm.visualize_batch(val_batch, self.num_samples)

        self.dirpath.mkdir(parents=True, exist_ok=True)
        self._make_grid(
            train_vis,
            self.dirpath.joinpath("train.jpg"),
            nrow=4,
            normalize=True,
        )
        self._make_grid(
            val_vis,
            self.dirpath.joinpath("val.jpg"),
            nrow=4,
            normalize=True,
        )
