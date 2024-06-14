import lightning as L
import torch.nn as nn
from object_detection_impl.utils.registry import load_obj
from omegaconf import DictConfig


class LitImageClassifier(L.LightningModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = load_obj(cfg.model.class_name)(cfg)
        self.loss = load_obj(cfg.loss.class_name)(
            **cfg.loss.get("params", {}),
        )
        self.metrics = nn.ModuleDict()
        for metric_name, metric in cfg.metric.items():
            if metric_name.startswith("_"):
                continue
            self.metrics.update(
                {
                    metric_name: load_obj(metric.class_name)(
                        **metric.params,
                    ),
                }
            )

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def predict_step(self, batch, batch_idx):
        image = batch["image"]
        if image.device != self.device:
            image = image.to(self.device)
        return self(image).argmax(dim=-1)

    def configure_optimizers(self):
        if self.cfg.scheduler.class_name.endswith(("OneCycleLR",)):
            self.cfg.scheduler.params.total_steps = (
                self.trainer.estimated_stepping_batches
            )
        optimizer = load_obj(self.cfg.optimizer.class_name)(
            self.model.parameters(), **self.cfg.optimizer.params
        )
        scheduler = load_obj(self.cfg.scheduler.class_name)(
            optimizer, **self.cfg.scheduler.params
        )

        return (
            [optimizer],
            [
                {
                    "scheduler": scheduler,
                    "interval": self.cfg.scheduler.step,
                    "monitor": self.cfg.scheduler.monitor,
                }
            ],
        )

    def training_step(self, batch, *args, **kwargs):
        image = batch["image"]
        logits = self(image)

        target = batch["target"]
        loss = self.loss(logits, target)
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        for metric in self.metrics:
            score = self.metrics[metric](logits, target)
            self.log(
                f"train_{metric}",
                score,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        self.log(
            "lr",
            lr,
            on_step=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, *args, **kwargs):
        image = batch["image"]
        logits = self(image)

        target = batch["target"]
        loss = self.loss(logits, target)

        self.log(
            "valid_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        for metric in self.metrics:
            score = self.metrics[metric](logits, target)
            self.log(
                f"valid_{metric}",
                score,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def test_step(self, batch, *args, **kwargs):
        image = batch["image"]
        target = batch["target"]
        logits = self(image)

        for metric in self.metrics:
            score = self.metrics[metric](logits, target)
            self.log(
                f"test_{metric}",
                score,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
