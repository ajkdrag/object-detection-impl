import lightning as L
import numpy as np
import structlog
import torch
from visionlab.augs.albumentations_aug import load_augs
from visionlab.utils.registry import load_obj
from visionlab.utils.vis import draw_labels_on_images
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split
from torch.utils.data._utils.collate import default_collate

log = structlog.get_logger()


class TorchvisionClfDatamodule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.path = cfg.datamodule.path
        self.name = cfg.datamodule.dataset.name
        self.splits = cfg.datamodule.splits
        self.dataset_cls = load_obj(cfg.datamodule.dataset.class_name)
        self.train_augs = load_augs(cfg.augmentation.train)
        self.val_augs = load_augs(cfg.augmentation.val)
        self.train_collate_fn = self.get_collate_fn(self.train_augs)
        self.val_collate_fn = self.get_collate_fn(self.val_augs)

    def get_collate_fn(self, transform):
        def collate_fn(batch):
            [
                sample.update(
                    {
                        "image": transform(
                            image=sample["image"],
                        )["image"]
                    }
                )
                for sample in batch
            ]
            return default_collate(batch)

        return collate_fn

    def prepare_data(self):
        self.dataset_cls(
            self.name,
            root=self.path,
            train=True,
            download=True,
        )
        self.dataset_cls(
            self.name,
            root=self.path,
            train=False,
            download=True,
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset_cls(
                self.name,
                root=self.path,
                train=True,
            )

            self.val_dataset = self.dataset_cls(
                self.name,
                root=self.path,
                train=False,
            )
            self.classes = self.train_dataset.classes

            if self.cfg.training.debug:
                self.train_dataset, _ = random_split(
                    self.train_dataset,
                    [0.1, 0.9],
                )
                self.val_dataset, _ = random_split(
                    self.val_dataset,
                    [0.1, 0.9],
                )
            log.info(f"train_dataset size: {len(self.train_dataset)}")
            log.info(f"val_dataset size: {len(self.val_dataset)}")

        if stage in ["fit", "test", "predict"] or stage is None:
            self.test_dataset = self.dataset_cls(
                self.name,
                root=self.path,
                train=False,
            )
            self.classes = self.test_dataset.classes

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.datamodule.batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            collate_fn=self.train_collate_fn,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.datamodule.batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            collate_fn=self.val_collate_fn,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.datamodule.test_batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            collate_fn=self.val_collate_fn,
            pin_memory=False,
            shuffle=False,
            drop_last=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.datamodule.test_batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            collate_fn=self.val_collate_fn,
            pin_memory=False,
            shuffle=True,
            drop_last=False,
        )

    def visualize_batch(self, inputs, outputs=None, num_samples=8):
        if outputs is None:
            outputs = inputs["target"]
            inputs = inputs["image"]
        outputs_resolved = np.array(self.classes)[outputs.detach().cpu()]

        std = torch.tensor(self.cfg.datamodule.dataset.std).view(3, 1, 1)
        mean = torch.tensor(self.cfg.datamodule.dataset.mean).view(3, 1, 1)
        images = inputs.detach().cpu() * std + mean

        return draw_labels_on_images(
            images[:num_samples],
            outputs_resolved[:num_samples],
            20,
            size=120,
        )
