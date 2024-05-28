import lightning as L
from object_detection_impl.augs.albumentations_aug import load_augs
from object_detection_impl.utils.registry import load_obj
from omegaconf import DictConfig
from torch.utils.data import DataLoader, random_split


class MnistDatamodule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.path = cfg.datamodule.path
        self.dataset_cls = load_obj(self.cfg.datamodule.dataset.class_name)
        self.train_augs = load_augs(self.cfg.augmentation.train)
        self.val_augs = load_augs(self.cfg.augmentation.val)

    def prepare_data(self):
        dataset_cls = load_obj(self.cfg.datamodule.dataset.class_name)
        dataset_cls(self.path, train=True, download=True)
        dataset_cls(self.path, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            mnist_full = self.dataset_cls(
                self.path,
                train=True,
                transform=self.train_augs,
            )
            self.train_dataset, self.val_dataset = random_split(
                mnist_full, [55000, 5000]
            )
            self.val_dataset.transform = self.val_augs

            if self.cfg.training.debug:
                self.train_dataset, _ = random_split(
                    self.train_dataset,
                    [1000, 54000],
                )
                self.val_dataset, _ = random_split(
                    self.val_dataset,
                    [1000, 4000],
                )

        if stage in ["test", "predict"] or stage is None:
            self.test_dataset = self.dataset_cls(
                self.path, train=False, transform=self.val_augs
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.datamodule.batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            shuffle=True,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.datamodule.batch_size,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def test_dataloader(self):
        bs = self.cfg.datamodule.test_batch_size if self.cfg.training.debug else 4
        return DataLoader(
            self.test_dataset,
            batch_size=bs,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            shuffle=False,
            drop_last=False,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
        bs = self.cfg.datamodule.test_batch_size if self.cfg.training.debug else 4
        return DataLoader(
            self.test_dataset,
            batch_size=bs,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            shuffle=True,
            drop_last=False,
        )
