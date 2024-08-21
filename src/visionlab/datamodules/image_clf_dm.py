import lightning as L
from visionlab.augs.albumentations_aug import load_augs
from visionlab.datamodules.parser_factory import FolderParserFactory
from visionlab.utils.registry import load_obj
from omegaconf import DictConfig
from torch.utils.data import DataLoader


class ImageClfDatamodule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.path = cfg.datamodule.path
        self.dataset_cls = load_obj(self.cfg.datamodule.dataset.class_name)
        self.train_augs = load_augs(self.cfg.augmentation.train)
        self.val_augs = load_augs(self.cfg.augmentation.val)

        self.folder_struct = cfg.datamodule.folder_struct.upper()
        self.folder_struct_parser = FolderParserFactory.get_folder_parser(
            self.folder_struct
        ).value(path_src=self.path, **cfg.datamodule.parser_params)

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        class_names = self.folder_struct_parser.class_names
        class_to_idx = self.folder_struct_parser.class_to_idx

        train_samples = self.folder_struct_parser.get_train()
        val_samples = self.folder_struct_parser.get_val()
        test_samples = self.folder_struct_parser.get_test()

        if self.cfg.training.debug:
            train_samples = train_samples[:100]
            val_samples = val_samples[:100]
            test_samples = test_samples[:100]

        if stage == "fit" or stage is None:
            self.train_dataset = self.dataset_cls(
                samples=train_samples,
                mode="train",
                classes=class_names,
                class_to_idx=class_to_idx,
                transform=self.train_augs,
            )

            self.val_dataset = self.dataset_cls(
                samples=val_samples,
                mode="val",
                classes=class_names,
                class_to_idx=class_to_idx,
                transform=self.val_augs,
            )

        if stage in ["test", "predict"] or stage is None:
            self.test_dataset = self.dataset_cls(
                samples=test_samples,
                mode="test",
                classes=class_names,
                class_to_idx=class_to_idx,
                transform=self.val_augs,
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
        bs = self.cfg.datamodule.test_batch_size if self.cfg.training.debug else 4
        return DataLoader(
            self.test_dataset,
            batch_size=bs,
            num_workers=self.cfg.datamodule.num_workers,
            pin_memory=self.cfg.datamodule.pin_memory,
            shuffle=True,
            drop_last=False,
        )
