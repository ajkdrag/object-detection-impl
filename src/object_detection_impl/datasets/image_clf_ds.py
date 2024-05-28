from pathlib import Path
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def create_transforms():
    return A.Compose(
        [
            A.Resize(
                width=224,
                height=224,
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ],
    )


class ImageClfDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[Path, str]],
        mode: str = "train",
        transform=None,
        classes=None,
        class_to_idx=None,
    ):
        super().__init__()
        self.samples = samples
        self.mode = mode
        self.transform = transform
        self.classes = classes
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, target = self.samples[idx]

        image = cv2.imread(f"{image_path}", cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image is None:
            raise FileNotFoundError(image_path)

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return {"image": image, "target": np.array(target).astype("int64")}
