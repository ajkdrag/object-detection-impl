import random
from pathlib import Path
from typing import List, Tuple


class SplitsSubfolderParser:
    """
    dataset/
    ├── train/
    │   ├── class1/
    │   │   ├── image1.jpg
    │   │   └── ...
    │   ├── class2/
    │   │   ├── image1.jpg
    │   │   └── ...
    │   └── ...
    ├── val/
    │   ├── class1/
    │   │   ├── image1.jpg
    │   │   └── ...
    │   ├── class2/
    │   │   ├── image1.jpg
    │   │   └── ...
    │   └── ...
    └── test/
        ├── class1/
        │   ├── image1.jpg
        │   └── ...
        ├── class2/
        │   ├── image1.jpg
        │   └── ...
        └── ...
    """

    def __init__(self, path_src: str):
        self.path_src = Path(path_src)
        self.class_names = self._get_class_names()
        self.class_to_idx = {
            cls_name: idx for idx, cls_name in enumerate(self.class_names)
        }

    def _get_class_names(self) -> List[str]:
        train_dir = self.path_src.joinpath("train")
        class_names = sorted(
            [
                d.name
                for d in train_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]
        )
        return class_names

    def _gather_samples(self, split: str) -> List[Tuple[Path, int]]:
        split_dir = self.path_src.joinpath(split)
        samples = []

        for class_idx, class_name in enumerate(self.class_names):
            class_dir = split_dir.joinpath(class_name)
            for image_path in class_dir.iterdir():
                if image_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
                    samples.append((image_path, class_idx))

        return samples

    def get_train(self, shuffle=True) -> List[Tuple[Path, int]]:
        train_samples = self._gather_samples("train")
        if shuffle:
            random.shuffle(train_samples)
        return train_samples

    def get_val(self) -> List[Tuple[Path, int]]:
        return self._gather_samples("valid")

    def get_test(self) -> List[Tuple[Path, int]]:
        return self._gather_samples("test")
