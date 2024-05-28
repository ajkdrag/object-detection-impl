import random
from pathlib import Path
from typing import List

from object_detection_impl.utils.misc import split_list


class ClassSubfolderParser:
    """
    dataset/
    ├── class1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class2/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── ...
    """

    def __init__(
        self,
        path_src: str,
        train_val_test_split: List[float] = [0.7, 0.2, 0.1],
    ):
        self.path_src = Path(path_src)
        self.train_val_test_split = train_val_test_split
        self.class_names = self._get_class_names()
        self.class_to_idx = {
            class_name: i for i, class_name in enumerate(self.class_names)
        }
        all_samples = self._gather_samples()
        (
            self.train_samples,
            self.val_samples,
            self.test_samples,
        ) = split_list(all_samples, train_val_test_split)

    def _get_class_names(self) -> List[str]:
        return sorted(
            [
                d.name
                for d in self.path_src.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]
        )

    def _gather_samples(self):
        samples = []
        for class_name in self.class_names:
            class_path = self.path_src.joinpaht(class_name)
            for image_path in class_path.iterdir():
                if image_path.suffix in [".jpg", ".jpeg", ".png", ".bmp"]:
                    samples.append(
                        (
                            image_path,
                            self.mapping_dict[class_name],
                        )
                    )
        return samples

    def get_train(self, shuffle=True):
        if shuffle:
            random.shuffle(self.train_samples)
        self.train_samples

    def get_val(self):
        return self.val_samples

    def get_test(self):
        return self.test_samples
