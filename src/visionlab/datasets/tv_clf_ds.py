import cv2
import numpy as np
import torch
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST


class TorchvisionClfDataset:
    map_converters = {
        "gray2rgb": lambda x: cv2.cvtColor(x, cv2.COLOR_GRAY2RGB),
        "tensor2npy": lambda x: x.numpy(),
        "noop": lambda x: x,
    }

    def __init__(self, dataset_name, **kwargs):
        self.dataset_name = dataset_name
        self._dataset = self._get_dataset(**kwargs)
        self.classes = self._dataset.classes
        self.class_to_idx = self._dataset.class_to_idx
        self.converters = []

        first = self._dataset.data[0]
        if isinstance(first, torch.Tensor):
            self.converters.append(self.map_converters["tensor2npy"])

        if first.ndim == 2:
            self.converters.append(self.map_converters["gray2rgb"])

    def _get_dataset(self, **kwargs):
        if self.dataset_name == "CIFAR10":
            return CIFAR10(**kwargs)
        elif self.dataset_name == "MNIST":
            return MNIST(**kwargs)
        elif self.dataset_name == "FashionMNIST":
            return FashionMNIST(**kwargs)
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

    def __getitem__(self, idx):
        image, target = self._dataset.data[idx], int(
            self._dataset.targets[idx])

        for func in self.converters:
            image = func(image)

        return {"image": image, "target": np.array(target).astype("int64")}

    def __len__(self):
        return len(self._dataset)
