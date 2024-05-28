import cv2
import numpy as np
from torchvision.datasets import FashionMNIST


class MnistDataset(FashionMNIST):
    def __getitem__(self, idx):
        image, target = self.data[idx], int(self.targets[idx])

        image = cv2.cvtColor(image.numpy(), cv2.COLOR_GRAY2RGB)

        if self.transform is not None:
            image = self.transform(image=image)["image"]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {"image": image, "target": np.array(target).astype("int64")}
