import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from object_detection_impl.models.blocks.attentions import (
    Attentions,
    MultiHead_SA,
)
from object_detection_impl.models.blocks.composites import (
    ConvMixerBlock,
    DenseNetBlock,
    InceptionBlock,
    MLPMixerBlock,
    ResNetBlock,
)
from object_detection_impl.models.blocks.core import (
    ConvLayer,
)
from object_detection_impl.utils.registry import load_obj
from omegaconf import DictConfig


class LeNet5(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        self.layer1 = nn.Sequential(
            ConvLayer(3, 6, kernel_size=5, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.layer2 = nn.Sequential(
            ConvLayer(6, 16, kernel_size=5, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.head = load_obj(config.model.head.class_name)(
            in_features=16,
            **config.model.head.params,
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return self.head(x)


class LeNetV2(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            ConvLayer(3, 6, kernel_size=5, padding=0),
        )

        self.trunk = nn.Sequential(
            ResNetBlock(6, 12, stride=2),
            InceptionBlock(
                12,
                [6, 2],
                [8, 4],
                [3, 5],
            ),
        )

        self.head = load_obj(config.model.head.class_name)(
            in_features=12,
            **config.model.head.params,
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.trunk(x)
        return self.head(x)


class LeNetV3(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            ConvLayer(3, 6, kernel_size=5, padding=0),
        )

        self.trunk = nn.Sequential(
            MultiHead_SA(6),
            ResNetBlock(6, 6, attn=Attentions.ECA()),
            ConvLayer(6, 12, stride=2, padding=1),
            ResNetBlock(12, 12),
            ConvLayer(12, 24, stride=2, padding=1),
        )

        self.head = load_obj(config.model.head.class_name)(
            in_features=24,
            **config.model.head.params,
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.trunk(x)
        return self.head(x)


class LeNetV4(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            ConvLayer(3, 6, kernel_size=5, padding=0),
        )

        dense_1 = DenseNetBlock(6, 12)
        dense_2 = DenseNetBlock(dense_1.out_channels, 12)

        self.trunk = nn.Sequential(
            MultiHead_SA(6),
            dense_1,
            nn.MaxPool2d(kernel_size=2, stride=2),
            dense_2,
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.neck = load_obj(config.model.neck.class_name)(
            in_channels=dense_2.out_channels,
            **config.model.neck.get("params", {}),
        )

        self.head = load_obj(config.model.head.class_name)(
            in_features=dense_2.out_channels,
            **config.model.head.params,
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.trunk(x)
        return self.head(self.neck(x))


class LeNetV5(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.stem = load_obj(config.model.stem.class_name)(
            in_channels=3,
            out_channels=128,
            **config.model.stem.get("params", {}),
        )
        self.trunk = nn.Sequential(
            ConvMixerBlock(128, kernel_size=7),
            ConvMixerBlock(128, kernel_size=7),
            ConvMixerBlock(128, kernel_size=7),
            ConvMixerBlock(128, kernel_size=7),
        )
        self.head = load_obj(config.model.head.class_name)(
            in_features=128,
            **config.model.head.params,
        )

    def forward(self, x):
        return self.head(self.trunk(self.stem(x)))


class LeNetV6(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.stem = load_obj(config.model.stem.class_name)(
            in_channels=3,
            out_channels=128,
            **config.model.stem.get("params", {}),
        )
        patch_size = config.model.stem.params.get("patch_size", 1)
        num_patches = (config.model.input_size // patch_size) ** 2
        self.trunk = nn.Sequential(
            MLPMixerBlock(128, num_patches, 2, 0.5),
            MLPMixerBlock(128, num_patches, 2, 0.5),
            MLPMixerBlock(128, num_patches, 2, 0.5),
            MLPMixerBlock(128, num_patches, 2, 0.5),
            MLPMixerBlock(128, num_patches, 2, 0.5),
            MLPMixerBlock(128, num_patches, 2, 0.5),
        )

        self.neck = load_obj(config.model.neck.class_name)(
            in_channels=128,
            **config.model.neck.get("params", {}),
        )

        self.head = load_obj(config.model.head.class_name)(
            in_features=128,
            **config.model.head.get("params", {}),
        )

    def forward(self, x):
        return self.head(self.neck(self.trunk(self.stem(x))))
