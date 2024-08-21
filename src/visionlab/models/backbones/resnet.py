import timm
import torch.nn as nn
from visionlab.models.backbones.base import Backbone
from visionlab.utils.ml import freeze_until


class ResNetBackbone(Backbone):
    out_features: int = None

    def __init__(
        self,
        arch: str,
        pretrained: bool,
        n_layers: int = -2,
        freeze: bool = False,
        freeze_until_layer: str = None,
    ):
        super().__init__()
        if not arch.startswith("resnet"):
            raise ValueError(f"Unsupported ResNet backbone: {arch}")

        model = timm.create_model(arch, pretrained=pretrained)
        self.out_features = model.fc.in_features
        layers = list(model.children())[:n_layers]
        self.net = nn.Sequential(*layers)

        if freeze:
            freeze_until(self.net, freeze_until_layer)

    def forward(self, x):
        return self.net(x)
