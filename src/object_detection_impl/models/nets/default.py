import torch.nn as nn
from object_detection_impl.utils.registry import load_obj
from omegaconf import DictConfig


class DefaultNet(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        self.backbone = load_obj(config.model.backbone.class_name)(
            **config.model.backbone.params,
        )

        self.head = load_obj(config.model.head.class_name)(
            in_features=self.backbone.out_features,
            **config.model.head.params,
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.head(x)
