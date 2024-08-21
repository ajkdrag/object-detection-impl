import torch.nn as nn
from einops.layers.torch import Reduce
from object_detection_impl.models.blocks.composites import (
    MDMLPMixerBlock,
)
from object_detection_impl.utils.registry import load_obj
from omegaconf import DictConfig


class MDMLPTiny(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        n_layers = 6
        embed_sz = 64

        self.stem = load_obj(config.model.stem.class_name)(
            in_channels=3,
            out_channels=embed_sz,
            **config.model.stem.get("params", {}),
        )

        patch_num_h = self.stem.patch_num_h
        patch_num_w = self.stem.patch_num_w

        self.trunk = nn.Sequential(
            *[
                MDMLPMixerBlock(
                    in_channels=3,
                    norm_channels=embed_sz,
                    dropout=0.2,
                    patch_num_h=patch_num_h,
                    patch_num_w=patch_num_w,
                )
                for _ in range(n_layers)
            ],
        )

        self.neck = Reduce("n c p d -> n d", "mean")

        self.head = load_obj(config.model.head.class_name)(
            in_features=embed_sz,
            **config.model.head.get("params", {}),
        )

    def forward(self, x):
        return self.head(self.neck(self.trunk(self.stem(x))))
