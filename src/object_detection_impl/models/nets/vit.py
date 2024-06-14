import torch.nn as nn
from einops.layers.torch import Rearrange
from object_detection_impl.models.blocks.attentions import (
    LearnablePositionEnc,
    MultiHead_SA,
)
from object_detection_impl.utils.registry import load_obj
from omegaconf import DictConfig


class ViTTiny(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        n_layers = 8
        embed_sz = 64
        heads = 2
        c = 3
        h = w = config.model.input_size

        self.stem = load_obj(config.model.stem.class_name)(
            in_channels=c,
            out_channels=embed_sz,
            **config.model.stem.get("params", {}),
        )

        _, _, stem_h, stem_w = self.stem.out_shape(c, h, w)

        self.trunk = nn.Sequential(
            Rearrange("n d h w -> n (h w) d"),
            LearnablePositionEnc(sizes=(stem_h * stem_w, embed_sz)),
            * [
                MultiHead_SA(
                    embed_sz,
                    num_heads=heads,
                    dropout=0.0,
                    expansion=1,
                )
                for _ in range(n_layers)
            ],
        )

        self.neck = load_obj(config.model.neck.class_name)(
            in_channels=embed_sz,
            **config.model.neck.get("params", {}),
        )

        self.head = load_obj(config.model.head.class_name)(
            in_features=embed_sz,
            **config.model.head.params,
        )

    def forward(self, x):
        return self.head(self.neck(self.trunk(self.stem(x))))
