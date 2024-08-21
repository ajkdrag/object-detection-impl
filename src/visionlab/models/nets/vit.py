import torch.nn as nn
from einops.layers.torch import Rearrange
from visionlab.models.blocks.attentions import (
    LearnablePositionEnc,
)
from visionlab.models.blocks.composites import TransformerEncoder
from visionlab.utils.ml import init_linear
from visionlab.utils.registry import load_obj
from omegaconf import DictConfig


class ViTTiny(nn.Module):
    def __init__(self, config: DictConfig) -> None:
        super().__init__()
        n_layers = 2
        embed_sz = 128
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
            *[
                TransformerEncoder(
                    embed_sz,
                    num_heads=heads,
                    expansion=1,
                    dropout=0.2,
                )
                for _ in range(n_layers)
            ],
        )

        self.neck = load_obj(config.model.neck.class_name)(
            in_channels=embed_sz,
            **config.model.neck.get("params", {}),
        )

        self.head = load_obj(config.model.head.class_name)(
            in_channels=embed_sz,
            **config.model.head.params,
        )

        self.apply(init_linear)

    def forward(self, x):
        return self.head(self.neck(self.trunk(self.stem(x))))
