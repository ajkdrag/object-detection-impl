from object_detection_impl.models.blocks.attentions import (
    CBAMBlock,
    ChannelAttentionBlock,
    ECABlock,
    ECABlockV2,
    SEBlock,
    SpatialAttentionBlock,
)
from torch import nn

noop = nn.Identity()


class Attentions:
    @staticmethod
    def get(attn: str, *args, **kwargs):
        if attn == "ca":
            return ChannelAttentionBlock(*args, **kwargs)
        elif attn == "sa":
            return SpatialAttentionBlock(*args, **kwargs)
        elif attn == "cbam":
            return CBAMBlock(*args, **kwargs)
        elif attn == "se":
            return SEBlock(*args, **kwargs)
        elif attn == "eca":
            return ECABlock(*args, **kwargs)
        elif attn == "ecav2":
            return ECABlockV2(*args, **kwargs)
        else:
            return noop
