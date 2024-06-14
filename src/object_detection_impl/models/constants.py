from enum import Enum


class Shapes(Enum):
    NCHW = "n c h w"
    NPC = "n p c"


class Einops(Enum):
    FLATTEN_IMG = "n c h w -> n (h w) c"
    UNFLATTEN_IMG = "n (h w) c -> n c h w"
    G_AVG_POOL = "n p c -> n c"
