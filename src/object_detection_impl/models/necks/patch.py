from einops.layers.torch import Reduce
from torch import einsum, nn


class PatchNorm(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.LayerNorm(in_channels),
            Reduce("n p c -> n c", "mean"),
        )

    def forward(self, x):
        return self.block(x)


class SequencePooling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attn_pool = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, 1),
        )

    def forward(self, x):
        # x: [n, p, c]
        attn_weights = self.attn_pool(x).squeeze(-1)  # [n, p]
        return einsum(
            "n p, n p c -> n c",
            attn_weights.softmax(dim=1),
            x,
        )
