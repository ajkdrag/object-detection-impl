import torch
import torch.nn as nn


class LearnablePositionEnc(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.pos_enc = nn.Parameter(torch.zeros(1, *sizes))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Parameter):
            nn.init.trunc_normal_(m, std=0.2)

    def forward(self, x):
        return x + self.pos_enc


class SinCos2DPositionalEncoding(nn.Module):
    def __init__(
        self,
        sizes,
        temperature: int = 10000,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        h, w, dim = sizes

        assert (dim % 4) == 0, "feature dimension must be multiple of 4 for sincos emb"

        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        omega = torch.arange(dim // 4) / (dim // 4 - 1)
        omega = 1.0 / (temperature**omega)

        y = y.flatten()[:, None] * omega[None, :]
        x = x.flatten()[:, None] * omega[None, :]
        pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)

        # Shape: (h * w, dim)
        self.register_buffer("pe", pe.type(dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
        Returns:
            Tensor of shape (batch_size, seq_len, dim) with positional encodings added
        """
        return x + self.pe
