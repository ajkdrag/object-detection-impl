from typing import Any, Optional

from torch import nn


def freeze_until(net: Any, param_name: Optional[str]) -> None:
    """
    Freeze net until param_name

    https://opendatascience.slack.com/archives/CGK4KQBHD/p1588373239292300?thread_ts=1588105223.275700&cid=CGK4KQBHD

    Args:
        net:
        param_name:

    """
    found_name = False
    for name, params in net.named_parameters():
        if name == param_name:
            found_name = True
        params.requires_grad = found_name
    return found_name


def count_parameters(model: Any):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def swap_dims(x, d1, d2):
    # x: torch.tensor, d1 and d2: dims to swap
    dims = list(range(x.dim()))
    dims[d1], dims[d2] = dims[d2], dims[d1]
    return x.permute(dims)


def init_linear(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.zeros_(m.bias)
