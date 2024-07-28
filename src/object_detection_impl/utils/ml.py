from typing import Any, Optional

import torch
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
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1.0)
        nn.init.zeros_(m.bias)


def compute_output_shape(module, input_shape):
    with torch.no_grad():
        dummy_input = torch.zeros(1, *input_shape)
        output = module(dummy_input)
        return output.shape[1:]


def make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
