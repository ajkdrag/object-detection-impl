import torch
from object_detection_impl.utils.ml import init_linear
from object_detection_impl.utils.registry import load_module
from torch import nn


class ClfModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.block = load_module(cfg.model)
        self.output_shape = self.block.output_shape
        self.apply(init_linear)

    def forward(self, x):
        return self.block(x)

    def forward_dummy(self):
        c, h, w = [self.cfg.model.ip.get(i) for i in ("c", "h", "w")]
        return self.forward(torch.randn([self.cfg.training.bs, c, h, w]))
