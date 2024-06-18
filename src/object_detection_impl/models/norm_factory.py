from torch import nn

noop = nn.Identity()


class Norms:
    @staticmethod
    def get(norm: str, *args, **kwargs):
        if norm == "noop":
            return noop
        if norm == "bn1d":
            return nn.BatchNorm1d(*args, **kwargs)
        elif norm == "bn2d":
            return nn.BatchNorm2d(*args, **kwargs)
        elif norm == "ln":
            return nn.LayerNorm(*args, **kwargs)
        else:
            raise NotImplementedError
