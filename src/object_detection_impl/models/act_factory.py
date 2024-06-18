from torch import nn

noop = nn.Identity()


class Acts:
    @staticmethod
    def get(act: str, *args, **kwargs):
        if act == "noop":
            return noop
        if act == "relu":
            return nn.ReLU(*args, **kwargs)
        elif act == "gelu":
            return nn.GELU(*args, **kwargs)
        elif act == "sigmoid":
            return nn.Sigmoid(*args, **kwargs)
        else:
            raise NotImplementedError
