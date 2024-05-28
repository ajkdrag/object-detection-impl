import collections
from itertools import product
from typing import Dict, Generator

from omegaconf import DictConfig, OmegaConf


def product_dict(**kwargs: Dict) -> Generator:
    """
    Convert dict with lists in values into lists of all combinations

    This is necessary to convert config with experiment values
    into format usable by hydra
    Args:
        **kwargs:

    Returns:
        list of lists

    ---
    Example:
        >>> list_dict = {'a': [1, 2], 'b': [2, 3]}
        >>> list(product_dict(**list_dict))
        >>> [['a=1', 'b=2'], ['a=1', 'b=3'], ['a=2', 'b=2'], ['a=2', 'b=3']]

    """
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in product(*vals):
        zip_list = list(zip(keys, instance))
        yield [f"{i}={j}" for i, j in zip_list]


def config_to_hydra_dict(cfg: DictConfig) -> Dict:
    """
    Convert config into dict with lists of values, where key is full name of parameter

    This fuction is used to get key names which can be used in hydra.

    Args:
        cfg:

    Returns:
        converted dict

    """
    experiment_dict = {}
    for k, v in cfg.items():
        for k1, v1 in v.items():
            experiment_dict[f"{k!r}.{k1!r}"] = v1

    return experiment_dict


def flatten_omegaconf(d, sep="_"):
    d = OmegaConf.to_container(d)

    obj = collections.OrderedDict()

    def recurse(t, parent_key=""):
        if isinstance(t, list):
            for i, _ in enumerate(t):
                recurse(t[i], parent_key + sep + str(i) if parent_key else str(i))
        elif isinstance(t, dict):
            for k, v in t.items():
                recurse(v, parent_key + sep + k if parent_key else k)
        else:
            obj[parent_key] = t

    recurse(d)
    obj = {k: v for k, v in obj.items() if isinstance(v, (int, float))}

    return obj
