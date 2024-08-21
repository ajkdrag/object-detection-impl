import importlib
from collections import OrderedDict
from typing import Any

from object_detection_impl.utils.ml import compute_output_shape
from omegaconf import DictConfig
from torch import nn


def load_obj(obj_path: str, namespace: str = "object_detection_impl") -> Any:
    """
    Extract an object from a given path.
    https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            namespace: Default namespace.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else ""
    obj_name = obj_path_list[0]
    try:
        module_obj = importlib.import_module(f"{obj_path}")
    except ModuleNotFoundError:
        module_obj = importlib.import_module(f"{namespace}.{obj_path}")
    if not hasattr(module_obj, obj_name):
        raise AttributeError(
            f"Object `{obj_name}` cannot be loaded from `{obj_path}`.")
    return getattr(module_obj, obj_name)


def load_module(cfg):
    layers = cfg["layers"]
    c, h, w = (cfg["ip"].get(i) for i in ("c", "h", "w"))
    input_shape = (c, h, w)
    custom_modules = "models.blocks"

    module_dict = OrderedDict()
    for layer_name, layer_config in layers.items():
        from_layer, num_repeats, block, block_args = layer_config

        if from_layer == "ip":
            input_shape = (c, h, w)
        else:
            input_shape = module_dict[from_layer].output_shape

        if block.startswith("nn."):
            block_class = getattr(nn, block.split(".", 1)[1])
        else:
            try:
                block_class = load_obj(f"{custom_modules}.{block}")
            except KeyError:
                raise ImportError(f"Block '{block}' not found.")

        pos_args = []
        kwargs = {}
        for arg in block_args:
            if isinstance(arg, DictConfig):
                kwargs.update(arg)
            else:
                pos_args.append(arg)

        module = block_class(*pos_args, **kwargs)
        module.output_shape = compute_output_shape(module, input_shape)

        blocks = [module]
        for _ in range(num_repeats - 1):
            module = block_class(*pos_args, **kwargs)
            module.output_shape = compute_output_shape(module, input_shape)
            blocks.append(module)

        stage = nn.Sequential(*blocks)
        stage.output_shape = module.output_shape
        module_dict[layer_name] = stage
    model = nn.Sequential(module_dict)
    model.output_shape = compute_output_shape(model, (c, h, w))
    return model
