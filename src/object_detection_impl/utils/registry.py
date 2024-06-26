import importlib
from typing import Any


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
