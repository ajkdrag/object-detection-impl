import albumentations as A
from object_detection_impl.utils.registry import load_obj
from omegaconf import DictConfig
from omegaconf.listconfig import ListConfig


def load_augs(cfg: DictConfig) -> A.Compose:
    """
    Load albumentations

    Args:
        cfg:

    Returns:
        compose object
    """
    augs = []
    for a in cfg:
        if a.class_name == "albumentations.OneOf":
            small_augs = []
            for small_aug in a.params:
                # yaml can't contain tuples, so we need to convert manually
                params = {
                    k: (v if type(v) is not ListConfig else tuple(v))
                    for k, v in small_aug.params.items()
                }
                aug = load_obj(small_aug.class_name)(**params)
                small_augs.append(aug)
            aug = load_obj(a.class_name)(small_augs)
            augs.append(aug)

        else:
            params = {
                k: (v if type(v) is not ListConfig else tuple(v))
                for k, v in a.params.items()
            }
            aug = load_obj(a.class_name)(**params)
            augs.append(aug)

    return A.Compose(augs)
