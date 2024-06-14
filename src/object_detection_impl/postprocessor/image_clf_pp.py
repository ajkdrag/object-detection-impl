import numpy as np
import torch
from object_detection_impl.utils.vis import draw_labels_on_images
from omegaconf import DictConfig


def postprocess(inputs, outputs: torch.tensor, dl, cfg: DictConfig):
    outputs_resolved = np.array(dl.dataset.classes)[outputs.detach().cpu()]

    std = torch.tensor(cfg.datamodule.dataset.std).view(3, 1, 1)
    mean = torch.tensor(cfg.datamodule.dataset.mean).view(3, 1, 1)
    images = inputs["image"] * std + mean

    return draw_labels_on_images(images, outputs_resolved)
