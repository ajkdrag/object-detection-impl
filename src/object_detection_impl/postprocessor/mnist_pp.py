import numpy as np
import torch
from object_detection_impl.utils.vis import draw_labels_on_images
from omegaconf import DictConfig


def postprocess(inputs, outputs: torch.tensor, dl, cfg: DictConfig):
    outputs_resolved = np.array(dl.dataset.classes)[outputs.detach().cpu()]

    std = torch.tensor([0.1307, 0.1307, 0.1307]).view(3, 1, 1)
    mean = torch.tensor([0.3081, 0.3081, 0.3081]).view(3, 1, 1)
    images = inputs["image"] * std + mean

    return draw_labels_on_images(images, outputs_resolved, 20, size=120)