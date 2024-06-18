import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms.v2 import Resize
from torchvision.utils import draw_bounding_boxes, make_grid


def draw_labels_on_images(images, labels, label_h=30, size=None):
    labeled_images = []
    for image, label in zip(images, labels):
        if size is not None:
            image = Resize(size=size)(image)
        text_label = f"{label}"

        black_bg = torch.zeros(
            (3, label_h, image.shape[2]),
            dtype=torch.uint8,
        )

        labeled_bg = draw_bounding_boxes(
            image=black_bg,
            boxes=torch.tensor([[0, 0, black_bg.shape[-1], label_h]]),
            labels=[text_label],
            colors=["white"],
            font="../assets/Ubuntu-R.ttf",
            fill=False,
            width=2,
            font_size=label_h * 0.7,
        )

        labeled_image = torch.cat(
            (labeled_bg, (255 * image).byte()),
            dim=1,
        )
        labeled_images.append(labeled_image)

    return torch.stack(labeled_images, dim=0)


def gridify(visualized, fp, scale=None, save=True, **kwargs):
    im = to_pil_image(make_grid(visualized, **kwargs))
    if scale is not None:
        og_size = im.size
        size = tuple([int(x * scale) for x in og_size])
        im = im.resize(size)
    if save:
        im.save(fp)
    else:
        return im
