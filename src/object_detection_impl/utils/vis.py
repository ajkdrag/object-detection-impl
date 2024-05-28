import torch
from torchvision.transforms.v2 import Resize
from torchvision.utils import draw_bounding_boxes


def draw_labels_on_images(images, labels, label_h=30, size=None):
    labeled_images = []
    for image, label in zip(images, labels):
        if size is not None:
            image = Resize(size=size)(image)
        image = (image * 255).byte()
        text_label = f"{label}"

        # Create a black background above the image
        black_bg = torch.zeros((3, label_h, image.shape[2]), dtype=torch.uint8)

        # Draw the label on the black background
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

        # Concatenate the labeled background and the original image vertically
        labeled_image = torch.cat((labeled_bg, image), dim=1)
        labeled_images.append(labeled_image)

    return torch.stack(labeled_images, dim=0)
