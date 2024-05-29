from torchvision.datasets import ImageFolder

from .transform import get_image_transform


def image_dataest(root, image_size):
    transform = get_image_transform(image_size)
    return ImageFolder(root, transform)
