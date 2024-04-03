from torchvision.datasets import ImageFolder, VisionDataset

from .transform import get_image_transform


def image_dataest(root, image_size, class_cond=True):
    transform = get_image_transform(image_size)
    # check if root is a directory with subdirectories
    if class_cond == True:
        return ImageFolder(root, transform)
    else:
        return ImageDataset(root, transform)


import os


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(os.listdir(data_dir)):
        full_path = os.path.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif os.path.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


from torchvision.datasets.folder import default_loader


class ImageDataset(VisionDataset):
    def __init__(self, data_dir, transform):
        super().__init__(data_dir, transform=transform)
        self.image_files = _list_image_files_recursively(data_dir)
        self.loader = default_loader

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        path = self.image_files[idx]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample
