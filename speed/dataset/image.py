from torchvision.datasets import ImageFolder, VisionDataset

from .transform import get_image_transform


def image_dataset(root, image_size, class_cond=False, text_cond=False, ann_path=None):
    transform = get_image_transform(image_size)
    # check if root is a directory with subdirectories
    assert class_cond == False or text_cond == False, "class_cond and text_cond cannot be True at the same time"
    if class_cond == True:
        return ImageFolder(root, transform)
    elif text_cond == True:
        return ImageTextDataset(root, ann_path, transform)
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


import json


class ImageTextDataset(VisionDataset):
    def __init__(self, root, ann_path, transform, **kwargs):
        super().__init__(root, transform=transform)
        """
                image_root (string): Root directory of images (e.g. coco/images/)
                ann_root (string): directory to store the annotation file
                """
        super(ImageTextDataset, self).__init__(root, transform=transform, **kwargs)
        self.root = root
        self.anns = json.load(open(ann_path, "r"))
        self.loader = default_loader
        self.images_info = self.anns["images"]
        self.caption_info = self.anns["annotations"]

        # process
        self.images = {}

        for info in self.images_info:
            record = {
                "file_name": info["file_name"],
                "id": info["id"],
                "height": info["height"],
                "width": info["width"],
            }
            self.images[info["id"]] = record

        self.captions = []
        for caption in self.caption_info:
            record = {"caption": caption["caption"], "image_id": caption["image_id"]}
            self.captions.append(record)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        record = self.captions[index]
        image_id = record["image_id"]
        image_info = self.images[image_id]
        image_path = os.path.join(self.root, image_info["file_name"])
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        caption = record["caption"]
        return image, caption