import os

import numpy as np
from PIL import Image
from tqdm import tqdm


def uncomp_npz(npz_path, img_dir):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    with np.load(npz_path) as data:
        data = data["arr_0"]
        for i, arr in tqdm(enumerate(data)):
            # save in directory
            img = Image.fromarray(arr)
            img.save(os.path.join(img_dir, f"{i}.png"))

    print("done")


import argparse

parser = argparse.ArgumentParser(description="uncompress npz file to images")
parser.add_argument("--npz_path", type=str, help="path to npz file")
parser.add_argument("--img_dir", type=str, help="path to save images")

args = parser.parse_args()
npz_path = args.npz_path
img_dir = args.img_dir

uncomp_npz(npz_path, img_dir)
