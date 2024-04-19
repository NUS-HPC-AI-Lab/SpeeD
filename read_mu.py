import os
import random

from speedit.dataset.transform import get_image_transform

image_transform = get_image_transform(256)
sample_num = 500
image_dir = [
    "/mnt/public/yuanzhihang/mscoco/val2017",
    "/mnt/public/yuanzhihang/mscoco/train2017",
    "/mnt/public/yuanzhihang/metfaces",
    "/mnt/public/yuanzhihang/imagenet/val",
    "/mnt/public/yuanzhihang/imagenet/ILSVRC/Data/CLS-LOC/train",
    "/mnt/public/yuanzhihang/ffhq/images256x256",
]
# image_dir = ["/mnt/public/yuanzhihang/mscoco/val2017"]


from torchvision.datasets.folder import default_loader

# calcuate data mu which is the mean of all images
mu = 0
num = 0

max_num = 0
min_num = 0
median_num = 0
mean_num = 0
std_num = 0
component = 0
import numpy as np
import torch
from tqdm import tqdm

betas = torch.linspace(1e-4, 0.02, 1000).double()
# betas =  torch.flip(betas, dims=[0])
alphas = 1.0 - betas
alphas_bar = torch.cumprod(alphas, dim=0)
sqrt_alphas_bar = torch.sqrt(alphas_bar)
sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - alphas_bar)

t = [0, 320, 640, 999]
print("t: ", t)


def add_noise(image, t):
    return image * sqrt_alphas_bar[t] + torch.randn_like(image) * sqrt_one_minus_alphas_bar[t]


# gaussian mixture
from sklearn.mixture import GaussianMixture

images = {0: [], 320: [], 640: [], 999: []}

for dir in tqdm(image_dir):
    # get image path list from dir, consider the sub dir
    image_path_list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                image_path = os.path.join(root, file)
                image_path_list.append(image_path)

    # sample image_path_list
    if len(image_path_list) < sample_num:
        sample_image_path_list = image_path_list
    else:
        sample_image_path_list = random.sample(image_path_list, sample_num)

    # read image and transform
    for image_path in sample_image_path_list:
        x0 = default_loader(image_path)
        x0 = image_transform(x0)

        for ind in t:
            image = add_noise(x0, ind)
            images[ind].append(image.numpy())
        # # calculate mu
        # max_num += torch.max(image)
        # min_num += torch.min(image)
        # median_num += torch.median(image)
        # mean_num += torch.mean(image)
        # std_num += torch.std(image)
        # component += image
        # num = num + 1

# breakpoint()
dim = 1024
print("Dimensionality: ", dim)

means = {}
for ind in t:
    t_image = images[ind]
    gm = GaussianMixture(n_components=1, random_state=0)
    from sklearn.decomposition import KernelPCA

    pca = KernelPCA(n_components=dim, random_state=0)

    t_image = np.stack(t_image).reshape(len(t_image), -1)
    print(len(t_image))

    d2_images = pca.fit_transform(t_image)

    gm.fit(d2_images)

    # breakpoint()
    print("t: ", ind)
    breakpoint()
    print("Means: ")
    print(gm.means_)
    print("Covariances: ")
    print(gm.covariances_)

    # save in json


# mean_max = max_num / num
# mean_min = min_num / num
# mean_median = median_num / num
# mean_mean = mean_num / num
# mean_std = std_num / num
#
# print("sample wise")
# print("max:", mean_max.item(), "min:", mean_min.item(), "median:", mean_median.item(), "mean:", mean_mean.item(), "std:", mean_std.item(), sep=", ")
#
# component = component / num
# max_component = torch.max(component)
# min_component = torch.min(component)
# median_component = torch.median(component)
# mean_component = torch.mean(component)
# std_component = torch.std(component)
#
# print("component wise")
# print("max:", max_component.item(), "min:", min_component.item(), "median:", median_component.item(), "mean:", mean_component.item(), "std:", std_component.item(), sep=", ")

# mean_image = mu / num
# 获得最大最大，最小，中位数，均值，方差
# max_num = torch.max(mean_image)
# min_num = torch.min(mean_image)
# median_num = torch.median(mean_image)
# mean_num = torch.mean(mean_image)
# std_num = torch.std(mean_image)
#
# print("max:", max_num, "min:", min_num, "median:", median_num, "mean:", mean_num, "std:", std_num)
