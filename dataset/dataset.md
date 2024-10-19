# Dataset

### Unconditional image generation

* **FFHQ (Flickr-Faces-Hight-Quality )**

 **Flickr-Faces-HQ (FFHQ)** consists of **70,000** high-quality PNG images at 1024×1024 resolution and contains considerable variation in terms of age, ethnicity and image background.

project page:   [NVlabs/ffhq-dataset: Flickr-Faces-HQ Dataset (FFHQ) (github.com)](https://github.com/NVlabs/ffhq-dataset)

download link:  https://drive.google.com/open?id=1tZUcXDBeOibC6jcMCtgRRz67pzrAHeHL

```
git clone https://github.com/NVlabs/ffhq-dataset.git
python ffhq-dataset\download_ffhq.py --json --images -s
```



* **MetFaces**

MetFaces is an image dataset of human faces extracted from works of art. The dataset consists of **1336** high-quality PNG images at 1024×1024 resolution. The images were downloaded via the Metropolitan Museum of Art Collection API, and automatically aligned and cropped using dlib. Various automatic filters were used to prune the set.

project page:  [NVlabs/metfaces-dataset (github.com)](https://github.com/NVlabs/metfaces-dataset?tab=readme-ov-file)

download link: https://drive.google.com/open?id=1iChdwdW7mZFUyivKtDwL8ehCNhYKQz6D

```
wget https://drive.google.com/open?id=1iChdwdW7mZFUyivKtDwL8ehCNhYKQz6D
```



* LSUN

 The LSUN datasets can be conveniently downloaded via the script available [here](https://github.com/fyu/lsun). We performed a custom split into training and validation images, and provide the corresponding filenames at https://ommer-lab.com/files/lsun.zip. After downloading, extract them to `./data/lsun`. The beds/cats/churches subsets should also be placed/symlinked at `./data/lsun/bedrooms`/`./data/lsun/cats`/`./data/lsun/churches`, respectively.

```
# download GitHub repo
git clone https://github.com/fyu/lsun.git

# download provided file
wget https://ommer-lab.com/files/lsun.zip

# download data with lsun.zip
python lsun/download.py
```



### Class-conditional image generation

* **ImageNet1K**



### Text to image

* **MSCOCO validation set**

```
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip val2017.zip
unzip annotations_trainval2017.zip
```



* **LAION 400M**

project page:  [LAION-400-MILLION OPEN DATASET | LAION](https://laion.ai/blog/laion-400-open-dataset/)

download scripts:   [img2dataset/dataset_examples/laion400m.md at main · rom1504/img2dataset (github.com)](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion400m.md)

[laion-400m](https://laion.ai/laion-400-open-dataset/) is a 400M image text dataset

**Download the metadata**

```
wget -l1 -r --no-parent https://the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/
mv the-eye.eu/public/AI/cah/laion400m-met-release/laion400m-meta/ .
```

**Download the images with img2dataset**

```
pip install img2dataset
```

```
img2dataset --url_list laion400m-meta --input_format "parquet"\
         --url_col "URL" --caption_col "TEXT" --output_format webdataset\
           --output_folder laion400m-data --processes_count 16 --thread_count 128 --image_size 256\
             --save_additional_columns '["NSFW","similarity","LICENSE"]' --enable_wandb True
```

**Benchmark**

This can be downloaded at 1300 sample/s so it takes 3.5 days to download with one 16 cores 2Gbps machine. The result is 10TB
