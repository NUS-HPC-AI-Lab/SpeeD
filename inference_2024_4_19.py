checkpoint_file = "/mnt/public/yuanzhihang/outputs/metfaces/clts/train/checkpoints/0050000.pt"

import os
import time

while not os.path.exists(checkpoint_file):
    print("waiting ...")
    time.sleep(60 * 30)

os.system(
    f"torchrun --nproc_per_node=8 main.py -c configs/comparison/metfaces/clts.yaml -p inference ckpt_path={checkpoint_file}"
)

resume = "/mnt/public/yuanzhihang/outputs/metfaces/base/train/checkpoints/0045000.pt"
os.system(
    f"torchrun --nproc_per_node=8 main.py -c configs/comparison/metfaces/base.yaml -p inference resume_training={resume}"
)
os.system(
    f"torchrun --nproc_per_node=8 main.py -c configs/comparison/metfaces/base.yaml -p inference ckpt_path=/mnt/public/yuanzhihang/outputs/metfaces/base/train/checkpoints/0050000.pt"
)
