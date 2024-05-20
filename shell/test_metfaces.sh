#torchrun --nproc_per_node=8 main.py -c configs/comparison/metfaces/base.yaml -p inference ckpt_path="/home/zyk/fasterdit/outputs/metfaces/base/train/checkpoints/0050000.pt"
torchrun --nproc_per_node=8 main.py -c configs/comparison/metfaces/base.yaml -p inference ckpt_path="/home/zyk/fasterdit/outputs/metfaces/base/train/checkpoints/0100000.pt"
torchrun --nproc_per_node=8 main.py -c configs/comparison/metfaces/base.yaml -p inference ckpt_path="/home/zyk/fasterdit/outputs/metfaces/base/train/checkpoints/0150000.pt"
torchrun --nproc_per_node=8 main.py -c configs/comparison/metfaces/base.yaml -p inference ckpt_path="/home/zyk/fasterdit/outputs/metfaces/base/train/checkpoints/0200000.pt"
