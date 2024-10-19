torchrun --nproc_per_node=8 main.py -c configs/comparison/metfaces/faster.yaml -p train
torchrun --nproc_per_node=8 main.py -c configs/comparison/metfaces/faster.yaml -p inference ckpt_path="outputs/metfaces/faster2/train/checkpoints/0050000.pt"
python -m pytorch_fid "root_path_dataset_ImageNet" outputs/metfaces/faster2/inference/0050000

torchrun --nproc_per_node=8 main.py -c configs/comparison/ffhq/faster.yaml -p train
torchrun --nproc_per_node=8 main.py -c configs/comparison/ffhq/faster.yaml -p inference ckpt_path="outputs/ffhq/faster2/train/checkpoints/0050000.pt"
python -m pytorch_fid "root_path_dataset_ImageNet" outputs/ffhq/faster2/inference/0050000