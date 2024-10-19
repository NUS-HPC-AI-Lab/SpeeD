torchrun --nproc_per_node=8 main.py -c configs/maskdit/unet_base.yaml -p train
torchrun --nproc_per_node=8 main.py -c configs/maskdit/unet_faster.yaml -p train
python -m pytorch_fid outputs/imagenet/unet_base/inference/0400000
python main.py -c configs/maskdit/unet_base.yaml -p sample ckpt_path="outputs/imagenet/unet_base/train/checkpoints/0400000.pt"