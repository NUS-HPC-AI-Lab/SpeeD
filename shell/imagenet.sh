torchrun --nproc_per_node=8 main.py -c configs/maskdit/unet_base.yaml -p train resume_training="/home/zyk/fasterdit/outputs/imagenet/unet_base/train/checkpoints/0100000.pt"

torchrun --nproc_per_node=8 main.py -c configs/maskdit/unet_faster.yaml -p train

torchrun --nproc_per_node=8 --master_port=30003 main.py -c configs/maskdit/unet_base.yaml -p inference ckpt_path="/home/zyk/fasterdit/outputs/imagenet/unet_base/train/checkpoints/0400000.pt"
python -m pytorch_fid /home/zyk/fasterdit/outputs/imagenet/unet_base/inference/0400000
python main.py -c configs/maskdit/unet_base.yaml -p sample ckpt_path="/home/zyk/fasterdit/outputs/imagenet/unet_base/train/checkpoints/0400000.pt"
