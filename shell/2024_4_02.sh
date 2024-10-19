torchrun --nproc_per_node=8 --master_port=30001 main.py -c configs/image/unconditional/metfaces_faster.yaml -p train
#
torchrun --nproc_per_node=8 --master_port=30002 main.py -c configs/image/unconditional/metfaces_base.yaml -p train



torchrun --nproc_per_node=8 --master_port=30001 main.py -c configs/image/unconditional/metfaces_faster.yaml -p inference ckpt_path="/mnt/public/yuanzhihang/outputs/metfaces/faster/train/checkpoints/0040000.pt"

torchrun --nproc_per_node=8 --master_port=30001 main.py -c configs/image/unconditional/metfaces_base.yaml -p inference ckpt_path="/mnt/public/yuanzhihang/outputs/metfaces/base/train/checkpoints/0040000.pt"


# eval
python -m pytorch_fid /home/yuanzhihang/metfaces/images /mnt/public/yuanzhihang/outputs/metfaces/faster/inference/0040000
python -m pytorch_fid /home/yuanzhihang/metfaces/images /mnt/public/yuanzhihang/outputs/metfaces/base/inference/0040000
