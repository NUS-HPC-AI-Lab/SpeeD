#torchrun --nproc_per_node=8 --master_port=30002 main.py -c configs/image/unconditional/ffhq_faster.yaml -p train
#torchrun --nproc_per_node=8 --master_port=30002 main.py -c configs/image/unconditional/ffhq_base.yaml -p train
#
#python main.py -c configs/image/unconditional/ffhq_base.yaml -p train ckpt_path=""

torchrun --nproc_per_node=8 --master_port=30002 main.py -c configs/image/unconditional/ffhq_faster.yaml -p inference ckpt_path="/mnt/public/yuanzhihang/outputs/ffhq/faster/train/checkpoints/0070000.pt"
torchrun --nproc_per_node=8 --master_port=30002 main.py -c configs/image/unconditional/ffhq_base.yaml -p inference ckpt_path="/mnt/public/yuanzhihang/outputs/ffhq/base/train/checkpoints/0070000.pt"

torchrun --nproc_per_node=8 --master_port=30002 main.py -c configs/image/unconditional/ffhq_faster.yaml -p inference ckpt_path="/mnt/public/yuanzhihang/outputs/ffhq/faster/train/checkpoints/0080000.pt"
torchrun --nproc_per_node=8 --master_port=30002 main.py -c configs/image/unconditional/ffhq_base.yaml -p inference ckpt_path="/mnt/public/yuanzhihang/outputs/ffhq/base/train/checkpoints/0080000.pt"

#torchrun --nproc_per_node=8 --master_port=30002 main.py -c configs/image/unconditional/ffhq_faster.yaml -p inference ckpt_path="/mnt/public/yuanzhihang/outputs/ffhq/faster/train/checkpoints/0060000.pt"
#torchrun --nproc_per_node=8 --master_port=30002 main.py -c configs/image/unconditional/ffhq_base.yaml -p inference ckpt_path="/mnt/public/yuanzhihang/outputs/ffhq/base/train/checkpoints/0060000.pt"
#
#torchrun --nproc_per_node=8 --master_port=30002 main.py -c configs/image/unconditional/ffhq_faster.yaml -p inference ckpt_path="/mnt/public/yuanzhihang/outputs/ffhq/faster/train/checkpoints/0030000.pt"
#torchrun --nproc_per_node=8 --master_port=30002 main.py -c configs/image/unconditional/ffhq_base.yaml -p inference ckpt_path="/mnt/public/yuanzhihang/outputs/ffhq/base/train/checkpoints/0030000.pt"
#
#torchrun --nproc_per_node=8 --master_port=30002 main.py -c configs/image/unconditional/ffhq_faster.yaml -p inference ckpt_path="/mnt/public/yuanzhihang/outputs/ffhq/faster/train/checkpoints/0020000.pt"
#torchrun --nproc_per_node=8 --master_port=30002 main.py -c configs/image/unconditional/ffhq_base.yaml -p inference ckpt_path="/mnt/public/yuanzhihang/outputs/ffhq/base/train/checkpoints/0020000.pt"
#
#torchrun --nproc_per_node=8 --master_port=30002 main.py -c configs/image/unconditional/ffhq_faster.yaml -p inference ckpt_path="/mnt/public/yuanzhihang/outputs/ffhq/faster/train/checkpoints/0010000.pt"
#torchrun --nproc_per_node=8 --master_port=30002 main.py -c configs/image/unconditional/ffhq_base.yaml -p inference ckpt_path="/mnt/public/yuanzhihang/outputs/ffhq/base/train/checkpoints/0010000.pt"
#

#
#torchrun --nproc_per_node=8 --master_port=30002 main.py -c configs/image/unconditional/ffhq_faster.yaml -p inference ckpt_path="/mnt/public/yuanzhihang/outputs/ffhq/faster/train/checkpoints/0010000.pt"
#torchrun --nproc_per_node=8 --master_port=30002 main.py -c configs/image/unconditional/ffhq_base.yaml -p inference ckpt_path="/mnt/public/yuanzhihang/outputs/ffhq/base/train/checkpoints/0010000.pt"
