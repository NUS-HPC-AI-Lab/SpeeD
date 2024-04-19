#torchrun --nproc_per_node=8 --master_port=30003 main.py -c configs/image/text2img/mscoco_faster.yaml -p inference ckpt_path="/mnt/public/yuanzhihang/outputs/mscoco/base/train/checkpoints/0100000.pt"

#torchrun --nproc_per_node=8 --master_port=30003 main.py -c configs/image/text2img/mscoco_faster.yaml -p inference ckpt_path="/mnt/public/yuanzhihang/outputs/mscoco/base/train/checkpoints/0100000.pt"
#
#torchrun --nproc_per_node=8 --master_port=30003 main.py -c configs/image/text2img/mscoco_faster.yaml -p inference ckpt_path="/mnt/public/yuanzhihang/outputs/mscoco/base/train/checkpoints/0200000.pt"
#
#torchrun --nproc_per_node=8 --master_port=30003 main.py -c configs/image/text2img/mscoco_faster.yaml -p inference ckpt_path="/mnt/public/yuanzhihang/outputs/mscoco/base/train/checkpoints/0300000.pt"
#
#torchrun --nproc_per_node=8 --master_port=30003 main.py -c configs/image/text2img/mscoco_faster.yaml -p inference ckpt_path="/mnt/public/yuanzhihang/outputs/mscoco/base/train/checkpoints/0400000.pt"

#torchrun --nproc_per_node=8 --master_port=30003 main.py -c configs/image/text2img/mscoco_base.yaml -p inference ckpt_path="/mnt/public/yuanzhihang/outputs/mscoco/real_base/train/checkpoints/0100000.pt"

#python main.py -c configs/image/text2img/mscoco_base.yaml -p inference ckpt_path="/mnt/public/yuanzhihang/outputs/mscoco/real_base/train/checkpoints/0100000.pt"
#
#python main.py -c configs/image/text2img/mscoco_faster.yaml -p sample ckpt_path="/mnt/public/yuanzhihang/outputs/mscoco/base/train/checkpoints/0100000.pt"
#python main.py -c configs/image/text2img/mscoco_faster.yaml -p sample ckpt_path="/mnt/public/yuanzhihang/outputs/mscoco/base/train/checkpoints/0200000.pt"
#python main.py -c configs/image/text2img/mscoco_faster.yaml -p sample ckpt_path="/mnt/public/yuanzhihang/outputs/mscoco/base/train/checkpoints/0300000.pt"
#python main.py -c configs/image/text2img/mscoco_faster.yaml -p sample ckpt_path="/mnt/public/yuanzhihang/outputs/mscoco/base/train/checkpoints/0400000.pt"

#python -m pytorch_fid /mnt/public/yuanzhihang/mscoco/val2017 /mnt/public/yuanzhihang/outputs/mscoco/base/inference/0100000
#python -m pytorch_fid /mnt/public/yuanzhihang/mscoco/val2017 /mnt/public/yuanzhihang/outputs/mscoco/base/inference/0200000
#python -m pytorch_fid /mnt/public/yuanzhihang/mscoco/val2017 /mnt/public/yuanzhihang/outputs/mscoco/base/inference/0300000
#python -m pytorch_fid /mnt/public/yuanzhihang/mscoco/val2017 /mnt/public/yuanzhihang/outputs/mscoco/base/inference/0400000

#python -m pytorch_fid /mnt/public/yuanzhihang/mscoco/val /mnt/public/yuanzhihang/outputs/mscoco/real_base/inference/0100000
