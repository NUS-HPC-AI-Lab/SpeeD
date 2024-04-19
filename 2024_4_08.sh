torchrun --nproc_per_node=8 --master_port=30003 main.py -c configs/image/unconditional/metfaces_theory.yaml -p train


python main.py -c configs/image/text2img/mscoco_faster.yaml -p sample ckpt_path=/mnt/public/yuanzhihang/outputs/mscoco/base/train/checkpoints/0140000.pt

python main.py -c configs/image/text2img/mscoco_base.yaml -p sample ckpt_path=/mnt/public/yuanzhihang/outputs/mscoco/real_base/train/checkpoints/0140000.pt
