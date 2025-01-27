# Comparison on Celeb-A

# CUDA_VISIBLE_DEVICES=0,1 ./torchrun --nproc_per_node=2 --master_port=29966 main.py -c configs/comparison/celeba/faster.yaml -p train # a6000 env:speed
# CUDA_VISIBLE_DEVICES=4,5,6,7 ./torchrun --nproc_per_node=4 --master_port=29966 main.py -c configs/comparison/celeba/base.yaml -p train # v100 env:base
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./torchrun --nproc_per_node=4 --master_port=29955 main.py -c configs/comparison/celeba/min_snr.yaml -p train # v100 env:base
# CUDA_VISIBLE_DEVICES=4,5,6,7 ./torchrun --nproc_per_node=4 --master_port=29966 main.py -c configs/comparison/celeba/p2.yaml -p train # 4090 env:speed
# CUDA_VISIBLE_DEVICES=4,5,6,7 ./torchrun --nproc_per_node=4 --master_port=29966 main.py -c configs/comparison/celeba/lognorm.yaml -p train # 4090 env:speed
# CUDA_VISIBLE_DEVICES=4,5,6,7 ./torchrun --nproc_per_node=4 --master_port=29966 main.py -c configs/comparison/celeba/clts.yaml -p train # v100 env:base

# Ablation on k

# CUDA_VISIBLE_DEVICES=4,5,6,7 ./torchrun --nproc_per_node=4 --master_port=29966 main.py -c configs/ablation/k/k_1.yaml -p train
# CUDA_VISIBLE_DEVICES=4,5,6,7 ./torchrun --nproc_per_node=4 --master_port=29966 main.py -c configs/ablation/k/k_10.yaml -p train
# CUDA_VISIBLE_DEVICES=4,5,6,7 ./torchrun --nproc_per_node=4 --master_port=29966 main.py -c configs/ablation/k/k_25.yaml -p train


# New Baselines: DiT-S/8 on ImageNet
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./torchrun --nproc_per_node=4 --master_port=29965 main.py -c configs/comparison/imagenet/faster.yaml -p train # v100 env:base
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./torchrun --nproc_per_node=4 --master_port=29967 main.py -c configs/comparison/imagenet/base.yaml -p train # a6000 env:speed
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./torchrun --nproc_per_node=4 --master_port=29968 main.py -c configs/comparison/imagenet/min_snr.yaml -p train
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./torchrun --nproc_per_node=4 --master_port=29969 main.py -c configs/comparison/imagenet/p2.yaml -p train
# CUDA_VISIBLE_DEVICES=0,1,2,3 ./torchrun --nproc_per_node=4 --master_port=29961 main.py -c configs/comparison/imagenet/lognorm.yaml -p train
CUDA_VISIBLE_DEVICES=0,1,2,3 ./torchrun --nproc_per_node=4 --master_port=29962 main.py -c configs/comparison/imagenet/clts.yaml -p train