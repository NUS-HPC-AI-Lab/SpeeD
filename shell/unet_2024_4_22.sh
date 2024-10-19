torchrun --nproc_per_node=8 main.py -c configs/comparison/unet/base.yaml -p train

torchrun --nproc_per_node=8 main.py -c configs/comparison/unet/faster.yaml -p train

torchrun --nproc_per_node=8 --master_port=30002 main.py -c configs/comparison/unet/ffhq.yaml -p train

torchrun --nproc_per_node=8 --master_port=30003 main.py -c configs/comparison/unet/ffhq_faster.yaml -p train
