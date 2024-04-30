torchrun --nproc_per_node=8 main.py -c configs/image/imagenet_256/mdt/baseline.yaml -p train
torchrun --nproc_per_node=8 main.py -c configs/image/imagenet_256/mdt/faster.yaml -p train
