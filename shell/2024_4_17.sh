#FFHQ_DIR=""
#METFACES_DIR=""

torchrun --nproc_per_node=8 main.py -c configs/comparison/metfaces/p2.yaml -p train
torchrun --nproc_per_node=8 main.py -c configs/comparison/metfaces/min_snr.yaml -p train

torchrun --nproc_per_node=8 main.py -c configs/comparison/ffhq/p2.yaml -p train
torchrun --nproc_per_node=8 main.py -c configs/comparison/ffhq/min_snr.yaml -p train


# data in: /home/zyk/metfaces/images
