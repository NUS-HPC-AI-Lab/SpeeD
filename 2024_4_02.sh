torchrun --nproc_per_node=8 --master_port=30001 main.py -c configs/image/unconditional/metfaces_faster.yaml -p train data.batch_size=8
#
torchrun --nproc_per_node=8 --master_port=30002 main.py -c configs/image/unconditional/metfaces_base.yaml -p train data.batch_size=8
