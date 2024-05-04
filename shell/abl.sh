
#EXP="k_1"
#torchrun --nproc_per_node=8 main.py -c configs/ablation/k/$EXP.yaml -p train
#torchrun --nproc_per_node=8 main.py -c configs/ablation/k/$EXP.yaml -p inference ckpt_path="outputs/abl/$EXP/train/checkpoints/0050000.pt"
#python -m pytorch_fid outputs/abl/$EXP/inference/0050000/ /home/zyk/ffhq-dataset/images256x256


#EXP="k_5"
#torchrun --nproc_per_node=8 main.py -c configs/ablation/k/$EXP.yaml -p train
#torchrun --nproc_per_node=8 main.py -c configs/ablation/k/$EXP.yaml -p inference ckpt_path="outputs/abl/$EXP/train/checkpoints/0050000.pt"
#python -m pytorch_fid outputs/abl/$EXP/inference/0050000/ /home/zyk/ffhq-dataset/images256x256 resume_training="/home/zyk/speedit/outputs/abl/k_5/train/checkpoints/0020000.pt"
#
#
#
#EXP="k_10"
#torchrun --nproc_per_node=8 main.py -c configs/ablation/k/$EXP.yaml -p train
#torchrun --nproc_per_node=8 main.py -c configs/ablation/k/$EXP.yaml -p inference ckpt_path="outputs/abl/$EXP/train/checkpoints/0050000.pt"
#python -m pytorch_fid outputs/abl/$EXP/inference/0050000/ /home/zyk/ffhq-dataset/images256x256
#
#
#
EXP="k_25"
#torchrun --nproc_per_node=8 main.py -c configs/ablation/k/$EXP.yaml -p train
#torchrun --nproc_per_node=8 main.py -c configs/ablation/k/$EXP.yaml -p inference ckpt_path="outputs/abl/$EXP/train/checkpoints/0050000.pt"
#python -m pytorch_fid outputs/abl/$EXP/inference/0050000/ /home/zyk/ffhq-dataset/images256x256
fidelity --gpu 0 --isc --input1 outputs/abl/$EXP/inference/0050000/
