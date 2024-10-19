
EXP="lam_5"
##torchrun --nproc_per_node=8 main.py -c configs/ablation/lambda/$EXP.yaml -p train
#torchrun --nproc_per_node=8 main.py -c configs/ablation/lambda/$EXP.yaml -p inference ckpt_path="outputs/abl/$EXP/train/checkpoints/0030000.pt"
#python -m pytorch_fid outputs/abl/$EXP/inference/0030000/ /data1/xinpeng/ffhq/images256x256
fidelity --gpu 0 --isc --input1 outputs/abl/$EXP/inference/0030000/

#
EXP="lam_6"
#torchrun --nproc_per_node=8 main.py -c configs/ablation/lambda/$EXP.yaml -p train
#torchrun --nproc_per_node=8 main.py -c configs/ablation/lambda/$EXP.yaml -p inference ckpt_path="outputs/abl/$EXP/train/checkpoints/0030000.pt"
#python -m pytorch_fid outputs/abl/$EXP/inference/0030000/ /data1/xinpeng/ffhq/images256x256
fidelity --gpu 0 --isc --input1 outputs/abl/$EXP/inference/0030000/

#

EXP="lam_8"
#torchrun --nproc_per_node=8 main.py -c configs/ablation/lambda/$EXP.yaml -p train
#torchrun --nproc_per_node=8 main.py -c configs/ablation/lambda/$EXP.yaml -p inference ckpt_path="outputs/abl/$EXP/train/checkpoints/0030000.pt"
#python -m pytorch_fid outputs/abl/$EXP/inference/0030000/ /data1/xinpeng/ffhq/images256x256
fidelity --gpu 0 --isc --input1 outputs/abl/$EXP/inference/0030000/



EXP="lam_10"
#torchrun --nproc_per_node=8 main.py -c configs/ablation/lambda/$EXP.yaml -p train
#torchrun --nproc_per_node=8 main.py -c configs/ablation/lambda/$EXP.yaml -p inference ckpt_path="outputs/abl/$EXP/train/checkpoints/0030000.pt"
#python -m pytorch_fid outputs/abl/$EXP/inference/0030000/ /data1/xinpeng/ffhq/images256x256
fidelity --gpu 0 --isc --input1 outputs/abl/$EXP/inference/0030000/
