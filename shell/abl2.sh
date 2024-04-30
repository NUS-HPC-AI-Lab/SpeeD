
EXP="lam_5"
torchrun --nproc_per_node=8 main.py -c configs/ablation/lambda/$EXP.yaml -p train
torchrun --nproc_per_node=8 main.py -c configs/ablation/lambda/$EXP.yaml -p inference ckpt_path="outputs/abl/$EXP/train/checkpoints/0050000.pt"
python -m pytorch_fid outputs/abl/$EXP/inference/0050000/ /home/zyk/ffhq-dataset/images256x256


EXP="lam_6"
torchrun --nproc_per_node=8 main.py -c configs/ablation/lambda/$EXP.yaml -p train
torchrun --nproc_per_node=8 main.py -c configs/ablation/lambda/$EXP.yaml -p inference ckpt_path="outputs/abl/$EXP/train/checkpoints/0050000.pt"
python -m pytorch_fid outputs/abl/$EXP/inference/0050000/ /home/zyk/ffhq-dataset/images256x256



EXP="lam_8"
torchrun --nproc_per_node=8 main.py -c configs/ablation/lambda/$EXP.yaml -p train
torchrun --nproc_per_node=8 main.py -c configs/ablation/lambda/$EXP.yaml -p inference ckpt_path="outputs/abl/$EXP/train/checkpoints/0050000.pt"
python -m pytorch_fid outputs/abl/$EXP/inference/0050000/ /home/zyk/ffhq-dataset/images256x256



EXP="lam_10"
torchrun --nproc_per_node=8 main.py -c configs/ablation/lambda/$EXP.yaml -p train
torchrun --nproc_per_node=8 main.py -c configs/ablation/lambda/$EXP.yaml -p inference ckpt_path="outputs/abl/$EXP/train/checkpoints/0050000.pt"
python -m pytorch_fid outputs/abl/$EXP/inference/0050000/ /home/zyk/ffhq-dataset/images256x256
