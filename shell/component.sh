
EXP="no_dual"
#torchrun --nproc_per_node=8 main.py -c configs/ablation/component/$EXP.yaml -p train
#torchrun --nproc_per_node=8 main.py -c configs/ablation/component/$EXP.yaml -p inference ckpt_path="outputs/component/no_dual/train/checkpoints/0050000.pt"
#python -m pytorch_fid outputs/abl/$EXP/inference/0050000/ /data1/xinpeng/ffhq/images256x256
#fidelity --gpu 0 --isc --input1 outputs/abl/$EXP/inference/0050000/


EXP="only_sample"
#torchrun --nproc_per_node=8 main.py -c configs/ablation/component/$EXP.yaml -p train
#torchrun --nproc_per_node=8 main.py -c configs/ablation/component/$EXP.yaml -p inference ckpt_path="outputs/component/$EXP/train/checkpoints/0050000.pt"
#python -m pytorch_fid outputs/component/$EXP/inference/0050000/ /data1/xinpeng/ffhq/images256x256
#fidelity --gpu 0 --isc --input1 outputs/component/$EXP/inference/0050000/

EXP="weight"
#torchrun --nproc_per_node=8 main.py -c configs/ablation/component/$EXP.yaml -p train
#torchrun --nproc_per_node=8 main.py -c configs/ablation/component/$EXP.yaml -p inference ckpt_path="outputs/component/$EXP/train/checkpoints/0050000.pt"
python -m pytorch_fid outputs/component/$EXP/inference/0050000/ /data1/xinpeng/ffhq/images256x256
fidelity --gpu 0 --isc --input1 outputs/component/$EXP/inference/0050000/
#
EXP="no_weight"
torchrun --nproc_per_node=8 main.py -c configs/ablation/component/$EXP.yaml -p train
torchrun --nproc_per_node=8 main.py -c configs/ablation/component/$EXP.yaml -p inference ckpt_path="outputs/component/$EXP/train/checkpoints/0050000.pt"
python -m pytorch_fid outputs/component/$EXP/inference/0050000/ /data1/xinpeng/ffhq/images256x256
fidelity --gpu 0 --isc --input1 outputs/component/$EXP/inference/0050000/
