# torchrun --nproc_per_node=8 main.py -c configs/comparison/metfaces/lognorm.yaml -p inference ckpt_path="/home/zyk/speedit/outputs/metfaces/lognorm/train/checkpoints/0040000.pt"
# torchrun --nproc_per_node=8 main.py -c configs/comparison/metfaces/lognorm.yaml -p inference ckpt_path="/home/zyk/speedit/outputs/metfaces/lognorm/train/checkpoints/0030000.pt"
# torchrun --nproc_per_node=8 main.py -c configs/comparison/metfaces/lognorm.yaml -p inference ckpt_path="/home/zyk/speedit/outputs/metfaces/lognorm/train/checkpoints/0020000.pt"

python -m pytorch_fid /home/zyk/speedit/outputs/metfaces/lognorm/inference/0020000 /home/zyk/metfaces/images
python -m pytorch_fid /home/zyk/speedit/outputs/metfaces/lognorm/inference/0030000 /home/zyk/metfaces/images
python -m pytorch_fid /home/zyk/speedit/outputs/metfaces/lognorm/inference/0040000 /home/zyk/metfaces/images
