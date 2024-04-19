#torchrun --nproc_per_node=8 --master_port=30001 main.py -c configs/image/unconditional/metfaces_512/faster.yaml -p train

#torchrun --nproc_per_node=8 --master_port=30001 main.py -c configs/image/unconditional/metfaces_512/baseline.yaml -p train

#python main.py -c configs/image/unconditional/metfaces_512/baseline.yaml -p sample ckpt_path="/mnt/public/yuanzhihang/outputs/metfaces_512/faster/train/checkpoints/0020000.pt"
#python main.py -c configs/image/unconditional/metfaces_512/baseline.yaml -p sample ckpt_path="/mnt/public/yuanzhihang/outputs/metfaces_512/faster/train/checkpoints/0040000.pt"
#python main.py -c configs/image/unconditional/metfaces_512/baseline.yaml -p sample ckpt_path="/mnt/public/yuanzhihang/outputs/metfaces_512/faster/train/checkpoints/0060000.pt"
#python main.py -c configs/image/unconditional/metfaces_512/baseline.yaml -p sample ckpt_path="/mnt/public/yuanzhihang/outputs/metfaces_512/faster/train/checkpoints/0080000.pt"
#python main.py -c configs/image/unconditional/metfaces_512/baseline.yaml -p sample ckpt_path="/mnt/public/yuanzhihang/outputs/metfaces_512/faster/train/checkpoints/0100000.pt"


#python main.py -c configs/image/unconditional/metfaces_512/baseline.yaml -p sample ckpt_path="/mnt/public/yuanzhihang/outputs/metfaces_512/base/train/checkpoints/0020000.pt"
#python main.py -c configs/image/unconditional/metfaces_512/baseline.yaml -p sample ckpt_path="/mnt/public/yuanzhihang/outputs/metfaces_512/base/train/checkpoints/0040000.pt"
#python main.py -c configs/image/unconditional/metfaces_512/baseline.yaml -p sample ckpt_path="/mnt/public/yuanzhihang/outputs/metfaces_512/base/train/checkpoints/0060000.pt"
#python main.py -c configs/image/unconditional/metfaces_512/baseline.yaml -p sample ckpt_path="/mnt/public/yuanzhihang/outputs/metfaces_512/base/train/checkpoints/0080000.pt"
#python main.py -c configs/image/unconditional/metfaces_512/baseline.yaml -p sample ckpt_path="/mnt/public/yuanzhihang/outputs/metfaces_512/base/train/checkpoints/0100000.pt"

#torchrun --nproc_per_node=8 main.py -c configs/image/unconditional/metfaces_512/faster.yaml -p inference ckpt_path="/mnt/public/yuanzhihang/outputs/metfaces_512/faster/train/checkpoints/0040000.pt"
#torchrun --nproc_per_node=8 main.py -c configs/image/unconditional/metfaces_512/baseline.yaml -p inference ckpt_path="/mnt/public/yuanzhihang/outputs/metfaces_512/base/train/checkpoints/0040000.pt"
#
#
#torchrun --nproc_per_node=8 main.py -c configs/image/unconditional/metfaces_512/faster.yaml -p inference ckpt_path="/mnt/public/yuanzhihang/outputs/metfaces_512/faster/train/checkpoints/0100000.pt"
#torchrun --nproc_per_node=8 main.py -c configs/image/unconditional/metfaces_512/baseline.yaml -p inference ckpt_path="/mnt/public/yuanzhihang/outputs/metfaces_512/base/train/checkpoints/0100000.pt"

#python -m pytorch_fid /mnt/public/yuanzhihang/outputs/metfaces_512/faster/inference/0040000 /mnt/public/yuanzhihang/metfaces
python -m pytorch_fid /mnt/public/yuanzhihang/outputs/metfaces_512/faster/inference/0100000 /mnt/public/yuanzhihang/metfaces
#python -m pytorch_fid /mnt/public/yuanzhihang/outputs/metfaces_512/base/inference/0040000 /mnt/public/yuanzhihang/metfaces
python -m pytorch_fid /mnt/public/yuanzhihang/outputs/metfaces_512/base/inference/0100000 /mnt/public/yuanzhihang/metfaces

#python
