python -m pytorch_fid /mnt/public/yuanzhihang/ffhq/images256x256 /mnt/public/yuanzhihang/outputs/ffhq/faster/inference/0080000
python -m pytorch_fid /mnt/public/yuanzhihang/ffhq/images256x256 /mnt/public/yuanzhihang/outputs/ffhq/base/inference/0080000
python -m pytorch_fid /mnt/public/yuanzhihang/ffhq/images256x256 /mnt/public/yuanzhihang/outputs/ffhq/faster/inference/0070000
python -m pytorch_fid /mnt/public/yuanzhihang/ffhq/images256x256 /mnt/public/yuanzhihang/outputs/ffhq/base/inference/0070000
#python -m pytorch_fid /mnt/public/yuanzhihang/ffhq/images256x256 /mnt/public/yuanzhihang/outputs/ffhq/faster/inference/0040000
#python -m pytorch_fid /mnt/public/yuanzhihang/ffhq/images256x256 /mnt/public/yuanzhihang/outputs/ffhq/base/inference/0040000
#python -m pytorch_fid /mnt/public/yuanzhihang/ffhq/images256x256 /mnt/public/yuanzhihang/outputs/ffhq/faster/inference/0030000
#python -m pytorch_fid /mnt/public/yuanzhihang/ffhq/images256x256 /mnt/public/yuanzhihang/outputs/ffhq/base/inference/0030000
#python -m pytorch_fid /mnt/public/yuanzhihang/ffhq/images256x256 /mnt/public/yuanzhihang/outputs/ffhq/faster/inference/0020000
#python -m pytorch_fid /mnt/public/yuanzhihang/ffhq/images256x256 /mnt/public/yuanzhihang/outputs/ffhq/base/inference/0020000
#python -m pytorch_fid /mnt/public/yuanzhihang/ffhq/images256x256 /mnt/public/yuanzhihang/outputs/ffhq/faster/inference/0010000
#python -m pytorch_fid /mnt/public/yuanzhihang/ffhq/images256x256 /mnt/public/yuanzhihang/outputs/ffhq/base/inference/0010000


# 6:21
torchrun --nproc_per_node=8 --master_port=30003 main.py -c configs/image/text2img/mscoco_faster.yaml -p train
torchrun --nproc_per_node=8 --master_port=30003 main.py -c configs/image/text2img/mscoco_base.yaml -p train
