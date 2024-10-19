 torchrun --nproc_per_node=8 main.py -c configs/image/unconditional/ffhq_base.yaml -p train data.dataset.root="/home/zyk/ffhq-dataset/images256x256" max_training_steps=50_000
 torchrun --nproc_per_node=8 main.py -c configs/image/unconditional/ffhq_base.yaml -p inference data.dataset.root="/home/zyk/ffhq-dataset/images256x256" ckpt_path="/home/zyk/speedit/outputs/ffhq/quad_base/train/checkpoints/0050000.pt"

torchrun --nproc_per_node=8 main.py -c configs/image/unconditional/ffhq_faster.yaml -p train data.dataset.root="/home/zyk/ffhq-dataset/images256x256" max_training_steps=50_000
torchrun --nproc_per_node=8 main.py -c configs/image/unconditional/ffhq_faster.yaml -p inference data.dataset.root="/home/zyk/ffhq-dataset/images256x256" ckpt_path="/home/zyk/speedit/outputs/ffhq/quad_faster/train/checkpoints/0050000.pt"
