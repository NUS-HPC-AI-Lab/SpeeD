# set ffhq dir to place holder
FFHQ_DIR="/data1/xinpeng/ffhq/images256x256"
OUTPUT_DIR="."

echo "FFHQ_DIR: $FFHQ_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"

experiment_dir=$OUTPUT_DIR/outputs/ffhq

#torchrun --nproc_per_node=8 main.py -c configs/comparison/ffhq/min_snr.yaml -p inference data.dataset.root=$FFHQ_DIR experiment_dir=$experiment_dir/min_snr ckpt_path=$experiment_dir/min_snr/train/checkpoints/0040000.pt
#torchrun --nproc_per_node=8 main.py -c configs/comparison/ffhq/min_snr.yaml -p inference data.dataset.root=$FFHQ_DIR experiment_dir=$experiment_dir/min_snr ckpt_path=$experiment_dir/min_snr/train/checkpoints/0030000.pt
#torchrun --nproc_per_node=8 main.py -c configs/comparison/ffhq/min_snr.yaml -p inference data.dataset.root=$FFHQ_DIR experiment_dir=$experiment_dir/min_snr ckpt_path=$experiment_dir/min_snr/train/checkpoints/0020000.pt
#
#torchrun --nproc_per_node=8 main.py -c configs/comparison/ffhq/min_snr.yaml -p inference data.dataset.root=$FFHQ_DIR experiment_dir=$experiment_dir/min_snr ckpt_path=$experiment_dir/min_snr/train/checkpoints/0040000.pt
#torchrun --nproc_per_node=8 main.py -c configs/comparison/ffhq/min_snr.yaml -p inference data.dataset.root=$FFHQ_DIR experiment_dir=$experiment_dir/min_snr ckpt_path=$experiment_dir/min_snr/train/checkpoints/0030000.pt
#torchrun --nproc_per_node=8 main.py -c configs/comparison/ffhq/min_snr.yaml -p inference data.dataset.root=$FFHQ_DIR experiment_dir=$experiment_dir/min_snr ckpt_path=$experiment_dir/min_snr/train/checkpoints/0020000.pt

#torchrun --nproc_per_node=8 main.py -c configs/comparison/ffhq/lognorm.yaml -p inference data.dataset.root=$FFHQ_DIR experiment_dir=$experiment_dir/lognorm ckpt_path=$experiment_dir/lognorm/train/checkpoints/0040000.pt
#torchrun --nproc_per_node=8 main.py -c configs/comparison/ffhq/lognorm.yaml -p inference data.dataset.root=$FFHQ_DIR experiment_dir=$experiment_dir/lognorm ckpt_path=$experiment_dir/lognorm/train/checkpoints/0030000.pt
#torchrun --nproc_per_node=8 main.py -c configs/comparison/ffhq/lognorm.yaml -p inference data.dataset.root=$FFHQ_DIR experiment_dir=$experiment_dir/lognorm ckpt_path=$experiment_dir/lognorm/train/checkpoints/0020000.pt
#
#torchrun --nproc_per_node=8 main.py -c configs/comparison/ffhq/clts.yaml -p inference data.dataset.root=$FFHQ_DIR experiment_dir=$experiment_dir/clts ckpt_path=$experiment_dir/clts/train/checkpoints/0040000.pt
#torchrun --nproc_per_node=8 main.py -c configs/comparison/ffhq/clts.yaml -p inference data.dataset.root=$FFHQ_DIR experiment_dir=$experiment_dir/clts ckpt_path=$experiment_dir/clts/train/checkpoints/0030000.pt
#torchrun --nproc_per_node=8 main.py -c configs/comparison/ffhq/clts.yaml -p inference data.dataset.root=$FFHQ_DIR experiment_dir=$experiment_dir/clts ckpt_path=$experiment_dir/clts/train/checkpoints/0020000.pt

#python -m pytorch_fid $FFHQ_DIR /home/zyk/fasterdit/outputs/ffhq/p2/inference/0020000
#python -m pytorch_fid $FFHQ_DIR /home/zyk/fasterdit/outputs/ffhq/p2/inference/0030000
#python -m pytorch_fid $FFHQ_DIR /home/zyk/fasterdit/outputs/ffhq/p2/inference/0040000
#python -m pytorch_fid $FFHQ_DIR /home/zyk/fasterdit/outputs/ffhq/p2/inference/0050000

#python -m pytorch_fid $FFHQ_DIR /home/zyk/fasterdit/outputs/ffhq/min_snr/inference/0020000
#python -m pytorch_fid $FFHQ_DIR /home/zyk/fasterdit/outputs/ffhq/min_snr/inference/0030000
#python -m pytorch_fid $FFHQ_DIR /home/zyk/fasterdit/outputs/ffhq/min_snr/inference/0040000
#python -m pytorch_fid $FFHQ_DIR /home/zyk/fasterdit/outputs/ffhq/min_snr/inference/0050000
#
#python -m pytorch_fid $FFHQ_DIR /home/zyk/fasterdit/outputs/ffhq/lognorm/inference/0020000
#python -m pytorch_fid $FFHQ_DIR /home/zyk/fasterdit/outputs/ffhq/lognorm/inference/0030000
#python -m pytorch_fid $FFHQ_DIR /home/zyk/fasterdit/outputs/ffhq/lognorm/inference/0040000
#python -m pytorch_fid $FFHQ_DIR /home/zyk/fasterdit/outputs/ffhq/lognorm/inference/0050000
#
#python -m pytorch_fid $FFHQ_DIR /home/zyk/fasterdit/outputs/ffhq/clts/inference/0020000
#python -m pytorch_fid $FFHQ_DIR /home/zyk/fasterdit/outputs/ffhq/clts/inference/0030000
#python -m pytorch_fid $FFHQ_DIR /home/zyk/fasterdit/outputs/ffhq/clts/inference/0040000
#python -m pytorch_fid $FFHQ_DIR /home/zyk/fasterdit/outputs/ffhq/clts/inference/0050000

#torchrun --nproc_per_node=8 main.py -c configs/comparison/ffhq/theory.yaml -p inference data.dataset.root=$FFHQ_DIR experiment_dir=$experiment_dir/theory ckpt_path=$experiment_dir/theory/train/checkpoints/0040000.pt
#torchrun --nproc_per_node=8 main.py -c configs/comparison/ffhq/theory.yaml -p inference data.dataset.root=$FFHQ_DIR experiment_dir=$experiment_dir/theory ckpt_path=$experiment_dir/theory/train/checkpoints/0030000.pt
#torchrun --nproc_per_node=8 main.py -c configs/comparison/ffhq/theory.yaml -p inference data.dataset.root=$FFHQ_DIR experiment_dir=$experiment_dir/theory ckpt_path=$experiment_dir/theory/train/checkpoints/0020000.pt

python -m pytorch_fid $FFHQ_DIR /home/zyk/fasterdit/outputs/ffhq/theory/inference/0020000
python -m pytorch_fid $FFHQ_DIR /home/zyk/fasterdit/outputs/ffhq/theory/inference/0030000
python -m pytorch_fid $FFHQ_DIR /home/zyk/fasterdit/outputs/ffhq/theory/inference/0040000
python -m pytorch_fid $FFHQ_DIR /home/zyk/fasterdit/outputs/ffhq/theory/inference/0050000
