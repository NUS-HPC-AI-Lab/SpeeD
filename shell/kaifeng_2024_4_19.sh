# set ffhq dir to place holder
FFHQ_DIR="/data1/xinpeng/ffhq/images256x256"
OUTPUT_DIR="."

echo "FFHQ_DIR: $FFHQ_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"

experiment_dir=$OUTPUT_DIR/outputs/ffhq

##torchrun --nproc_per_node=8 main.py -c configs/comparison/ffhq/p2.yaml -p train data.dataset.root=$FFHQ_DIR experiment_dir=$experiment_dir/p2
#torchrun --nproc_per_node=8 main.py -c configs/comparison/ffhq/p2.yaml -p inference data.dataset.root=$FFHQ_DIR experiment_dir=$experiment_dir/p2 ckpt_path=$experiment_dir/p2/train/checkpoints/0040000.pt
#torchrun --nproc_per_node=8 main.py -c configs/comparison/ffhq/p2.yaml -p inference data.dataset.root=$FFHQ_DIR experiment_dir=$experiment_dir/p2 ckpt_path=$experiment_dir/p2/train/checkpoints/0030000.pt
#torchrun --nproc_per_node=8 main.py -c configs/comparison/ffhq/p2.yaml -p inference data.dataset.root=$FFHQ_DIR experiment_dir=$experiment_dir/p2 ckpt_path=$experiment_dir/p2/train/checkpoints/0020000.pt
#torchrun --nproc_per_node=8 main.py -c configs/comparison/ffhq/p2.yaml -p inference data.dataset.root=$FFHQ_DIR experiment_dir=$experiment_dir/p2 ckpt_path=$experiment_dir/p2/train/checkpoints/0010000.pt

#torchrun --nproc_per_node=8 main.py -c configs/comparison/ffhq/min_snr.yaml -p train train data.dataset.root=$FFHQ_DIR experiment_dir=$experiment_dir/min_snr
#torchrun --nproc_per_node=8 main.py -c configs/comparison/ffhq/min_snr.yaml -p inference data.dataset.root=$FFHQ_DIR experiment_dir=$experiment_dir/min_snr ckpt_path=$experiment_dir/min_snr/train/checkpoints/0050000.pt
torchrun --nproc_per_node=8 main.py -c configs/comparison/ffhq/min_snr.yaml -p inference data.dataset.root=$FFHQ_DIR experiment_dir=$experiment_dir/min_snr ckpt_path=$experiment_dir/min_snr/train/checkpoints/0040000.pt
torchrun --nproc_per_node=8 main.py -c configs/comparison/ffhq/min_snr.yaml -p inference data.dataset.root=$FFHQ_DIR experiment_dir=$experiment_dir/min_snr ckpt_path=$experiment_dir/min_snr/train/checkpoints/0030000.pt
torchrun --nproc_per_node=8 main.py -c configs/comparison/ffhq/min_snr.yaml -p inference data.dataset.root=$FFHQ_DIR experiment_dir=$experiment_dir/min_snr ckpt_path=$experiment_dir/min_snr/train/checkpoints/0020000.pt


#python -m pytorch_fid $experiment_dir/min_snr/inference/0050000 $FFHQ_DIR $experiment_dir/min_snr/inference/0050000.txt
#
#torchrun --nproc_per_node=8 main.py -c configs/comparison/ffhq/lognorm.yaml -p train train data.dataset.root=$FFHQ_DIR experiment_dir=$experiment_dir/lognorm
#torchrun --nproc_per_node=8 main.py -c configs/comparison/ffhq/lognorm.yaml -p inference data.dataset.root=$FFHQ_DIR experiment_dir=$experiment_dir/lognorm ckpt_path=$experiment_dir/lognorm/train/checkpoints/0050000.pt
#python -m pytorch_fid $experiment_dir/lognorm/inference/0050000 $FFHQ_DIR $experiment_dir/lognorm/inference/0050000.txt
#
#torchrun --nproc_per_node=8 main.py -c configs/comparison/ffhq/clts.yaml -p train train data.dataset.root=$FFHQ_DIR experiment_dir=$experiment_dir/clts
#torchrun --nproc_per_node=8 main.py -c configs/comparison/ffhq/clts.yaml -p inference data.dataset.root=$FFHQ_DIR experiment_dir=$experiment_dir/clts ckpt_path=$experiment_dir/clts/train/checkpoints/0050000.pt
#python -m pytorch_fid $experiment_dir/clts/inference/0050000 $FFHQ_DIR $experiment_dir/clts/inference/0050000.txt
