import argparse

from pytorch_fid import fid_score

parser = argparse.ArgumentParser(description="PyTorch Inception Score implementation")
parser.add_argument("--real", type=str, default="", help="path to the real images")
parser.add_argument("--fake", type=str, default="", help="path to the generated images")

args = parser.parse_args()

sfid = fid_score.main(_path1=args.real, _path2=args.fake)
print("sfid: ", sfid)
