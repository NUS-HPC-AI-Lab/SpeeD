# Evalution

### Class-conditional Image generation

* Tensorflow

The metrics include FID, sFID, Precision, Recall, and Inception Score. Our calculation based on  [ADM's script](https://github.com/openai/guided-diffusion/tree/main/evaluations) which is based on Tensorflow.

You can use script in  [SpeeDiT/evaluations/image](https://github.com/1zeryu/SpeeDiT/tree/master/evaluations/image)

```bash
python evaluations/image/evaluator.py ref_npz_file sample_npz_file
```

 You can get more details about **reference file**  and **ADM's evaluation** in  [SpeeDiT/evaluations/image/ADM.md](https://github.com/1zeryu/SpeeDiT/blob/master/evaluations/image/ADM.md).

* Pytorch

We provide another method to test FiD using  [pytorch-fid](https://github.com/mseitzer/pytorch-fid)  and can get the same results as ADM's evaluation.

You need to extract the npz file into a image directory first,

```bash
python evaluations/image/uncomp.py --npz_file "you reference npz file path" --img_dir "save path for reference image"
```

Then,

```
# install pytorch_fid
pip install pytorch_fid

python -m pytorch_fid ref_image_dir inference_image_dir
```
