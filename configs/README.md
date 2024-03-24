# Loading config

The code of config loading can find in  [SpeeDiT/tools/config_utils.py](https://github.com/1zeryu/SpeeDiT/blob/master/tools/config_utils.py).

## Recursive strategy

We use a recursive loading strategy. For an example, if we implement experiment using command line.

```
torchrun --nproc_per_node=8 main.py -c configs/image/imagenet_256/dit_s2.yaml -p train
```

The config structure as follows:

```
configs/
--base.yaml
--image/
  --imagenet_256/
	--base.yaml
	--dit_s2.yaml
```

It load from config root path "configs/", and search for **base.yaml** in that directory. If yes, loading it and recursive search for next directory. When you reach the last level of the directory, search for **base.yaml** and finally read in the specified config file.

**inherit**: The config  file load at the later level can inherit the settings at the previous level.

**Override**: The config file load  at the later level can override the settings at the previous level.

The loading order in the example is

1. configs/base.yaml
2. configs/image/imagenet_256/base.yaml
3. configs/image/imagenet_256/dit_s2.yaml

## Command line rewriting

The setting can modify by command line by form "a=b". For an example,  change classifier-free guidance scale in sampling by command line:

```
python main.py -c configs/image/imagenet_256/base.yaml -p sample guidance_scale=1.5
```

## Override by phase config

Sometimes, an experiment has different settings in training, inference, and sampling.  you can write the phase config in config file. The phase config overrides the global config when the phase is executed.

For example,

```
# configs/image/imagenet_256/base.yaml
cfg_scale: 1.5
sample:
  guidance_scale: 3.8
  diffusion:
    timestep_respacing: '250'
```

if you run with "-p sample", the setting of diffusion and guidance_scale will override by sample.diffusion and sample.guidance_scale.
