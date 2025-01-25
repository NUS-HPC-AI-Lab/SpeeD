import math
import random

import torch
import torch.distributed as dist
from PIL import Image

from speed.utils.train_utils import *
from tools.log_utils import *
from tools.os_utils import *

from .base import BaseExperiment


class Text2ImgExperiment(BaseExperiment):
    def __init__(self, config):
        super(Text2ImgExperiment, self).__init__(config)

    def sample(self):
        torch.manual_seed(self.config.seed)
        torch.set_grad_enabled(False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.load_checkpoint(self.config.ckpt_path)
        self.model = self.model.to(device)
        self.vae = self.vae.to(device)
        self.encoder.y_embedder = self.model.y_embedder
        prompts = self.config.prompts
        n = len(prompts)
        z = torch.randn(n, 4, self.latent_size, self.latent_size, device=device)
        y = list(prompts)

        cfg_scale = self.config.guidance_scale
        print("sampling with guidance scale:", cfg_scale)

        samples = self.sample_imgs(z, y, cfg_scale)

        # save images
        for i, sample in enumerate(samples):
            filename = f"{self.sample_path}/{i}.png"
            Image.fromarray(sample).save(filename)
            print(f"{self.sample_path}/{i}.png")
        print("Done.")

    def inference(self):
        config = self.config
        self.init_inference()
        n = config.per_proc_batch_size
        global_batch_size = n * dist.get_world_size()

        prompt_file = config.prompt_path
        prompt_list = read_prompt_file_to_list(prompt_file)
        total_samples = int(math.ceil(config.num_samples / global_batch_size) * global_batch_size)
        print(f"Total number of images that will be sampled: {total_samples}")

        print(f"global batch size: {global_batch_size}")
        print(f"Total number of prompt: {len(prompt_list)}")
        assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
        samples_needed_this_gpu = int(total_samples // dist.get_world_size())
        assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
        iterations = int(samples_needed_this_gpu // n)
        pbar = range(iterations)

        from tqdm import tqdm

        self.encoder.y_embedder = self.model.y_embedder

        pbar = tqdm(pbar, desc="Sampling")
        total = 0

        cfg_scale = self.config.guidance_scale
        print("sampling with guidance scale:", cfg_scale)

        for i in pbar:
            # random select n prompts from prompt_list
            y = random.choices(prompt_list, k=n)
            z = torch.randn(n, self.model.in_channels, self.latent_size, self.latent_size, device=self.device)
            samples = self.sample_imgs(z, y, cfg_scale)
            for i, sample in enumerate(samples):
                index = i * dist.get_world_size() + self.rank + total
                Image.fromarray(sample).save(f"{self.sample_path}/{index:06d}.png")
            total += global_batch_size

        dist.barrier()
        if is_main_process():
            create_npz_from_sample_folder(self.sample_path, config.num_samples)
            print("Done.")
        dist.barrier()
        dist.destroy_process_group()
