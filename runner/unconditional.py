import math
from time import time

import torch
import torch.distributed as dist

from speedit.networks import *
from speedit.utils.train_utils import *
from tools.common_utils import *
from tools.log_utils import *
from tools.os_utils import *

from .base import BaseExperiment


class UnconditionalExperiment(BaseExperiment):
    def __init__(self, config):
        super().__init__(config)

    def init_model_and_diffusion(self, config):
        assert config.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)"
        self.latent_size = config.image_size // 8

        model_kwargs = config.model
        model_kwargs.update({"input_size": self.latent_size})
        self.model = instantiate_from_config(**model_kwargs)
        print("initailize model with config", model_kwargs)
        print(f"\033[34m Parameters: {sum(p.numel() for p in self.model.parameters()):,}\033[0m")

        # init vae
        vae_kwargs = config.vae
        self.vae = instantiate_from_config(**vae_kwargs)
        print("initailize vae with config", vae_kwargs)

        # init diffusion
        diffusion_kwargs = config.diffusion
        self.diffusion = instantiate_from_config(**diffusion_kwargs)
        print("initailize diffusion with config", diffusion_kwargs)

    def train_one_step(self, x, y, train_steps):
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            x = self.vae.encode(x).latent_dist.sample().mul_(0.18215)
        loss_dict = self.diffusion.train_step(self.model, x, None, device=self.device, train_steps=train_steps)
        loss = loss_dict["loss"].mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        update_ema(self.ema, self.model.module)

        step_kwargs = {"train_steps": train_steps, "loss": loss}
        return step_kwargs

    def train(self):
        self.init_training()
        train_steps = int(self.start_step)
        log_steps = 0
        running_loss = 0
        start_time = time()

        epochs = self.config.epoch
        max_training_steps = self.config.get("max_training_steps", None)

        enable_sample_log = self.config.get("enable_sample_log", False)
        if enable_sample_log:
            self.config.get("sample_log_every", 1000)

        print(f"Training for {epochs} epochs...")
        for epoch in range(epochs):
            self.sampler.set_epoch(epoch)
            print(f"Beginning epoch {epoch}...")
            for x in self.loader:
                x = x.to(self.device)
                step_kwargs = self.train_one_step(x, None, train_steps)
                train_steps += 1
                log_steps += 1
                running_loss += step_kwargs["loss"].item()

                # log step
                if train_steps % self.log_every == 0 or train_steps == 1 or epoch == 0:
                    # Measure training speed:
                    torch.cuda.synchronize()
                    end_time = time()
                    steps_per_sec = log_steps / (end_time - start_time)
                    # Reduce loss history over all processes:
                    avg_loss = torch.tensor(running_loss / log_steps, device=self.device)
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / dist.get_world_size()
                    print(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                    self.tensorboard_log.log(**{"train/loss": avg_loss})
                    self.wandb_log.log(**{"train/loss": avg_loss})
                    # Reset monitoring variables:
                    running_loss = 0
                    log_steps = 0
                    start_time = time()

                if train_steps % self.ckpt_every == 0:
                    self.save_checkpoint(train_steps)
                    dist.barrier()

            if max_training_steps is not None and train_steps >= max_training_steps:
                break

        self.model.eval()  # important! This disables randomized embedding dropout
        # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
        print("Done!")
        cleanup()

    def sample_imgs(self, z, y, cfg_scale=1.0):
        model = self.model.eval()
        diffusion = self.diffusion
        vae = self.vae
        assert cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
        z = torch.cat([z, z], 0)

        samples = diffusion.sample(model, z, None, device=self.device, cfg_scale=cfg_scale)

        samples = vae.decode(samples / 0.18215).sample.detach().cpu()
        # samples = (
        #     torch.clamp(samples.mul(255).add_(0.5), 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8)
        # )
        return samples

    def sample(self):
        torch.manual_seed(self.config.seed)
        torch.set_grad_enabled(False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        # self.load_checkpoint(self.config.ckpt_path)
        self.model = self.model.to(device)
        self.vae = self.vae.to(device)

        n = 8
        z = torch.randn(n, 4, self.latent_size, self.latent_size, device=device)

        samples = self.sample_imgs(z, None)

        # save images
        from torchvision.utils import save_image

        for i, sample in enumerate(samples):
            filename = f"{self.sample_path}/{i}.png"
            save_image(sample, filename, normalize=True, value_range=(-1, 1))
            print(f"{self.sample_path}/{i}.png")
        print("Done.")

    def init_inference(self):
        path = self.config.ckpt_path
        self.load_checkpoint(path)
        self.model.eval()

        self.model = self.model.to(self.device)
        self.vae = self.vae.to(self.device)

    def inference(self):
        config = self.config
        self.init_inference()
        n = config.per_proc_batch_size
        global_batch_size = n * dist.get_world_size()
        print(f"global batch size: {global_batch_size}")
        total_samples = int(math.ceil(config.num_samples / global_batch_size) * global_batch_size)
        print(f"Total number of images that will be sampled: {total_samples}")
        assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
        samples_needed_this_gpu = int(total_samples // dist.get_world_size())
        assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
        iterations = int(samples_needed_this_gpu // n)
        pbar = range(iterations)

        from tqdm import tqdm

        pbar = tqdm(pbar) if self.rank == 0 else pbar
        total = 0

        for _ in pbar:
            z = torch.randn(n, self.model.in_channels, self.latent_size, self.latent_size, device=self.device)
            samples = self.sample_imgs(z, None)
            from torchvision.utils import save_image

            for i, sample in enumerate(samples):
                index = i * dist.get_world_size() + self.rank + total
                filename = f"{self.sample_path}/{index:06d}.png"
                save_image(sample, filename, normalize=True, value_range=(-1, 1))
            total += global_batch_size

        dist.barrier()
        if is_main_process():
            # create_npz_from_sample_folder(self.sample_path, config.num_samples)
            print("Done.")
        dist.barrier()
        dist.destroy_process_group()

    def clip_score(self):
        config = self.config
        self.init_inference()
        n = config.per_proc_batch_size
        global_batch_size = n * dist.get_world_size()
        print(f"global batch size: {global_batch_size}")
        total_samples = int(math.ceil(config.num_samples / global_batch_size) * global_batch_size)
        print(f"Total number of images that will be sampled: {total_samples}")
        assert total_samples % dist.get_world_size() == 0, "total_samples must be divisible by world_size"
        samples_needed_this_gpu = int(total_samples // dist.get_world_size())
        assert samples_needed_this_gpu % n == 0, "samples_needed_this_gpu must be divisible by the per-GPU batch size"
        iterations = int(samples_needed_this_gpu // n)
        pbar = range(iterations)

        from tqdm import tqdm

        pbar = tqdm(pbar) if self.rank == 0 else pbar
