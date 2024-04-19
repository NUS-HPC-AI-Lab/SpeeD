import builtins
import math
import os
from copy import deepcopy
from time import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from speedit.networks import *
from speedit.utils.train_utils import *
from tools.common_utils import *
from tools.log_utils import *
from tools.os_utils import *


class BaseExperiment(object):
    def __init__(self, config):
        self.init_device_seed(config)
        self._init_config(config)
        self.init_model_and_diffusion(config)
        self.init_task(config)

    def init_task(self, config):
        self.num_classes = config.get("num_classes", 1)

    def init_device_seed(self, config):
        if config.phase in ["train", "inference"]:
            assert torch.cuda.is_available(), f"{config.phase} currently requires at least one GPU."
            # Setup DDP:
            dist.init_process_group("nccl")
            self.rank = dist.get_rank()
            self.device = self.rank % torch.cuda.device_count()
            self.seed = config.seed * dist.get_world_size() + self.rank
            torch.manual_seed(self.seed)
            torch.cuda.set_device(self.device)
            builtins.print(f"Starting rank={self.rank}, seed={self.seed}, world_size={dist.get_world_size()}.")

    def _init_config(self, config):
        self.config = config
        self.output_dir = os.path.join(config.experiment_dir, config.phase)
        init_working_space(self.output_dir)
        if config.phase == "train":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.checkpoint_path = os.path.join(self.output_dir, "checkpoints")
            self.log_path = os.path.join(self.output_dir, "logs")
            init_working_space(self.checkpoint_path)
            init_working_space(self.log_path)

        elif config.phase == "inference" or config.phase == "sample":
            ckpt_file_name = os.path.splitext(os.path.basename(config.ckpt_path))
            self.sample_path = os.path.join(self.output_dir, ckpt_file_name[0])
            init_working_space(self.sample_path)
            torch.backends.cuda.matmul.allow_tf32 = config.get("allow_tf32", True)
            torch.set_grad_enabled(False)

        save_config(config, self.output_dir)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    def init_log(self, config):
        self.tensorboard_log = Tensorboard_log(config, path=self.log_path)
        self.wandb_log = Wandb_log(config)
        self.ckpt_every = config.ckpt_every
        self.log_every = config.log_every

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

        # init condition model
        encoder_kwargs = config.condition_encoder
        self.encoder = instantiate_from_config(**encoder_kwargs)
        print("initailize encoder with config", encoder_kwargs)

        # init diffusion
        diffusion_kwargs = config.diffusion
        self.diffusion = instantiate_from_config(**diffusion_kwargs)
        print("initailize diffusion with config", diffusion_kwargs)

    def init_dataset(self):
        data_config = self.config.data
        dataset_kwargs = data_config.dataset
        self.dataset = instantiate_from_config(**dataset_kwargs)

        self.sampler = DistributedSampler(
            self.dataset,
            num_replicas=dist.get_world_size(),
            rank=self.rank,
            shuffle=True,
            seed=self.seed,
        )

        batch_size = data_config.batch_size
        num_workers = data_config.num_workers
        global_batch_size = batch_size * dist.get_world_size()

        self.loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=self.sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        print("initialize dataloader with config", data_config)
        print("\033[34m the data global batch size is \033[0m", global_batch_size)
        print(f"\033[34m Dataset contains {len(self.dataset):,} \033[0m")

    def resume_training(self, path):
        if os.path.isfile(path):
            print("loading model from checkpoint '{}'".format(path))
            state = torch.load(path, map_location="cpu")
            self.model.load_state_dict(state["model"])
            self.ema.load_state_dict(state["ema"])
            self.opt.load_state_dict(state["opt"])

            # convert optimizer to cuda
            for state in self.opt.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

            self.start_step = state["step"].item()
        else:
            raise ValueError("no checkpoint found at {}".format(path))

    def save_checkpoint(self, train_steps):
        if is_main_process():
            checkpoint = {
                "model": self.model.module.state_dict(),
                "ema": self.ema.state_dict(),
                "opt": self.opt.state_dict(),
                "args": self.config,
                "step": train_steps,
            }
            checkpoint_path = os.path.join(self.checkpoint_path, f"{train_steps:07d}.pt")
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        dist.barrier()

    def load_checkpoint(self, path):
        if os.path.isfile(path):
            print("loading model from checkpoint '{}'".format(path))
            state = torch.load(path, map_location="cpu")
            self.model.load_state_dict(state["ema"])
        else:
            raise ValueError("no checkpoint found at {}".format(path))

    def init_training(self):
        config = self.config
        self.init_log(config)

        self.ema = deepcopy(self.model)

        opt_kwargs = {"params": self.model.parameters(), **config.optimizer}
        self.opt = instantiate_from_config(**opt_kwargs)
        print("initailize optimizer with config", opt_kwargs)

        # prepare for data
        self.init_dataset()
        self.start_step = 0

        if self.config.get("resume_training", None) is not None:
            self.resume_training(self.config.resume_training)

        self.ema = self.ema.to(self.device)
        requires_grad(self.ema, False)
        self.model = DDP(self.model.to(self.device), device_ids=[self.rank])
        self.vae = self.vae.to(self.device)
        # self.encoder = self.encoder.to(self.device)

        update_ema(self.ema, self.model.module, decay=0)
        self.model.train()
        self.ema.eval()

    def train_one_step(self, x, y, train_steps):
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            x = self.vae.encode(x).latent_dist.sample().mul_(0.18215)
            model_kwargs = self.encoder.encode(y)
        loss_dict = self.diffusion.train_step(self.model, x, model_kwargs, device=self.device)
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

        print(f"Training for {epochs} epochs...")
        for epoch in range(epochs):
            self.sampler.set_epoch(epoch)
            print(f"Beginning epoch {epoch}...")
            for x, y in self.loader:
                x = x.to(self.device)
                step_kwargs = self.train_one_step(x, y, train_steps)
                train_steps += 1
                log_steps += 1
                running_loss += step_kwargs["loss"].item()

                # log step
                if train_steps % self.log_every == 0 or train_steps == 1:
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

                if train_steps % self.ckpt_every == 0 or train_steps == 1:
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
        n = z.shape[0]
        assert cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
        z = torch.cat([z, z], 0)
        model_args = self.encoder.encode(y)
        y_null = self.encoder.null(n)
        model_args["y"] = torch.cat([model_args["y"], y_null], 0)

        samples = diffusion.sample(model, z, model_args, device=self.device, cfg_scale=cfg_scale)

        samples = vae.decode(samples / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        return samples

    def sample(self):
        torch.manual_seed(self.config.seed)
        torch.set_grad_enabled(False)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.load_checkpoint(self.config.ckpt_path)
        self.model = self.model.to(device)
        self.vae = self.vae.to(device)

        sample_classes = self.config.sample_classes or 0
        n = len(sample_classes)
        z = torch.randn(n, 4, self.latent_size, self.latent_size, device=device)
        y = torch.tensor(sample_classes, device=device)

        cfg_scale = self.config.guidance_scale
        print("sampling with guidance scale:", cfg_scale)

        samples = self.sample_imgs(z, y, cfg_scale)

        # save images
        for i, sample in enumerate(samples):
            filename = f"{self.sample_path}/{i}.png"
            Image.fromarray(sample).save(filename)
            print(f"{self.sample_path}/{i}.png")
        print("Done.")

    def init_inference(self):
        path = self.config.ckpt_path
        self.load_checkpoint(path)
        self.model.eval()

        self.model = self.model.to(self.device)
        self.vae = self.vae.to(self.device)
        self.encoder = self.encoder.to(self.device)

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

        cfg_scale = self.config.guidance_scale
        print("sampling with guidance scale:", cfg_scale)

        for _ in pbar:
            z = torch.randn(n, self.model.in_channels, self.latent_size, self.latent_size, device=self.device)
            y = torch.randint(0, self.num_classes, (n,), device=self.device)
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
