from functools import partial

import numpy as np
import torch
from torch.nn import functional as F

import speedit.diffusion.iddpm.gaussian_diffusion as gd
from speedit.diffusion.iddpm.respace import SpacedDiffusion, space_timesteps


class IDDPM(SpacedDiffusion):
    def __init__(
        self,
        timestep_respacing,
        noise_schedule="linear",
        use_kl=False,
        sigma_small=False,
        predict_xstart=False,
        learn_sigma=True,
        rescale_learned_sigmas=False,
        diffusion_steps=1000,
        cfg_scale=4.0,
    ):
        betas = gd.get_named_beta_schedule(noise_schedule, diffusion_steps)
        if use_kl:
            loss_type = gd.LossType.RESCALED_KL
        elif rescale_learned_sigmas:
            loss_type = gd.LossType.RESCALED_MSE
        else:
            loss_type = gd.LossType.MSE
        if timestep_respacing is None or timestep_respacing == "":
            timestep_respacing = [diffusion_steps]

        super().__init__(
            use_timesteps=space_timesteps(diffusion_steps, timestep_respacing),
            betas=betas,
            model_mean_type=(gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X),
            model_var_type=(
                (gd.ModelVarType.FIXED_LARGE if not sigma_small else gd.ModelVarType.FIXED_SMALL)
                if not learn_sigma
                else gd.ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type,
        )
        self.cfg_scale = cfg_scale

        grad = np.gradient(self.sqrt_one_minus_alphas_cumprod)

        # set the meaningful steps in diffusion, which is more important in inference
        self.meaningful_steps = np.argmax(grad < 1e-4) + 1

        # p2 weighting from: Perception Prioritized Training of Diffusion Models
        self.p2_gamma = 1
        self.p2_k = 1
        self.snr = 1.0 / (1 - self.alphas_cumprod) - 1
        sqrt_one_minus_alphas_bar = torch.from_numpy(self.sqrt_one_minus_alphas_cumprod)
        # sample more meaningful step
        p = torch.tanh(1e6 * (torch.gradient(sqrt_one_minus_alphas_bar)[0] - 1e-4)) + 1.5
        self.p = F.normalize(p, p=1, dim=0)
        self.weights = self._weights()

    def _weights(self):
        # process where all noise to noisy image with content has more weighting in training
        # the weights act on the mse loss
        weights = 1 / (self.p2_k + self.snr) ** self.p2_gamma
        weights = weights
        return weights

    def train_step(self, model, x, model_kwargs, device):
        n = x.shape[0]
        t = torch.multinomial(self.p, n // 2 + 1, replacement=True).to(device)
        dual_t = torch.where(t < self.meaningful_steps, self.meaningful_steps - t, t - self.meaningful_steps)
        t = torch.cat([t, dual_t], dim=0)[:n]
        weights = self.weights
        loss_dict = self.training_losses(model, x, t, model_kwargs, weights)
        return loss_dict

    def sample(
        self,
        model,
        z,
        model_args,
        device,
        cfg_scale=None,
    ):
        cfg_scale = cfg_scale or self.cfg_scale

        forward = partial(forward_with_cfg, model, cfg_scale=cfg_scale)
        samples = self.p_sample_loop(
            forward,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_args,
            progress=True,
            device=device,
        )
        samples, _ = samples.chunk(2, dim=0)
        return samples


def forward_with_cfg(model, x, timestep, y, cfg_scale, **kwargs):
    # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
    half = x[: len(x) // 2]
    combined = torch.cat([half, half], dim=0)
    model_out = model.forward(combined, timestep, y, **kwargs)
    model_out = model_out["x"] if isinstance(model_out, dict) else model_out
    eps, rest = model_out[:, :3], model_out[:, 3:]
    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    eps = torch.cat([half_eps, half_eps], dim=0)
    return torch.cat([eps, rest], dim=1)
