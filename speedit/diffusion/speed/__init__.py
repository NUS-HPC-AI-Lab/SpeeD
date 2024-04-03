from functools import partial

import numpy as np
import torch
from torch.nn import functional as F

import speedit.diffusion.iddpm.gaussian_diffusion as gd
from speedit.diffusion.iddpm import IDDPM


class Speed_IDDPM(IDDPM):
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
        weighting="p2",
        sampling="speed",
    ):
        super().__init__(
            timestep_respacing,
            noise_schedule,
            use_kl,
            sigma_small,
            predict_xstart,
            learn_sigma,
            rescale_learned_sigmas,
            diffusion_steps,
            cfg_scale,
        )

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
        self.weights = self._weights(weighting)
        self.sampling = sampling

    def _weights(self, weighting):
        # process where all noise to noisy image with content has more weighting in training
        # the weights act on the mse loss
        if weighting == "p2":
            weights = 1 / (self.p2_k + self.snr) ** self.p2_gamma
            weights = weights

        elif weighting == "lognorm":
            # todo: implemnt lognorm weighting from SD3
            weights = None

        else:
            weights = None
        return weights

    def _sample_time(self, n):
        sampling = self.sampling

        if sampling == "lognorm":
            # todo: log norm sampling in SD3
            raise NotImplementedError

        elif sampling == "speed":
            t = torch.multinomial(self.p, n // 2 + 1, replacement=True)
            dual_t = torch.where(t < self.meaningful_steps, self.meaningful_steps - t, t - self.meaningful_steps)
            t = torch.cat([t, dual_t], dim=0)[:n]
        else:
            raise ValueError(f"Unknown sampling method: {sampling}")

        return t

    def train_step(self, model, z, y, device):
        n = z.shape[0]
        t = self._sample_time(n).to(device)
        weights = self.weights
        model_kwargs = y
        loss_dict = self.training_losses(model, z, t, model_kwargs, weights=weights)
        return loss_dict
