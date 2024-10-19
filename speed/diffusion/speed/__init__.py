from functools import partial

import numpy as np
import torch
from torch.nn import functional as F

import speed.diffusion.iddpm.gaussian_diffusion as gd
from speed.diffusion.iddpm import IDDPM
from speed.diffusion.mask_iddpm import MASK_IDDPM


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
        weighting="ours",
        sampling="ours",
        k=1,
        lam=5,
        tau=700,
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
        # sqrt_one_minus_alphas_bar = torch.from_numpy(self.sqrt_one_minus_alphas_cumprod)
        # # sample more meaningful step
        # p = torch.tanh(1e6 * (torch.gradient(sqrt_one_minus_alphas_bar)[0] - 1e-4)) + 1.5
        self.lam = lam
        self.k = k
        self.tau = tau

        higher = self.k
        lower = 1
        p = [higher] * self.tau + [lower] * (self.num_timesteps - self.tau)
        self.p = F.normalize(torch.tensor(p, dtype=torch.float32), p=1, dim=0)

        self.weights = self._weights(weighting)
        self.sampling = sampling.lower()

    def _weights(self, weighting):
        # process where all noise to noisy image with content has more weighting in training
        # the weights act on the mse loss
        if weighting == "p2":
            weights = 1 / (self.p2_k + self.snr) ** self.p2_gamma
            weights = weights

        elif weighting == "lognorm":
            # todo: implemnt lognorm weighting from SD3
            weights = None

        elif weighting == "theory":
            weights = np.gradient(1 - self.alphas_cumprod)
            k = 0.2
            p = weights
            weights = k + (1 - 2 * k) * (p - p.min()) / (p.max() - p.min())
            return weights

        elif weighting == "min_snr":
            snr = (self.sqrt_alphas_cumprod / self.sqrt_one_minus_alphas_cumprod) ** 2
            k = 5
            min_snr = np.stack([snr, k * np.ones_like(snr)], axis=1).min(axis=1)[0] / (snr + 1)
            weights = min_snr

        elif weighting == "ours":
            weights = np.gradient(1 - self.alphas_cumprod)
            k = 1 - self.lam
            p = weights
            weights = k + (1 - 2 * k) * (p - p.min()) / (p.max() - p.min())
            weights = weights[: self.tau].tolist() + [1] * (self.num_timesteps - self.tau)
            weights = np.array(weights)
            return weights

        else:
            weights = None

        return weights

    def _sample_time(self, n, **kwargs):
        sampling = self.sampling

        if sampling == "lognorm":
            # todo: log norm sampling in SD3
            s = 1
            m = 0
            noise = self.sqrt_one_minus_alphas_cumprod
            pi = (
                (1 / (s * np.sqrt(2 * np.pi)))
                * (1 / (noise * (1 - noise)))
                * np.exp(-1 * ((np.log(noise / (1 - noise)) - m) ** 2 / 2 * s**2))
            )
            pi = torch.from_numpy(noise / (1 - noise) * pi)
            pi = F.normalize(pi, p=1, dim=0)
            t = torch.multinomial(pi, n, replacement=True)

        elif sampling == "speed":
            t = torch.multinomial(self.p, n // 2 + 1, replacement=True)
            dual_t = torch.where(t < self.meaningful_steps, self.meaningful_steps - t, t - self.meaningful_steps)
            t = torch.cat([t, dual_t], dim=0)[:n]

        elif sampling == "ours":
            t = torch.multinomial(self.p, n // 2 + 1, replacement=True)
            dual_t = torch.where(t < self.meaningful_steps, self.meaningful_steps - t, t - self.meaningful_steps)
            t = torch.cat([t, dual_t], dim=0)[:n]

        elif sampling == "uniform":
            t = torch.randint(0, self.num_timesteps, (n,))

        elif sampling == "clts":
            mu = 300
            target_steps = 50_000
            # pi = lambda * U(t) + (1 - lambda) * N(t)
            t = np.arange(self.num_timesteps)
            n_t = 1 / (self.num_timesteps * np.sqrt(2 * np.pi)) * np.exp(-((t - mu) ** 2) / 2 * self.num_timesteps**2)
            u_t = 1 / self.num_timesteps
            lam = kwargs["train_steps"] / target_steps
            pi = lam * n_t + (1 - lam) * u_t
            pi = F.normalize(torch.from_numpy(pi), p=1, dim=0)
            t = torch.multinomial(pi, n, replacement=True)

        elif sampling == "no_dual":
            t = torch.multinomial(self.p, n, replacement=True)

        else:
            raise NotImplementedError

        return t

    def train_step(self, model, z, y, device, **kwargs):
        n = z.shape[0]
        t = self._sample_time(n, **kwargs).to(device)
        weights = self.weights
        model_kwargs = y
        loss_dict = self.training_losses(model, z, t, model_kwargs, weights=weights)
        return loss_dict


class Speed_Mask_IDDPM(MASK_IDDPM):
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

        elif weighting == "theory":
            weights = np.gradient(self.sqrt_one_minus_alphas_cumprod) * self.betas
            weights = weights / weights.max()

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

    def train_step(self, model, x, y, device):
        n = x.shape[0]
        t = torch.randint(0, self.num_timesteps, (n,), device=device)
        model_kwargs = y
        weights = self.weights
        loss_dict = self.training_losses(model, x, t, model_kwargs, weights=weights)

        mask_model_kwargs = model_kwargs.copy()
        # add enable mask
        mask_model_kwargs["enable_mask"] = True
        mask_loss_dict = self.training_losses(model, x, t, mask_model_kwargs, weights=weights)

        total_loss_dict = {"loss": mask_loss_dict["loss"] + loss_dict["loss"]}
        return total_loss_dict
