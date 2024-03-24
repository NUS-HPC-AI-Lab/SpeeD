import numpy as np
import torch
import torch.nn.functional as F

from .respace import SpacedDiffusion


class SpeeDiffusion(SpacedDiffusion):
    def __init__(self, faster, **kwargs):
        super().__init__(**kwargs)
        self.faster = faster
        if faster:
            grad = np.gradient(self.sqrt_one_minus_alphas_cumprod)
            self.meaningful_steps = np.argmax(grad < 5e-5) + 1

            # p2 weighting from: Perception Prioritized Training of Diffusion Models
            self.p2_gamma = 1
            self.p2_k = 1
            self.snr = 1.0 / (1 - self.alphas_cumprod) - 1
            sqrt_one_minus_alphas_bar = torch.from_numpy(self.sqrt_one_minus_alphas_cumprod)
            p = torch.tanh(1e6 * (torch.gradient(sqrt_one_minus_alphas_bar)[0] - 1e-4)) + 1.5
            self.p = F.normalize(p, p=1, dim=0)
        else:
            self.meaningful_steps = self.num_timesteps

    def _weights(self):
        weights = 1 / (self.p2_k + self.snr) ** self.p2_gamma
        weights = weights
        return weights

    def t_sample(self, n, device):
        if self.faster:
            t = torch.multinomial(self.p, n // 2 + 1, replacement=True).to(device)
            dual_t = torch.where(t < self.meaningful_steps, self.meaningful_steps - t, t - self.meaningful_steps)
            t = torch.cat([t, dual_t], dim=0)[:n]
            weights = self._weights()
        else:
            t = torch.randint(0, self.num_timesteps, (n,), device=device)
            weights = None

        return t, weights
