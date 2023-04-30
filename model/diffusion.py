import random
import numpy as np
import torch
from torch import nn

from .helpers import (
    cosine_beta_schedule,
    extract,
    condition_projection,
    Losses,
)


class GaussianDiffusion(nn.Module):
    def __init__(self, model, horizon, observation_dim, action_dim, class_dim, n_timesteps=200,
                 loss_type='Weighted_MSE', clip_denoised=False, ddim_discr_method='uniform',
                 ):
        super().__init__()
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.class_dim = class_dim
        self.model = model

        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = n_timesteps
        self.clip_denoised = clip_denoised
        self.eta = 0.0
        self.random_ratio = 1.0

        # ---------------------------ddim--------------------------------
        ddim_timesteps = 10

        if ddim_discr_method == 'uniform':
            c = n_timesteps // ddim_timesteps
            ddim_timestep_seq = np.asarray(list(range(0, n_timesteps, c)))
        elif ddim_discr_method == 'quad':
            ddim_timestep_seq = (
                    (np.linspace(0, np.sqrt(n_timesteps), ddim_timesteps)) ** 2
            ).astype(int)
        else:
            assert RuntimeError()

        self.ddim_timesteps = ddim_timesteps
        self.ddim_timestep_seq = ddim_timestep_seq
        # ----------------------------------------------------------------

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        self.loss_type = loss_type
        self.loss_fn = Losses[loss_type](None, self.action_dim, self.class_dim)

    # ------------------------------------------ sampling ------------------------------------------#

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, cond, t):
        x_recon = self.model(x, t)

        if self.clip_denoised:
            x_recon.clamp(-1., 1.)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return \
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) \
            / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    @torch.no_grad()
    def p_sample_ddim(self, x, cond, t, t_prev, if_prev=False):
        b, *_, device = *x.shape, x.device
        x_recon = self.model(x, t)

        if self.clip_denoised:
            x_recon.clamp(-1., 1.)
        else:
            assert RuntimeError()

        eps = self._predict_eps_from_xstart(x, t, x_recon)
        alpha_bar = extract(self.alphas_cumprod, t, x.shape)
        if if_prev:
            alpha_bar_prev = extract(self.alphas_cumprod_prev, t_prev, x.shape)
        else:
            alpha_bar_prev = extract(self.alphas_cumprod, t_prev, x.shape)
        sigma = (
                self.eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
        )

        noise = torch.randn_like(x) * self.random_ratio
        mean_pred = (
                x_recon * torch.sqrt(alpha_bar_prev)
                + torch.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps
        )

        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return mean_pred + nonzero_mask * sigma * noise

    @torch.no_grad()
    def p_sample(self, x, cond, t):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, cond=cond, t=t)
        noise = torch.randn_like(x) * self.random_ratio
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, cond, if_jump):
        device = self.betas.device
        batch_size = len(cond[0])
        horizon = self.horizon
        shape = (batch_size, horizon, self.class_dim + self.action_dim + self.observation_dim)

        x = torch.randn(shape, device=device) * self.random_ratio  # xt for Noise and diffusion
        # x = torch.zeros(shape, device=device)   # for Deterministic
        x = condition_projection(x, cond, self.action_dim, self.class_dim)

        '''
        The if-else below is for diffusion, should be removed for Noise and Deterministic
        '''
        if not if_jump:
            for i in reversed(range(0, self.n_timesteps)):
                timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
                x = self.p_sample(x, cond, timesteps)
                x = condition_projection(x, cond, self.action_dim, self.class_dim)

        else:
            for i in reversed(range(0, self.ddim_timesteps)):
                timesteps = torch.full((batch_size,), self.ddim_timestep_seq[i], device=device, dtype=torch.long)
                if i == 0:
                    timesteps_prev = torch.full((batch_size,), 0, device=device, dtype=torch.long)
                    x = self.p_sample_ddim(x, cond, timesteps, timesteps_prev, True)
                else:
                    timesteps_prev = torch.full((batch_size,), self.ddim_timestep_seq[i-1], device=device, dtype=torch.long)
                    x = self.p_sample_ddim(x, cond, timesteps, timesteps_prev)
                x = condition_projection(x, cond, self.action_dim, self.class_dim)

        '''
        The two lines below is for Noise and Deterministic
        '''
        # x = self.model(x, None)
        # x = condition_projection(x, cond, self.action_dim, self.class_dim)

        return x

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start) * self.random_ratio

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, cond, t):
        noise = torch.randn_like(x_start) * self.random_ratio   # for Noise and diffusion
        # noise = torch.zeros_like(x_start)   # for Deterministic
        # x_noisy = noise   # for Noise and Deterministic

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)   # for diffusion, should be removed for Noise and Deterministic
        x_noisy = condition_projection(x_noisy, cond, self.action_dim, self.class_dim)

        x_recon = self.model(x_noisy, t)
        x_recon = condition_projection(x_recon, cond, self.action_dim, self.class_dim)

        loss = self.loss_fn(x_recon, x_start)
        return loss

    def loss(self, x, cond):
        batch_size = len(x)   # for diffusion
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()   # for diffusion
        # t = None    # for Noise and Deterministic
        return self.p_losses(x, cond, t)

    def forward(self, cond, if_jump=False):
        return self.p_sample_loop(cond, if_jump)
