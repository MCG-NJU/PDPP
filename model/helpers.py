import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from torch.optim.lr_scheduler import LambdaLR
import os
import numpy as np
import logging
from tensorboardX import SummaryWriter


# -----------------------------------------------------------------------------#
# ---------------------------------- modules ----------------------------------#
# -----------------------------------------------------------------------------#

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 2, 1, 0)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 2, 1, 0)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """
        Conv1d --> GroupNorm --> Mish
    """

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=32, drop_out=0.0, if_zero=False):
        super().__init__()
        if drop_out > 0.0:
            self.block = nn.Sequential(
                zero_module(
                    nn.Conv1d(inp_channels, out_channels, kernel_size, padding=1),
                ),
                Rearrange('batch channels horizon -> batch channels 1 horizon'),
                nn.GroupNorm(n_groups, out_channels),
                Rearrange('batch channels 1 horizon -> batch channels horizon'),
                nn.Mish(),
                nn.Dropout(p=drop_out),
            )
        elif if_zero:
            self.block = nn.Sequential(
                zero_module(
                    nn.Conv1d(inp_channels, out_channels, kernel_size, padding=1),
                ),
                Rearrange('batch channels horizon -> batch channels 1 horizon'),
                nn.GroupNorm(n_groups, out_channels),
                Rearrange('batch channels 1 horizon -> batch channels horizon'),
                nn.Mish(),

            )
        else:
            self.block = nn.Sequential(
                nn.Conv1d(inp_channels, out_channels, kernel_size, padding=1),
                Rearrange('batch channels horizon -> batch channels 1 horizon'),
                nn.GroupNorm(n_groups, out_channels),
                Rearrange('batch channels 1 horizon -> batch channels horizon'),
                nn.Mish(),
            )

    def forward(self, x):
        return self.block(x)


# -----------------------------------------------------------------------------#
# ---------------------------------- sampling ---------------------------------#
# -----------------------------------------------------------------------------#

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008, dtype=torch.float32):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas_clipped = np.clip(betas, a_min=0, a_max=0.999)
    return torch.tensor(betas_clipped, dtype=dtype)


def condition_projection(x, conditions, action_dim, class_dim):
    for t, val in conditions.items():
        if t != 'task':
            x[:, t, class_dim + action_dim:] = val.clone()

    x[:, 1:-1, class_dim + action_dim:] = 0.
    x[:, :, :class_dim] = conditions['task']

    return x


# -----------------------------------------------------------------------------#
# ---------------------------------- Loss -------------------------------------#
# -----------------------------------------------------------------------------#

class Weighted_MSE(nn.Module):

    def __init__(self, weights, action_dim, class_dim):
        super().__init__()
        # self.register_buffer('weights', weights)
        self.action_dim = action_dim
        self.class_dim = class_dim

    def forward(self, pred, targ):
        """
        :param pred: [B, T, task_dim+action_dim+observation_dim]
        :param targ: [B, T, task_dim+action_dim+observation_dim]
        :return:
        """

        loss_action = F.mse_loss(pred, targ, reduction='none')
        loss_action[:, 0, self.class_dim:self.class_dim + self.action_dim] *= 10.
        loss_action[:, -1, self.class_dim:self.class_dim + self.action_dim] *= 10.
        loss_action = loss_action.sum()
        return loss_action


Losses = {
    'Weighted_MSE': Weighted_MSE,
}

# -----------------------------------------------------------------------------#
# -------------------------------- lr_schedule --------------------------------#
# -----------------------------------------------------------------------------#

def get_lr_schedule_with_warmup(optimizer, num_training_steps, last_epoch=-1):
    num_warmup_steps = num_training_steps * 20 / 120
    decay_steps = num_training_steps * 30 / 120

    def lr_lambda(current_step):
        if current_step <= num_warmup_steps:
            return max(0., float(current_step) / float(max(1, num_warmup_steps)))
        else:
            return max(0.5 ** ((current_step - num_warmup_steps) // decay_steps), 0.)

    return LambdaLR(optimizer, lr_lambda, last_epoch)

# -----------------------------------------------------------------------------#
# ---------------------------------- logging ----------------------------------#
# -----------------------------------------------------------------------------#

# Taken from PyTorch's examples.imagenet.main
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger:
    def __init__(self, log_dir, n_logged_samples=10, summary_writer=SummaryWriter, if_exist=False):
        self._log_dir = log_dir
        print('logging outputs to ', log_dir)
        self._n_logged_samples = n_logged_samples
        self._summ_writer = summary_writer(log_dir, flush_secs=120, max_queue=10)
        if not if_exist:
            log = logging.getLogger(log_dir)
            if not log.handlers:
                log.setLevel(logging.DEBUG)
                if not os.path.exists(log_dir):
                    os.mkdir(log_dir)
                fh = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
                fh.setLevel(logging.INFO)
                formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
                fh.setFormatter(formatter)
                log.addHandler(fh)
            self.log = log

    def log_scalar(self, scalar, name, step_):
        self._summ_writer.add_scalar('{}'.format(name), scalar, step_)

    def log_scalars(self, scalar_dict, group_name, step, phase):
        """Will log all scalars in the same plot."""
        self._summ_writer.add_scalars('{}_{}'.format(group_name, phase), scalar_dict, step)

    def flush(self):
        self._summ_writer.flush()

    def log_info(self, info):
        self.log.info("{}".format(info))
