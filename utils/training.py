import copy
from model.helpers import AverageMeter
from .accuracy import *


def cycle(dl):
    while True:
        for data in dl:
            yield data


class EMA():
    """
        empirical moving average
    """

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            datasetloader,
            ema_decay=0.995,
            train_lr=1e-5,
            gradient_accumulate_every=1,
            step_start_ema=400,
            update_ema_every=10,
            log_freq=100,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataloader = cycle(datasetloader)
        self.optimizer = torch.optim.AdamW(diffusion_model.parameters(), lr=train_lr, weight_decay=0.0)
        # self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, diffusion_model.parameters()), lr=train_lr, weight_decay=0.0)

        self.reset_parameters()
        self.step = 0

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    # -----------------------------------------------------------------------------#
    # ------------------------------------ api ------------------------------------#
    # -----------------------------------------------------------------------------#

    def train(self, n_train_steps, if_calculate_acc, args, scheduler):
        self.model.train()
        self.ema_model.train()
        losses = AverageMeter()
        self.optimizer.zero_grad()

        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                bs, T = batch[1].shape  # [bs, (T+1), ob_dim]
                global_img_tensors = batch[0].cuda().contiguous().float()
                img_tensors = torch.zeros((bs, T, args.class_dim + args.action_dim + args.observation_dim))
                img_tensors[:, 0, args.class_dim+args.action_dim:] = global_img_tensors[:, 0, :]
                img_tensors[:, -1, args.class_dim+args.action_dim:] = global_img_tensors[:, -1, :]
                img_tensors = img_tensors.cuda()

                video_label = batch[1].view(-1).cuda()  # [bs*T]
                task_class = batch[2].view(-1).cuda()   # [bs]

                action_label_onehot = torch.zeros((video_label.size(0), self.model.module.action_dim))
                # [bs*T, ac_dim]
                ind = torch.arange(0, len(video_label))
                action_label_onehot[ind, video_label] = 1.
                action_label_onehot = action_label_onehot.reshape(bs, T, -1).cuda()
                img_tensors[:, :, args.class_dim:args.class_dim+args.action_dim] = action_label_onehot

                task_onehot = torch.zeros((task_class.size(0), args.class_dim))
                # [bs*T, ac_dim]
                ind = torch.arange(0, len(task_class))
                task_onehot[ind, task_class] = 1.
                task_onehot = task_onehot.cuda()
                temp = task_onehot.unsqueeze(1)
                task_class_ = temp.repeat(1, T, 1)      # [bs, T, args.class_dim]
                img_tensors[:, :, :args.class_dim] = task_class_

                cond = {0: global_img_tensors[:, 0, :].float(), T - 1: global_img_tensors[:, -1, :].float(),
                        'task': task_class_}
                x = img_tensors.float()
                loss = self.model.module.loss(x, cond)
                loss = loss / self.gradient_accumulate_every
                loss.backward()
                losses.update(loss.item(), bs)

            self.optimizer.step()
            self.optimizer.zero_grad()
            scheduler.step()

            if self.step % self.update_ema_every == 0:
                self.step_ema()
            self.step += 1

        if if_calculate_acc:
            with torch.no_grad():
                output = self.ema_model(cond)
                actions_pred = output[:, :, args.class_dim:args.class_dim+self.model.module.action_dim]\
                    .contiguous().view(-1, self.model.module.action_dim)  # [bs*T, action_dim]

                (acc1, acc5), trajectory_success_rate, MIoU1, MIoU2, a0_acc, aT_acc = \
                    accuracy(actions_pred.cpu(), video_label.cpu(), topk=(1, 5),
                             max_traj_len=self.model.module.horizon)

                return torch.tensor(losses.avg), acc1, acc5, torch.tensor(trajectory_success_rate), \
                       torch.tensor(MIoU1), torch.tensor(MIoU2), a0_acc, aT_acc

        else:
            return torch.tensor(losses.avg)
