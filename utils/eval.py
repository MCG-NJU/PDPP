from .accuracy import *
from model.helpers import AverageMeter


def validate(val_loader, model, args):
    model.eval()
    losses = AverageMeter()
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()
    trajectory_success_rate_meter = AverageMeter()
    MIoU1_meter = AverageMeter()
    MIoU2_meter = AverageMeter()

    A0_acc = AverageMeter()
    AT_acc = AverageMeter()

    for i_batch, sample_batch in enumerate(val_loader):
        # compute output
        global_img_tensors = sample_batch[0].cuda().contiguous().float()
        video_label = sample_batch[1].cuda()
        batch_size_current, T = video_label.size()
        task_class = sample_batch[2].view(-1).cuda()
        cond = {}

        with torch.no_grad():
            cond[0] = global_img_tensors[:, 0, :]
            cond[T - 1] = global_img_tensors[:, -1, :]

            task_onehot = torch.zeros((task_class.size(0), args.class_dim))  # [bs*T, ac_dim]
            ind = torch.arange(0, len(task_class))
            task_onehot[ind, task_class] = 1.
            task_onehot = task_onehot.cuda()
            temp = task_onehot.unsqueeze(1)
            task_class_ = temp.repeat(1, T, 1)  # [bs, T, args.class_dim]
            cond['task'] = task_class_

            video_label_reshaped = video_label.view(-1)

            action_label_onehot = torch.zeros((video_label_reshaped.size(0), args.action_dim))
            ind = torch.arange(0, len(video_label_reshaped))
            action_label_onehot[ind, video_label_reshaped] = 1.
            action_label_onehot = action_label_onehot.reshape(batch_size_current, T, -1).cuda()

            x_start = torch.zeros((batch_size_current, T, args.class_dim + args.action_dim + args.observation_dim))
            x_start[:, 0, args.class_dim + args.action_dim:] = global_img_tensors[:, 0, :]
            x_start[:, -1, args.class_dim + args.action_dim:] = global_img_tensors[:, -1, :]

            x_start[:, :, args.class_dim:args.class_dim + args.action_dim] = action_label_onehot
            x_start[:, :, :args.class_dim] = task_class_
            output = model(cond)
            actions_pred = output.contiguous()
            loss = model.module.loss_fn(actions_pred, x_start.cuda())

            actions_pred = actions_pred[:, :, args.class_dim:args.class_dim + args.action_dim].contiguous()
            actions_pred = actions_pred.view(-1, args.action_dim)  # [bs*T, action_dim]

            (acc1, acc5), trajectory_success_rate, MIoU1, MIoU2, a0_acc, aT_acc = \
                accuracy(actions_pred.cpu(), video_label_reshaped.cpu(), topk=(1, 5), max_traj_len=args.horizon)

        losses.update(loss.item(), batch_size_current)
        acc_top1.update(acc1.item(), batch_size_current)
        acc_top5.update(acc5.item(), batch_size_current)
        trajectory_success_rate_meter.update(trajectory_success_rate.item(), batch_size_current)
        MIoU1_meter.update(MIoU1, batch_size_current)
        MIoU2_meter.update(MIoU2, batch_size_current)
        A0_acc.update(a0_acc, batch_size_current)
        AT_acc.update(aT_acc, batch_size_current)

    return torch.tensor(losses.avg), torch.tensor(acc_top1.avg), torch.tensor(acc_top5.avg), \
           torch.tensor(trajectory_success_rate_meter.avg), \
           torch.tensor(MIoU1_meter.avg), torch.tensor(MIoU2_meter.avg), \
           torch.tensor(A0_acc.avg), torch.tensor(AT_acc.avg)
