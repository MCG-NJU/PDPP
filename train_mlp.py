import glob
import os
import random
import time
from collections import OrderedDict

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from torch.distributed import ReduceOp
import torch.nn.functional as F
from dataloader.data_load_mlp import PlanningDataset
from model.helpers import get_lr_schedule_with_warmup, Logger
import torch.nn as nn
from utils import *
from logging import log
from utils.args import get_args
import numpy as np


def cycle(dl):
    while True:
        for data in dl:
            yield data


class head(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(head, self).__init__()
        middle_dim1 = input_dim // 3
        middle_dim2 = input_dim * 4
        self.fc1 = nn.Linear(input_dim, middle_dim1)
        self.fc2 = nn.Linear(middle_dim1, middle_dim2)
        self.fc3 = nn.Linear(middle_dim2, middle_dim1)
        self.fc4 = nn.Linear(middle_dim1, output_dim)

        # # nn.init.xavier_normal_(self.fc.weight)
        torch.nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in')
        torch.nn.init.constant_(self.fc1.bias, 0.0)
        torch.nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in')
        torch.nn.init.constant_(self.fc2.bias, 0.0)
        torch.nn.init.kaiming_normal_(self.fc3.weight, mode='fan_in')
        torch.nn.init.constant_(self.fc3.bias, 0.0)
        torch.nn.init.kaiming_normal_(self.fc4.weight, mode='fan_in')
        torch.nn.init.constant_(self.fc4.bias, 0.0)
        self.dropout = nn.Dropout(0.)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = torch.nn.functional.relu(x)
        x = self.fc3(x)
        x = torch.nn.functional.relu(x)
        x = torch.mean(x, dim=1)
        x = self.fc4(x)
        return x


class Affine(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, 1, channel))
        self.beta = nn.Parameter(torch.zeros(1, 1, channel))

    def forward(self, x):
        return x * self.alpha + self.beta


class PreAffinePostLayerScale(nn.Module):  # https://arxiv.org/abs/2103.17239
    def __init__(self, dim, depth, fn):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif 18 < depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.affine = Affine(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.affine(x)) * self.scale + x


def l2_regularization(model, l2_alpha):
    l2_loss = []
    for module in model.module.modules():
        if type(module) is nn.Conv1d or type(module) is nn.Linear:
            l2_loss.append((module.weight ** 2).sum() / 2.0)
    return l2_alpha * sum(l2_loss)


class ResMLP(nn.Module):
    def __init__(self, input=9600, dim=3200, expansion_factor=4, depth=2, class_num=18):
        super().__init__()
        wrapper = lambda i, fn: PreAffinePostLayerScale(dim, i + 1, fn)  # 封装
        self.embedding = nn.Linear(input, dim)
        self.mlp = nn.Sequential()
        for i in range(depth):
            self.mlp.add_module('fc1_%d' % i, wrapper(i, nn.Conv1d(dim, dim, 1)))
            # nn.Conv1d(patch_size ** 2 = 256, patch_size ** 2 = 256, 1)
            self.mlp.add_module('fc1_%d' % i, wrapper(i, nn.Sequential(
                nn.Linear(dim, dim * expansion_factor),
                nn.GELU(),
                nn.Linear(dim * expansion_factor, dim),
            )))

        self.aff = Affine(dim)
        self.classifier = nn.Linear(dim, class_num)

    def forward(self, x):
        y = self.embedding(x)  # [bs, 3200]
        y = self.mlp(y)
        y = self.aff(y)
        y = torch.mean(y, dim=1)  # bs,dim
        out = self.classifier(y)
        return out


def reduce_tensor(tensor):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def main():
    args = get_args()
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    if os.path.exists(args.json_path_val):
        pass
    else:
        train_dataset = PlanningDataset(
            args.root,
            args=args,
            is_val=False,
            model=None,
        )

        test_dataset = PlanningDataset(
            args.root,
            args=args,
            is_val=True,
            model=None,
        )
    args.log_root += '_mlp'
    if args.verbose:
        print(args)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        torch.backends.cudnn.deterministic = True

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    # print('ngpus_per_node:', ngpus_per_node)

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.distributed:
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
            args.num_thread_reader = int(args.num_thread_reader / ngpus_per_node)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    # Data loading code
    train_dataset = PlanningDataset(
        args.root,
        args=args,
        is_val=False,
        model=None,
    )
    # Test data loading code
    test_dataset = PlanningDataset(
        args.root,
        args=args,
        is_val=True,
        model=None,
    )
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        train_sampler = None
        test_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.num_thread_reader,
        pin_memory=args.pin_memory,
        sampler=train_sampler,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size_val,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_thread_reader,
        sampler=test_sampler,
    )

    # create model
    # model = ResMLP(input=args.observation_dim, dim=args.observation_dim, class_num=args.class_dim)
    model = head(args.observation_dim, args.class_dim)

    if args.pretrain_cnn_path:
        net_data = torch.load(args.pretrain_cnn_path)
        model.model.load_state_dict(net_data)
        model.ema_model.load_state_dict(net_data)
    if args.distributed:
        if args.gpu is not None:
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    scheduler = get_lr_schedule_with_warmup(optimizer, int(args.n_train_steps * args.epochs))

    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoint_mlp', args.checkpoint_dir)
    if args.checkpoint_dir != '' and not (os.path.isdir(checkpoint_dir)) and args.rank == 0:
        os.mkdir(checkpoint_dir)

    if args.resume:
        checkpoint_path = get_last_checkpoint(checkpoint_dir)
        if checkpoint_path:
            log("=> loading checkpoint '{}'".format(checkpoint_path), args)
            checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(args.rank))
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["model"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            if args.rank == 0:
                # creat logger
                tb_logdir = checkpoint["tb_logdir"]
                tb_logger = Logger(tb_logdir)
            log("=> loaded checkpoint '{}' (epoch {}){}".format(checkpoint_path, checkpoint["epoch"], args.gpu), args)
        else:
            if args.rank == 0:
                # creat logger
                time_pre = time.strftime("%Y%m%d%H%M%S", time.localtime())
                logname = args.log_root + '_' + time_pre + '_' + args.dataset
                tb_logdir = os.path.join(args.log_root, logname)
                if not (os.path.exists(tb_logdir)):
                    os.makedirs(tb_logdir)
                tb_logger = Logger(tb_logdir)
                tb_logger.log_info(args)
            log("=> no checkpoint found at '{}'".format(args.resume), args)

    if args.cudnn_benchmark:
        cudnn.benchmark = True
    total_batch_size = args.world_size * args.batch_size
    log(
        "Starting training loop for rank: {}, total batch size: {}".format(
            args.rank, total_batch_size
        ), args
    )

    max_eva = 0
    old_max_epoch = 0
    save_max = os.path.join(os.path.dirname(__file__), 'save_max_mlp')

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if (epoch + 1) % 2 == 0 and args.evaluate:
            losses, acc = test(test_loader, model)

            losses_reduced = reduce_tensor(losses.cuda()).item()
            acc_reduced = reduce_tensor(acc.cuda()).item()

            if args.rank == 0:
                logs = OrderedDict()
                logs['Val/EpochLoss'] = losses_reduced
                logs['Val/EpochAcc@1'] = acc_reduced
                for key, value in logs.items():
                    tb_logger.log_scalar(value, key, epoch + 1)

                tb_logger.flush()

                if acc_reduced >= max_eva:
                    save_checkpoint2(
                        {
                            "epoch": epoch + 1,
                            "model": model.state_dict(),
                            "tb_logdir": tb_logdir,
                            "scheduler": scheduler.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        }, save_max, old_max_epoch, epoch + 1
                    )
                    max_eva = acc_reduced
                    old_max_epoch = epoch + 1

        # train for one epoch
        if (epoch + 1) % 2 == 0:  # calculate on training set
            losses, acc_top1 = train(train_loader, args.n_train_steps, model, scheduler, args, optimizer, True)
            losses_reduced = reduce_tensor(losses.cuda()).item()
            acc_top1_reduced = reduce_tensor(acc_top1.cuda()).item()

            if args.rank == 0:
                logs = OrderedDict()
                logs['Train/EpochLoss'] = losses_reduced
                logs['Train/EpochAcc@1'] = acc_top1_reduced
                for key, value in logs.items():
                    tb_logger.log_scalar(value, key, epoch + 1)

                tb_logger.flush()
        else:
            losses = train(train_loader, args.n_train_steps, model, scheduler, args, optimizer, False).cuda()
            losses_reduced = reduce_tensor(losses).item()
            if args.rank == 0:
                print('lrs:')
                for p in optimizer.param_groups:
                    print(p['lr'])
                print('---------------------------------')

                logs = OrderedDict()
                logs['Train/EpochLoss'] = losses_reduced
                for key, value in logs.items():
                    tb_logger.log_scalar(value, key, epoch + 1)

                tb_logger.flush()

        if (epoch + 1) % args.save_freq == 0:
            if args.rank == 0:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "tb_logdir": tb_logdir,
                        "scheduler": scheduler.state_dict(),
                    }, checkpoint_dir, epoch + 1
                )


def test(val_loader, model):
    model.eval()
    losses = AverageMeter()
    acc_top1 = AverageMeter()
    for i_batch, sample_batch in enumerate(val_loader):
        global_img_tensors = sample_batch[0].cuda()
        batch_size_current, T, dim = global_img_tensors.size()
        task_class = sample_batch[2].cuda()

        with torch.no_grad():
            task_class = task_class.view(-1)
            observations = torch.zeros(batch_size_current, 2, dim)
            observations[:, 0, :] = global_img_tensors[:, 0, :]
            observations[:, 1, :] = global_img_tensors[:, -1, :]

            task_s = model(observations.cuda())  # [bs, 18]
            task_class_one_hot = task_class

            # loss = F.mse_loss(task_s, task_class_one_hot.cuda())
            loss = F.cross_entropy(task_s, task_class_one_hot.cuda())

            task_pred = task_s.argmax(dim=-1)
            correct = task_pred.eq(task_class)
            acc = torch.sum(correct) / batch_size_current * 100

        losses.update(loss.item(), batch_size_current)
        acc_top1.update(acc.item(), batch_size_current)

    return torch.tensor(losses.avg), torch.tensor(acc_top1.avg)


def train(train_loader, n_train_steps, model, scheduler, args, optimizer, if_calculate_acc):
    model.train()
    losses = AverageMeter()
    train_loader_ = cycle(train_loader)
    optimizer.zero_grad()
    for step in range(n_train_steps):
        for i in range(args.gradient_accumulate_every):
            batch = next(train_loader_)

            bs, T, dim = batch[0].shape  # [bs, (T+1), ob_dim]
            with torch.set_grad_enabled(True):
                task_class = batch[2].view(-1).cuda()  # [bs]

                global_img_tensors = batch[0].cuda()
                observations = torch.zeros(bs, 2, dim)
                observations[:, 0, :] = global_img_tensors[:, 0, :]
                observations[:, 1, :] = global_img_tensors[:, -1, :]

                task_s = model(observations.cuda())  # [bs, 18]
                task_class_one_hot = task_class

                # loss = F.mse_loss(task_s, task_class_one_hot.cuda())
                loss = F.cross_entropy(task_s, task_class_one_hot.cuda())

                loss = loss / args.gradient_accumulate_every
            loss.backward()
            losses.update(loss.item(), bs)

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    if if_calculate_acc:
        with torch.no_grad():
            task_pred = task_s.argmax(dim=-1)
            correct = task_pred.eq(task_class)
            acc = torch.sum(correct) / bs * 100
        return torch.tensor(losses.avg), torch.tensor(acc)

    else:
        return torch.tensor(losses.avg)


def log(output, args):
    with open(os.path.join(os.path.dirname(__file__), 'log', args.checkpoint_dir + '.txt'), "a") as f:
        f.write(output + '\n')


def save_checkpoint(state, checkpoint_dir, epoch, n_ckpt=3):
    torch.save(state, os.path.join(checkpoint_dir, "epoch{:0>4d}.pth.tar".format(epoch)))
    if epoch - n_ckpt >= 0:
        oldest_ckpt = os.path.join(checkpoint_dir, "epoch{:0>4d}.pth.tar".format(epoch - n_ckpt))
        if os.path.isfile(oldest_ckpt):
            os.remove(oldest_ckpt)


def save_checkpoint2(state, checkpoint_dir, old_epoch, epoch):
    torch.save(state, os.path.join(checkpoint_dir, "epoch{:0>4d}.pth.tar".format(epoch)))
    if old_epoch > 0:
        oldest_ckpt = os.path.join(checkpoint_dir, "epoch{:0>4d}.pth.tar".format(old_epoch))
        if os.path.isfile(oldest_ckpt):
            os.remove(oldest_ckpt)


def get_last_checkpoint(checkpoint_dir):
    all_ckpt = glob.glob(os.path.join(checkpoint_dir, 'epoch*.pth.tar'))
    if all_ckpt:
        all_ckpt = sorted(all_ckpt)
        return all_ckpt[-1]
    else:
        return ''


if __name__ == "__main__":
    main()
