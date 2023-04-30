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

import utils
from dataloader.data_load import PlanningDataset
from model import diffusion, temporal
from model.helpers import get_lr_schedule_with_warmup

from utils import *
from logging import log
from utils.args import get_args
import numpy as np
from model.helpers import Logger


def reduce_tensor(tensor):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def main():
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    args = get_args()
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    if args.verbose:
        print(args)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()

    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    # print('gpuid:', args.gpu)

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
    temporal_model = temporal.TemporalUnet(   # temporal
        args.action_dim + args.observation_dim + args.class_dim,
        dim=256,
        dim_mults=(1, 2, 4), )

    # for param in temporal_model.named_parameters():
    #     if 'time_mlp' not in param[0]:
    #         param[1].requires_grad = False

    diffusion_model = diffusion.GaussianDiffusion(
        temporal_model, args.horizon, args.observation_dim, args.action_dim, args.class_dim, args.n_diffusion_steps,
        loss_type='Weighted_MSE', clip_denoised=True, )

    model = utils.Trainer(diffusion_model, train_loader, args.ema_decay, args.lr, args.gradient_accumulate_every,
                          args.step_start_ema, args.update_ema_every, args.log_freq)

    if args.pretrain_cnn_path:
        net_data = torch.load(args.pretrain_cnn_path)
        model.model.load_state_dict(net_data)
        model.ema_model.load_state_dict(net_data)
    if args.distributed:
        if args.gpu is not None:
            model.model.cuda(args.gpu)
            model.ema_model.cuda(args.gpu)
            model.model = torch.nn.parallel.DistributedDataParallel(
                model.model, device_ids=[args.gpu], find_unused_parameters=True)
            model.ema_model = torch.nn.parallel.DistributedDataParallel(
                model.ema_model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.model.cuda()
            model.ema_model.cuda()
            model.model = torch.nn.parallel.DistributedDataParallel(model.model, find_unused_parameters=True)
            model.ema_model = torch.nn.parallel.DistributedDataParallel(model.ema_model,
                                                                        find_unused_parameters=True)

    elif args.gpu is not None:
        model.model = model.model.cuda(args.gpu)
        model.ema_model = model.ema_model.cuda(args.gpu)
    else:
        model.model = torch.nn.DataParallel(model.model).cuda()
        model.ema_model = torch.nn.DataParallel(model.ema_model).cuda()

    scheduler = get_lr_schedule_with_warmup(model.optimizer, int(args.n_train_steps * args.epochs))

    checkpoint_dir = os.path.join(os.path.dirname(__file__), 'checkpoint', args.checkpoint_dir)
    if args.checkpoint_dir != '' and not (os.path.isdir(checkpoint_dir)) and args.rank == 0:
        os.mkdir(checkpoint_dir)

    if args.resume:
        checkpoint_path = get_last_checkpoint(checkpoint_dir)
        if checkpoint_path:
            log("=> loading checkpoint '{}'".format(checkpoint_path), args)
            checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(args.rank))
            args.start_epoch = checkpoint["epoch"]
            model.model.load_state_dict(checkpoint["model"])
            model.ema_model.load_state_dict(checkpoint["ema"])
            model.optimizer.load_state_dict(checkpoint["optimizer"])
            model.step = checkpoint["step"]
            # for p in model.optimizer.param_groups:
            #     p['lr'] = 1e-5
            scheduler.load_state_dict(checkpoint["scheduler"])
            tb_logdir = checkpoint["tb_logdir"]
            if args.rank == 0:
                # creat logger
                tb_logger = Logger(tb_logdir)
                log("=> loaded checkpoint '{}' (epoch {}){}".format(checkpoint_path, checkpoint["epoch"], args.gpu), args)
        else:
            # log("=> loading checkpoint '{}' to initialize".format(checkpoint_path), args)
            # checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(args.rank))
            # model.model.load_state_dict(checkpoint["model"], strict=False)
            # model.ema_model.load_state_dict(checkpoint["ema"], strict=False)
            time_pre = time.strftime("%Y%m%d%H%M%S", time.localtime())
            logname = args.log_root + '_' + time_pre + '_' + args.dataset
            tb_logdir = os.path.join(args.log_root, logname)
            if args.rank == 0:
                # creat logger
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
    max_acc = 0
    old_max_epoch = 0
    save_max = os.path.join(os.path.dirname(__file__), 'save_max')

    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        if (epoch + 1) % 10 == 0:  # calculate on training set
            losses, acc_top1, acc_top5, trajectory_success_rate_meter, MIoU1_meter, MIoU2_meter, \
            acc_a0, acc_aT = model.train(args.n_train_steps, True, args, scheduler)
            losses_reduced = reduce_tensor(losses.cuda()).item()
            acc_top1_reduced = reduce_tensor(acc_top1.cuda()).item()
            acc_top5_reduced = reduce_tensor(acc_top5.cuda()).item()
            trajectory_success_rate_meter_reduced = reduce_tensor(trajectory_success_rate_meter.cuda()).item()
            MIoU1_meter_reduced = reduce_tensor(MIoU1_meter.cuda()).item()
            MIoU2_meter_reduced = reduce_tensor(MIoU2_meter.cuda()).item()
            acc_a0_reduced = reduce_tensor(acc_a0.cuda()).item()
            acc_aT_reduced = reduce_tensor(acc_aT.cuda()).item()

            if args.rank == 0:
                logs = OrderedDict()
                logs['Train/EpochLoss'] = losses_reduced
                logs['Train/EpochAcc@1'] = acc_top1_reduced
                logs['Train/EpochAcc@5'] = acc_top5_reduced
                logs['Train/Traj_Success_Rate'] = trajectory_success_rate_meter_reduced
                logs['Train/MIoU1'] = MIoU1_meter_reduced
                logs['Train/MIoU2'] = MIoU2_meter_reduced
                logs['Train/acc_a0'] = acc_a0_reduced
                logs['Train/acc_aT'] = acc_aT_reduced
                for key, value in logs.items():
                    tb_logger.log_scalar(value, key, epoch + 1)

                tb_logger.flush()
        else:
            losses = model.train(args.n_train_steps, False, args, scheduler).cuda()
            losses_reduced = reduce_tensor(losses).item()
            if args.rank == 0:
                print('lrs:')
                for p in model.optimizer.param_groups:
                    print(p['lr'])
                print('---------------------------------')

                logs = OrderedDict()
                logs['Train/EpochLoss'] = losses_reduced
                for key, value in logs.items():
                    tb_logger.log_scalar(value, key, epoch + 1)

                tb_logger.flush()

        if ((epoch + 1) % 5 == 0) and args.evaluate:  # or epoch > 18
            losses, acc_top1, acc_top5, \
            trajectory_success_rate_meter, MIoU1_meter, MIoU2_meter, \
            acc_a0, acc_aT = validate(test_loader, model.ema_model, args)

            losses_reduced = reduce_tensor(losses.cuda()).item()
            acc_top1_reduced = reduce_tensor(acc_top1.cuda()).item()
            acc_top5_reduced = reduce_tensor(acc_top5.cuda()).item()
            trajectory_success_rate_meter_reduced = reduce_tensor(trajectory_success_rate_meter.cuda()).item()
            MIoU1_meter_reduced = reduce_tensor(MIoU1_meter.cuda()).item()
            MIoU2_meter_reduced = reduce_tensor(MIoU2_meter.cuda()).item()
            acc_a0_reduced = reduce_tensor(acc_a0.cuda()).item()
            acc_aT_reduced = reduce_tensor(acc_aT.cuda()).item()

            if args.rank == 0:
                logs = OrderedDict()
                logs['Val/EpochLoss'] = losses_reduced
                logs['Val/EpochAcc@1'] = acc_top1_reduced
                logs['Val/EpochAcc@5'] = acc_top5_reduced
                logs['Val/Traj_Success_Rate'] = trajectory_success_rate_meter_reduced
                logs['Val/MIoU1'] = MIoU1_meter_reduced
                logs['Val/MIoU2'] = MIoU2_meter_reduced
                logs['Val/acc_a0'] = acc_a0_reduced
                logs['Val/acc_aT'] = acc_aT_reduced
                for key, value in logs.items():
                    tb_logger.log_scalar(value, key, epoch + 1)

                tb_logger.flush()
                print(trajectory_success_rate_meter_reduced, max_eva)
            if trajectory_success_rate_meter_reduced >= max_eva:
                if not (trajectory_success_rate_meter_reduced == max_eva and acc_top1_reduced < max_acc):
                    save_checkpoint2(
                        {
                            "epoch": epoch + 1,
                            "model": model.model.state_dict(),
                            "ema": model.ema_model.state_dict(),
                            "optimizer": model.optimizer.state_dict(),
                            "step": model.step,
                            "tb_logdir": tb_logdir,
                            "scheduler": scheduler.state_dict(),
                        }, save_max, old_max_epoch, epoch + 1, args.rank
                    )
                    max_eva = trajectory_success_rate_meter_reduced
                    max_acc = acc_top1_reduced
                    old_max_epoch = epoch + 1

        if (epoch + 1) % args.save_freq == 0:
            if args.rank == 0:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "model": model.model.state_dict(),
                        "ema": model.ema_model.state_dict(),
                        "optimizer": model.optimizer.state_dict(),
                        "step": model.step,
                        "tb_logdir": tb_logdir,
                        "scheduler": scheduler.state_dict(),
                    }, checkpoint_dir, epoch + 1
                )


def log(output, args):
    with open(os.path.join(os.path.dirname(__file__), 'log', args.checkpoint_dir + '.txt'), "a") as f:
        f.write(output + '\n')


def save_checkpoint(state, checkpoint_dir, epoch, n_ckpt=3):
    torch.save(state, os.path.join(checkpoint_dir, "epoch{:0>4d}.pth.tar".format(epoch)))
    if epoch - n_ckpt >= 0:
        oldest_ckpt = os.path.join(checkpoint_dir, "epoch{:0>4d}.pth.tar".format(epoch - n_ckpt))
        if os.path.isfile(oldest_ckpt):
            os.remove(oldest_ckpt)


def save_checkpoint2(state, checkpoint_dir, old_epoch, epoch, rank):
    torch.save(state, os.path.join(checkpoint_dir, "epoch{:0>4d}_{}.pth.tar".format(epoch, rank)))
    if old_epoch > 0:
        oldest_ckpt = os.path.join(checkpoint_dir, "epoch{:0>4d}_{}.pth.tar".format(old_epoch, rank))
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
