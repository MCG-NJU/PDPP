import os
import random
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import numpy as np
import utils

from torch.distributed import ReduceOp
from dataloader.data_load import PlanningDataset
from model import diffusion, temporal
from utils import *
from utils.args import get_args


def custom_NLL(input, target):
    return torch.mean(-torch.sum(target * input, 1))


def test(val_loader, model, args, all_ref):
    model.eval()
    num_sampling = 1500     # 1500 for Noise and diffusion, 1 for Deterministic
    klv_list = []
    klv_list2 = []
    mc_prec = []
    mc_recall = []
    act_size = args.action_dim
    len_unique = []

    for i_batch, sample_batch in enumerate(val_loader):
        for i in range(len(sample_batch[0])):
            # compute output
            global_img_tensors = sample_batch[0][i].cuda().contiguous().unsqueeze(0)  # [1, T+1, ob_dim]
            video_label = sample_batch[1][i].cuda().unsqueeze(0)  # [1, T]
            _, T = video_label.size()
            task_class = sample_batch[2][i].view(-1).cuda().unsqueeze(0)  # [1, 1]
            # video_vid = sample_batch[3][i]

            cond = {}
            gt = video_label
            sample_listing = []
            ratio_list = []

            with torch.no_grad():
                cond[0] = global_img_tensors[:, 0, :].float().repeat(num_sampling, 1)
                cond[T - 1] = global_img_tensors[:, -1, :].float().repeat(num_sampling, 1)
                task_class = task_class.repeat(num_sampling, 1)

                task_onehot = torch.zeros((task_class.size(0), args.class_dim))
                ind = torch.arange(0, len(task_class))
                task_onehot[ind, task_class] = 1.
                task_onehot = task_onehot.cuda()
                temp = task_onehot.unsqueeze(1)
                task_class_ = temp.repeat(1, T, 1)  # [bs, T, args.class_dim]
                cond['task'] = task_class_

                output = model(cond, if_jump=True)
                actions_pred = output.contiguous()
                actions_pred_logits = actions_pred[:, :, args.class_dim:args.class_dim + args.action_dim].contiguous()
                actions_pred = torch.argmax(actions_pred_logits, dim=-1)
                actions_pred = actions_pred.view(num_sampling, -1)
                sample_listing = actions_pred

                bz = all_ref.shape[0]
                gt_sample = np.repeat(gt.cpu().numpy(), bz, axis=0)
                criter = (
                    (gt_sample[:, [0, -1]] == all_ref[:, [0, -1]])
                        .all(axis=1)
                        .nonzero()[0]
                )

                dist_samples = all_ref[criter]
                len_unique.append(len(np.unique(dist_samples, axis=0)))
                ref_onehot = torch.FloatTensor(args.horizon, act_size).cuda()
                ref_onehot.zero_()

                ######################################################################
                # dist_samples represents the samples in the test-set:               #
                #    1). Share the same start and end-goal semantic;                 #
                #                                                                    #
                # If can not find any dist_samples (aka dist_samples.shape[0] == 0): #
                #    1). Skip the nll evaluation (see below code)                    #
                ######################################################################
                if dist_samples.shape[0] != 0:
                    for vec in dist_samples:
                        vec = torch.from_numpy(vec).cuda()
                        ref_onehot_tmp = torch.FloatTensor(
                            args.horizon, act_size
                        ).cuda()
                        ref_onehot_tmp.zero_()
                        ref_onehot_tmp.scatter_(
                            1, vec.view(args.horizon, -1), 1)
                        ref_onehot += ref_onehot_tmp

                    ref_dist = ref_onehot
                    itm_onehot = torch.FloatTensor(args.horizon, act_size).cuda()
                    itm_onehot.zero_()

                    for itm in sample_listing:
                        ###########################################
                        # Convert indivisual sample into onehot() #
                        ###########################################
                        itm_onehot_tmp = torch.FloatTensor(args.horizon, act_size).cuda()
                        itm_onehot_tmp.zero_()
                        itm_onehot_tmp.scatter_(
                            1, itm.cuda().view(args.horizon, -1), 1)
                        itm_onehot += itm_onehot_tmp

                ###########################################
                # Evaluate on Mode-Coverage Prec & Recall #
                ###########################################
                ratio_list = []
                for sample in sample_listing:
                    ratio_list.append(
                        (sample.squeeze().cpu().numpy()
                         == dist_samples).all(1).any()
                    )
                ratio = sum(ratio_list) / num_sampling
                mc_prec.append(ratio)

                # all_samples = torch.stack(
                #     sample_listing).squeeze().cpu().numpy()
                all_samples = sample_listing.cpu().numpy()

                # dist_samples_unique = np.unique(dist_samples, axis=0)
                dist_samples_unique = dist_samples
                num_expert = dist_samples_unique.shape[0]
                list_expert = np.array_split(dist_samples_unique, num_expert)
                tmp_recall = []
                for item in list_expert:
                    tmp_recall.append((item == all_samples).all(1).any())
                mc_recall.append(sum(tmp_recall) / len(tmp_recall))

                ####################################
                #   Calculate the KL-Div  Metric   #
                ####################################

                # if ratio != 0 and sum(tmp_recall) != 0 and len(np.unique(actions_pred.cpu().numpy(), axis=0)) > 1:
                # # if True:
                #     print('vid :', video_vid)
                #     print('gt :', video_label)
                #     print('gts :', np.unique(dist_samples, axis=0))
                #     print('samples :', np.unique(actions_pred.cpu().numpy(), axis=0))
                #     print('--------------------------------------------------')

                ref_dist /= dist_samples.shape[0]
                itm_onehot /= num_sampling
                ref_dist *= 10
                itm_onehot *= 10

                ref_dist = ref_dist.softmax(dim=-1)
                itm_onehot = itm_onehot.softmax(dim=-1)

                klv_rst = (
                    torch.nn.functional.kl_div(
                        itm_onehot.log(),
                        ref_dist,
                        reduction='batchmean'
                    )
                    .cpu()
                    .numpy()
                )
                klv_rst = np.where(np.isnan(klv_rst), 0, klv_rst)
                klv_rst = np.where(np.isinf(klv_rst), 0, klv_rst)
                klv_list.append(klv_rst)

                ####################################
                #   Calculate the NLL  Metric   #
                ####################################

                klv_rst2 = (
                    custom_NLL(
                        itm_onehot.log(),
                        ref_dist,
                    )
                    .cpu()
                    .numpy()
                )
                klv_rst2 = np.where(np.isnan(klv_rst2), 0, klv_rst2)
                klv_rst2 = np.where(np.isinf(klv_rst2), 0, klv_rst2)
                klv_list2.append(klv_rst2)

    avg_mc = sum(mc_prec) / len(mc_prec)
    avg_mc_recall = sum(mc_recall) / len(mc_recall)
    avg_kl = sum(klv_list) / len(klv_list)
    avg_nll = sum(klv_list2) / len(klv_list2)
    len_unique_avg = sum(len_unique) / len(len_unique)
    return avg_nll, avg_mc, avg_mc_recall, len_unique_avg, avg_kl


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
    # print('ngpus_per_node:', ngpus_per_node)

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

    # Test data loading code
    test_dataset = PlanningDataset(
        args.root,
        args=args,
        is_val=True,
        model=None,
    )
    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        test_sampler.shuffle = False
    else:
        test_sampler = None

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size_val,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_thread_reader,
        sampler=test_sampler,
    )

    # create model
    temporal_model = temporal.TemporalUnet(
        args.action_dim + args.observation_dim + args.class_dim,
        dim=256,
        dim_mults=(1, 2, 4), )

    diffusion_model = diffusion.GaussianDiffusion(
        temporal_model, args.horizon, args.observation_dim, args.action_dim, args.class_dim, args.n_diffusion_steps,
        loss_type='Weighted_MSE', clip_denoised=True, )

    model = utils.Trainer(diffusion_model, None, args.ema_decay, args.lr, args.gradient_accumulate_every,
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

    if args.resume:
        checkpoint_path = ""
        if checkpoint_path:
            checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(args.rank))
            args.start_epoch = checkpoint["epoch"]
            model.model.load_state_dict(checkpoint["model"])
            model.ema_model.load_state_dict(checkpoint["ema"])
            model.step = checkpoint["step"]
        else:
            assert 0

    if args.cudnn_benchmark:
        cudnn.benchmark = True

    test_times = 1

    reference = []
    for x in test_loader:
        for i in range(len(x[1])):
            reference.append(x[1][i].cpu().numpy())

    all_ref = np.array(reference)

    for epoch in range(0, test_times):
        avg_nll, avg_mc, avg_mc_recall, len_unique_avg, avg_kl = test(test_loader, model.ema_model, args, all_ref)
        if args.rank == 0:
            print(
                "NLL {}, MC-Prec {}, MC-Rec {}, Avg.length {}, KL {}".format(
                    avg_nll,
                    avg_mc,
                    avg_mc_recall,
                    len_unique_avg,
                    avg_kl,
                )
            )


if __name__ == "__main__":
    main()
