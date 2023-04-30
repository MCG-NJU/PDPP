import json
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

from data_load_json import PlanningDataset
from utils import *
from utils.args import get_args
from train_mlp import ResMLP, head


def main():
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

    test_dataset = PlanningDataset(
        args.root,
        args=args,
        is_val=True,
        model=None,
    )

    # create model
    # model = ResMLP(input=args.observation_dim, dim=args.observation_dim, class_num=args.class_dim)
    model = head(args.observation_dim, args.class_dim)

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

    checkpoint_ = torch.load("",
                             map_location='cuda:{}'.format(args.rank))
    model.load_state_dict(checkpoint_["model"])

    if args.cudnn_benchmark:
        cudnn.benchmark = True

    model.eval()

    json_ret = []
    correct = 0

    for i in range(len(test_dataset)):
        frames_t, vid_names, frame_cnts = test_dataset[i]
        event_class = model(frames_t)
        event_class_id = torch.argmax(event_class)
        if event_class_id == vid_names['task_id']:
            correct += 1
        vid_names['event_class'] = event_class_id.item()
        json_current = {}
        json_current['id'] = vid_names
        json_current['instruction_len'] = frame_cnts

        json_ret.append(json_current)

    data_name = ""

    with open(data_name, 'w') as f:
        json.dump(json_ret, f)

    print('acc: ', correct / len(test_dataset))

if __name__ == "__main__":
    main()
