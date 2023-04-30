import argparse


def get_args(description='whl'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--checkpoint_root',
                        type=str,
                        default='checkpoint',
                        help='checkpoint dir root')
    parser.add_argument('--log_root',
                        type=str,
                        default='log',
                        help='log dir root')
    parser.add_argument('--checkpoint_dir',
                        type=str,
                        default='',
                        help='checkpoint model folder')
    parser.add_argument('--optimizer',
                        type=str,
                        default='adam',
                        help='opt algorithm')
    parser.add_argument('--num_thread_reader',
                        type=int,
                        default=40,
                        help='')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='batch size')
    parser.add_argument('--batch_size_val',
                        type=int,
                        default=1024,
                        help='batch size eval')
    parser.add_argument('--pretrain_cnn_path',
                        type=str,
                        default='',
                        help='')
    parser.add_argument('--momemtum',
                        type=float,
                        default=0.9,
                        help='SGD momemtum')
    parser.add_argument('--log_freq',
                        type=int,
                        default=500,
                        help='how many steps do we log once')
    parser.add_argument('--save_freq',
                        type=int,
                        default=1,
                        help='how many epochs do we save once')
    parser.add_argument('--gradient_accumulate_every',
                        type=int,
                        default=1,
                        help='accumulation_steps')
    parser.add_argument('--ema_decay',
                        type=float,
                        default=0.995,
                        help='')
    parser.add_argument('--step_start_ema',
                        type=int,
                        default=400,
                        help='')
    parser.add_argument('--update_ema_every',
                        type=int,
                        default=10,
                        help='')
    parser.add_argument('--crop_only',
                        type=int,
                        default=1,
                        help='random seed')
    parser.add_argument('--centercrop',
                        type=int,
                        default=0,
                        help='random seed')
    parser.add_argument('--random_flip',
                        type=int,
                        default=1,
                        help='random seed')
    parser.add_argument('--verbose',
                        type=int,
                        default=1,
                        help='')
    parser.add_argument('--fps',
                        type=int,
                        default=1,
                        help='')
    parser.add_argument('--cudnn_benchmark',
                        type=int,
                        default=0,
                        help='')
    parser.add_argument('--horizon',
                        type=int,
                        default=3,
                        help='')
    parser.add_argument('--dataset',
                        type=str,
                        default='crosstask',
                        help='dataset')
    parser.add_argument('--action_dim',
                        type=int,
                        default=105,
                        help='')
    parser.add_argument('--observation_dim',
                        type=int,
                        default=1536,
                        help='')
    parser.add_argument('--class_dim',
                        type=int,
                        default=18,
                        help='')
    parser.add_argument('--n_diffusion_steps',
                        type=int,
                        default=200,
                        help='')
    parser.add_argument('--n_train_steps',
                        type=int,
                        default=200,
                        help='training_steps_per_epoch')
    parser.add_argument('--root',
                        type=str,
                        default='',
                        help='root path of dataset crosstask')
    parser.add_argument('--json_path_train',
                        type=str,
                        default='',
                        help='path of the generated json file for train')
    parser.add_argument('--json_path_val',
                        type=str,
                        default='',
                        help='path of the generated json file for val')

    parser.add_argument('--epochs', default=120, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--lr', '--learning-rate', default=5e-4, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--resume', dest='resume', action='store_true',
                        help='resume training from last checkpoint')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--pin_memory', dest='pin_memory', action='store_true',
                        help='use pin_memory')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-file', default='dist-file', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-url', default='tcp://localhost:21712', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=217, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    args = parser.parse_args()
    return args