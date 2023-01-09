import argparse
import logging

def parse_args():
    parser = argparse.ArgumentParser(description='Training Argument')
    parser.add_argument('--root',
                        help='workspace path',
                        default='/home',
                        type=str)
    parser.add_argument('--exp_path',
                        help='specified experiment log path',
                        default=None,
                        type=str)
    parser.add_argument('--ft3d_dataset_dir',
                        default="/data/ft3d",
                        type=str)
    parser.add_argument('--kitti_dataset_dir',
                        default="/data/kitti",
                        type=str)
    parser.add_argument('--dataset',    
                        help="dataset",
                        default='FT3D',
                        type=str)
    parser.add_argument('--max_points',
                        help='maximum number of points sampled from a point cloud',
                        default=8192,
                        type=int)
    parser.add_argument('--corr_levels',
                        help='number of correlation pyramid levels',
                        default=3,
                        type=int)
    parser.add_argument('--base_scales',
                        help='voxelize base scale',
                        default=0.25,
                        type=float)
    parser.add_argument('--truncate_k',
                        help='value of truncate_k in corr block',
                        default=512,
                        type=int)
    parser.add_argument('--iters',
                        help='number of iterations in GRU module',
                        default=8,
                        type=int)
    parser.add_argument('--gamma',
                        help='exponential weights',
                        default=0.8,
                        type=float)
    parser.add_argument('--batch_size',
                        help='number of samples in a mini-batch',
                        default=16,
                        type=int)
    parser.add_argument('--batch_size_val',
                        help='batch size for validation during training, memory and speed tradeoff',
                        default=4,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus that used for training',
                        default='0,1,2,3',
                        type=str)
    parser.add_argument('--num_epochs',
                        help='number of epochs for training',
                        default=20,
                        type=int)
    parser.add_argument('--weights',
                        help='checkpoint weights to be loaded',
                        default=None,
                        type=str)
    parser.add_argument('--checkpoint_interval',
                        help='save checkpoint every N epoch',
                        default=5,
                        type=int)
    parser.add_argument('--refine',
                        help='refine mode',
                        action='store_true')
    parser.add_argument('--manual_seed',
                        help='seed for random',
                        default=7777,
                        type=int)
    parser.add_argument('--dist_url',
                        help='dist_url',
                        default='tcp://localhost:8888',
                        type=str)
    parser.add_argument('--world_size',
                        help='world_size',
                        default=1,
                        type=int)
    parser.add_argument('--multiprocessing_distributed',
                        help='multiprocessing_distributed',
                        default=True,
                        type=bool)
    parser.add_argument('--sync_bn',
                        help='sync_bn',
                        default=False,
                        type=bool)
    parser.add_argument('--rank',
                        help='rank',
                        default=0,
                        type=int)
    parser.add_argument('--dist_backend',
                        help='dist_backend',
                        default='nccl',
                        type=str)
    parser.add_argument('--workers',
                        help='workers',
                        default=16,
                        type=int)
    parser.add_argument('--save_freq',
                        help='save_freq',
                        default=1,
                        type=int)
    parser.add_argument('--arch',
                        help='arch',
                        default='refine',
                        type=str)
    parser.add_argument('--print_freq',
                        help='print_freq',
                        default=1,
                        type=int)
    parser.add_argument('--best_epe',
                        help='best_epe for middle train',
                        default=100,
                        type=float)
    parser.add_argument('--begin_epoch',
                        help='begin_epoch',
                        default=1,
                        type=int)
    args = parser.parse_args()
    return args



def get_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger