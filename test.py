import os
import time
import random
import numpy as np
import logging
# import argparse
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter

from util.common_util import find_free_port
from util.data_util import collate_fn
from util import transform as t

import warnings
# import numpy as np
from tqdm import tqdm
from datetime import datetime

from datasets.generic import Batch
from datasets.flyingthings3d_hplflownet import FT3D
from datasets.kitti_hplflownet import Kitti

from model.pt_RAFTSceneFlow import pt_RSF
from model.pt_RAFTSceneFlowRefine import pt_RSF_refine

from tools.test_ft3d import test_part_ft3d
from tools.test_kitti import test_part_kitti
from tools.test_kitti_refine import test_part_kitti_refine
from tools.test_ft3d_refine import test_part_ft3d_refine


from tools.parser import parse_args,get_logger



def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)

def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)

def log_init(args, mode='Test'):
    if args.exp_path is None:
        args.exp_path = datetime.now().strftime("exp-%y_%m_%d-%H_%M_%S_%f")
    args.exp_path = os.path.join(args.root, 'experiments', args.exp_path)
    if not os.path.exists(args.exp_path):
        os.mkdir(args.exp_path)

    log_dir = os.path.join(args.exp_path, 'logs')
    args.log_dir = log_dir
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    log_name = 'TestAlone_' + args.dataset + '.log'
    logging.basicConfig(
        filename=os.path.join(log_dir, log_name),
        filemode='w',
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO
    )
    warnings.filterwarnings('ignore')
    logging.info(args)
    logging.info('')
    

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.numgpus = len(args.gpus.split(',')) 
    log_init(args,'Test')

    if args.numgpus == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
        
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.gpus.split(','))
        
        
    if args.multiprocessing_distributed:
        port = find_free_port()
        args.dist_url = f"tcp://localhost:{port}"
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.test_gpu, args.ngpus_per_node, args)
    
    
    
def main_worker(gpu, ngpus_per_node, argss):
    global args
    args = argss
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    
    if not args.refine:
        model = pt_RSF(args)
    else:
        model = pt_RSF_refine(args)
        # model.feature_extractor.requires_grad = False
        # model.context_extractor.requires_grad = False
        # model.corr_block.requires_grad = False
        # model.update_block.requires_grad = False
        
    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
           
        
    if main_process():
        global logger, writer
        logger = get_logger("main-logger")
        writer = SummaryWriter(log_dir = args.log_dir,flush_secs=10)
        logger.info(args)
        logger.info('args.log_dir:{}'.format(args.log_dir))
        logger.info("=> Preparing test ...")
        
    args.begin_epoch = 1
    

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(
            model.cuda(),
            device_ids=[gpu],
            find_unused_parameters=True if "refine" in args.arch else False
        )
    else:
        args.batch_size = 1
        args.batch_size_val = 1
        args.workers = 1
        # model = torch.nn.DataParallel(model.cuda())
        model = model.cuda()
    
    if args.weights:
        if os.path.isfile(args.weights):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weights))
            checkpoint = torch.load(args.weights,map_location=lambda storage, loc: storage.cuda())
            if not args.refine:
                args.begin_epoch = checkpoint['epoch']
                if torch.cuda.device_count() > 1:
                    model.module.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint['state_dict'])
            else:
                if torch.cuda.device_count() > 1:
                    # model.module.load_state_dict(checkpoint['state_dict'], strict=False)
                    model.module.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint['state_dict'], strict=False)
                
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weights))
        else:
            raise RuntimeError(f"=> No checkpoint found at '{args.weights}")
    
    
    
    if args.dataset == 'ALL':
        # FT3d
        dataset_path = os.path.join(args.ft3d_dataset_dir, 'FlyingThings3D_subset_processed_35m')
        test_dataset = FT3D(root_dir=dataset_path, nb_points=args.max_points, mode='test')
        # KITTI
        dataset_kitti_path = os.path.join(args.kitti_dataset_dir, 'KITTI_processed_occ_final')
        test_kitti_dataset = Kitti(root_dir=dataset_kitti_path, nb_points=args.max_points)
    else:
        raise NotImplementedError
    
    if args.distributed:
        test_sampler  = torch.utils.data.distributed.DistributedSampler(test_dataset)
        test_kitti_sampler  = torch.utils.data.distributed.DistributedSampler(test_kitti_dataset)
    else:
        # train_sampler = None
        # val_sampler = None
        test_sampler = None
        test_kitti_sampler  = None

    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size_val, shuffle=False,
                                num_workers=args.workers,pin_memory=True, sampler=test_sampler, collate_fn=Batch, drop_last=False)
    test_kitti_dataloader = DataLoader(test_kitti_dataset, batch_size=args.batch_size_val, shuffle=False, num_workers=args.workers,
                                pin_memory=True,sampler=test_kitti_sampler,collate_fn=Batch, drop_last=False)
    
    if not args.refine:
        test_part_ft3d(args, model, test_dataloader, epoch=0)
        test_part_kitti(args, model, test_kitti_dataloader, epoch=0)
    else:
        test_part_ft3d_refine(args, model, test_dataloader, epoch=0)
        test_part_kitti_refine(args, model, test_kitti_dataloader, epoch=0)
           

if __name__ == '__main__':
    import gc
    gc.collect()
    args = parse_args()
    args.test_gpu = args.gpus.split(',')
    main(args)
