import torch
import time
import logging

import numpy as np
from tqdm import tqdm
import torch.distributed as dist
from tensorboardX import SummaryWriter
from tools.loss import sequence_loss
from tools.loss import compute_loss
from tools.metric import compute_epe_train, compute_epe

# import torch.optim.lr_scheduler as lr_scheduler

from util.common_util import AverageMeter
from tools.parser import get_logger

def main_process(args):
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)

def train_refine(args, epoch,model,train_dataloader,optimizer):
    logger = get_logger(__name__)
    writer = SummaryWriter(log_dir = args.log_dir,flush_secs=10)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    loss_train = AverageMeter()
    epe_train = AverageMeter()
    
    model.train()
    end = time.time()
    max_iter = args.num_epochs * len(train_dataloader)
    
    train_progress = tqdm(train_dataloader, ncols=150)
    for i, batch_data in enumerate(train_progress):
        data_time.update(time.time() - end)
        global_step = epoch * len(train_dataloader) + i
        batch_data = batch_data.myto()
        
        optimizer.zero_grad()
        
        est_flow = model(batch_data["sequence"], num_iters = args.iters)
        
        loss = compute_loss(est_flow, batch_data)
        epe = compute_epe_train(est_flow, batch_data)
    
        if args.multiprocessing_distributed:
            dist.barrier()
            dist.all_reduce(loss),dist.all_reduce(epe)
            loss /= args.numgpus
            epe  /= args.numgpus
        
        loss.backward()
                
        optimizer.step()
        
        epe_out = epe.detach().cpu().numpy()
        
        epe_train.update(epe_out)
        loss_train.update(loss.item())
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        # calculate remain time
        current_iter = epoch * len(train_dataloader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        
        if (i + 1) % args.print_freq == 0 and main_process(args):
            logger.info('Train Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'epe_train {epe_train.val:.4f}.'.format(epoch, args.num_epochs, i + 1, len(train_dataloader),
                                                          batch_time=batch_time, data_time=data_time,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_train,
                                                          epe_train=epe_train))
        
        if main_process(args):
            writer.add_scalar('loss_train_batch', loss_train.val, current_iter)
            writer.add_scalar(
                tag='Train/Loss',
                scalar_value=np.array(loss_train.avg).mean(),
                global_step=global_step
            )
            writer.add_scalar(
                tag='Train/EPE',
                scalar_value=np.array(epe_train.avg).mean(),
                global_step=global_step
            )
    if main_process(args):
        logger.info('Train result at epoch [{}/{}]: LOSS/EPE {:.4f}/{:.4f}.'.format(epoch, args.num_epochs, loss_train.avg, epe_train.avg))
        writer.close()
        
    return loss_train.avg, epe_train.avg