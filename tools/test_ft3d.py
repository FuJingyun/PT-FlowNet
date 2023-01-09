import torch
import time

import numpy as np
from tqdm import tqdm
import torch.distributed as dist
#from tensorboardX import SummaryWriter
from tools.loss import sequence_loss
# from tools.loss import compute_loss
from tools.metric import  compute_epe, compute_epe2d_ft3d

from util.common_util import AverageMeter
from tools.parser import get_logger
# from util.geometry import get_batch_2d_flow


def main_process(args):
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)

def test_part_ft3d(args, model,test_dataloader, epoch=0):
    logger = get_logger(__name__)
    if main_process(args):
        logger.info('>>>>>>>>>>>>>>>> Start Testing >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    epe_meter = AverageMeter()
    outlier_meter = AverageMeter()
    acc3dRelax_meter = AverageMeter()
    acc3dStrict_meter = AverageMeter()

    epe2d_meter = AverageMeter()
    acc2d_meter = AverageMeter()
    
    model.eval()
    end = time.time()

    run_progress = tqdm(test_dataloader, ncols=150)
    for i, batch_data in enumerate(run_progress):
        data_time.update(time.time() - end)
        global_step = epoch * len(test_dataloader) + i
        batch_data = batch_data.myto()
            
        with torch.no_grad():
            est_flow = model(batch_data["sequence"], 32)
            
        loss = sequence_loss(est_flow, batch_data, gamma = args.gamma)
        epe, acc3d_strict, acc3d_relax, outlier = compute_epe(est_flow[-1], batch_data)

        # 2D evaluation metrics
        epe2d, acc2d = compute_epe2d_ft3d(est_flow[-1], batch_data)
        
        if args.multiprocessing_distributed:
            dist.barrier()
            dist.all_reduce(loss),dist.all_reduce(epe),dist.all_reduce(acc3d_strict),dist.all_reduce(acc3d_relax),dist.all_reduce(outlier),dist.all_reduce(epe2d),dist.all_reduce(acc2d)
            loss /= args.numgpus
            epe  /= args.numgpus
            acc3d_strict /= args.numgpus
            acc3d_relax /= args.numgpus
            outlier /= args.numgpus
            epe2d /= args.numgpus
            acc2d /= args.numgpus
        
        epe, acc3d_strict, acc3d_relax, outlier,epe2d,acc2d = epe.cpu().numpy(), acc3d_strict.cpu().numpy(), acc3d_relax.cpu().numpy(), outlier.cpu().numpy(), epe2d.cpu().numpy(), acc2d.cpu().numpy()
        
        loss_meter.update(loss.item())
        epe_meter.update(epe)
        outlier_meter.update(outlier)
        acc3dRelax_meter.update(acc3d_relax)
        acc3dStrict_meter.update(acc3d_strict)

        epe2d_meter.update(epe2d)
        acc2d_meter.update(acc2d)
        
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0 and main_process(args):
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'EPE {epe_meter.val:.4f}.'
                        'Outlier{outlier_meter.val:.4f}'
                        'acc3dRelax{acc3dRelax_meter.val:.4f}'
                        'acc3dStrict{acc3dStrict_meter.val:.4f}'
                        'EPE2D {epe2d_meter.val:.4f}'
                        'acc2d{acc2d_meter.val:.4f}'.format(i + 1, len(test_dataloader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          epe_meter=epe_meter,
                                                          outlier_meter = outlier_meter,
                                                          acc3dRelax_meter = acc3dRelax_meter,
                                                          acc3dStrict_meter = acc3dStrict_meter,
                                                          epe2d_meter = epe2d_meter,
                                                          acc2d_meter = acc2d_meter
                                                          ))
    if main_process(args):
        logger.info('Test result: LOSS/EPE/Outlier/acc3dRelax/acc3dStrict/EPE2D/acc2d {:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}/{:.4f}.'.format(loss_meter.avg, epe_meter.avg,outlier_meter.avg,acc3dRelax_meter.avg,acc3dStrict_meter.avg,epe2d_meter.avg,acc2d_meter.avg))
        logger.info('<<<<<<<<<<<<<<<<< End Test <<<<<<<<<<<<<<<<<')
