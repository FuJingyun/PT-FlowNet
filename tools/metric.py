import numpy as np

import torch
import os.path as osp


def project_3d_to_2d(pc, f=-1050., cx=479.5, cy=269.5, constx=0, consty=0, constz=0):
    x = (pc[..., 0] * f + cx * pc[..., 2] + constx) / (pc[..., 2] + constz)
    y = (pc[..., 1] * f + cy * pc[..., 2] + consty) / (pc[..., 2] + constz)

    return x, y

def get_batch_2d_flow(pc1, pc2, predicted_pc2):

    px1, py1 = project_3d_to_2d(pc1)
    px2, py2 = project_3d_to_2d(predicted_pc2)
    px2_gt, py2_gt = project_3d_to_2d(pc2)

    flow_x = px2 - px1
    flow_y = py2 - py1

    flow_x_gt = px2_gt - px1
    flow_y_gt = py2_gt - py1

    flow_pred = np.concatenate((flow_x[..., None], flow_y[..., None]), axis=-1)
    flow_gt = np.concatenate((flow_x_gt[..., None], flow_y_gt[..., None]), axis=-1)
    return flow_pred, flow_gt


def get_batch_2d_flow_kitti(pc1, pc2, predicted_pc2,paths):
    focallengths = []
    cxs = []
    cys = []
    constx = []
    consty = []
    constz = []
    for path in paths:
        with open(path) as fd:
            lines = fd.readlines()
            P_rect_left = \
                np.array([float(item) for item in
                            [line for line in lines if line.startswith('P_rect_02')][0].split()[1:]],
                            dtype=np.float32).reshape(3, 4)
            focallengths.append(-P_rect_left[0, 0])
            cxs.append(P_rect_left[0, 2])
            cys.append(P_rect_left[1, 2])
            constx.append(P_rect_left[0, 3])
            consty.append(P_rect_left[1, 3])
            constz.append(P_rect_left[2, 3])
    focallengths = np.array(focallengths)[:, None, None]
    cxs = np.array(cxs)[:, None, None]
    cys = np.array(cys)[:, None, None]
    constx = np.array(constx)[:, None, None]
    consty = np.array(consty)[:, None, None]
    constz = np.array(constz)[:, None, None]

    px1, py1 = project_3d_to_2d(pc1, f=focallengths, cx=cxs, cy=cys,
                                constx=constx, consty=consty, constz=constz)
    px2, py2 = project_3d_to_2d(predicted_pc2, f=focallengths, cx=cxs, cy=cys,
                                constx=constx, consty=consty, constz=constz)
    px2_gt, py2_gt = project_3d_to_2d(pc2, f=focallengths, cx=cxs, cy=cys,
                                        constx=constx, consty=consty, constz=constz)

    flow_x = px2 - px1
    flow_y = py2 - py1

    flow_x_gt = px2_gt - px1
    flow_y_gt = py2_gt - py1

    flow_pred = np.concatenate((flow_x[..., None], flow_y[..., None]), axis=-1)
    flow_gt = np.concatenate((flow_x_gt[..., None], flow_y_gt[..., None]), axis=-1)
    return flow_pred, flow_gt

def evaluate_2d(flow_pred, flow_gt):
    """
    flow_pred: (N, 2)
    flow_gt: (N, 2)
    """

    epe2d = np.linalg.norm(flow_gt - flow_pred, axis=-1)
    epe2d_mean = epe2d.mean()

    flow_gt_norm = np.linalg.norm(flow_gt, axis=-1)
    relative_err = epe2d / (flow_gt_norm + 1e-5)

    acc2d = (np.logical_or(epe2d < 3., relative_err < 0.05)).astype(np.float).mean()

    return epe2d_mean, acc2d


def compute_epe2d_ft3d(est_flow, batch):
    # pc1:torch.Tensor  B x n x 3
    pc1 = batch["sequence"][0]
    # true_flow:torch.Tensor B x n x 3
    true_flow = batch["ground_truth"][1]
    # mask : B x n x 1
    mask = batch["ground_truth"][0].cpu().numpy()[..., 0]
    pc1_np = pc1.cpu().numpy()[mask > 0]
    true_flow_np = true_flow.cpu().numpy()[mask > 0]
    est_flow_np = est_flow.cpu().numpy()[mask > 0]

    flow_pred, flow_gt = get_batch_2d_flow(pc1_np, pc1_np + true_flow_np, pc1_np + est_flow_np)
    
    epe2D, acc2d = evaluate_2d(flow_pred, flow_gt)
    epe2D = torch.tensor(epe2D).cuda(non_blocking=True)
    acc2d = torch.tensor(acc2d).cuda(non_blocking=True)
    return epe2D, acc2d


def compute_epe2d_ft3d_nomask(est_flow, batch):
    # pc1:torch.Tensor  B x n x 3
    pc1 = batch["sequence"][0]
    # true_flow:torch.Tensor B x n x 3
    true_flow = batch["ground_truth"][1]
    # mask : B x n x 1
    # mask = batch["ground_truth"][0].cpu().numpy()[..., 0]
    pc1_np = pc1.cpu().numpy()
    true_flow_np = true_flow.cpu().numpy()
    est_flow_np = est_flow.cpu().numpy()
    flow_pred, flow_gt = get_batch_2d_flow(pc1_np, pc1_np + true_flow_np, pc1_np + est_flow_np)
    
    epe2D, acc2d = evaluate_2d(flow_pred, flow_gt)

    epe2D = torch.tensor(epe2D).cuda(non_blocking=True)
    acc2d = torch.tensor(acc2d).cuda(non_blocking=True)
    return epe2D, acc2d


def compute_epe2d_kitti(est_flow, batch, paths):
    # pc1:torch.Tensor  B x n x 3
    pc1 = batch["sequence"][0]
    # true_flow:torch.Tensor B x n x 3
    true_flow = batch["ground_truth"][1]

    pc1_np = pc1.cpu().numpy()
    true_flow_np = true_flow.cpu().numpy()
    est_flow_np = est_flow.cpu().numpy()

    flow_pred, flow_gt = get_batch_2d_flow_kitti(pc1_np, pc1_np + true_flow_np, pc1_np + est_flow_np,paths)
    
    epe2D, acc2d = evaluate_2d(flow_pred, flow_gt)

    epe2D = torch.tensor(epe2D).cuda(non_blocking=True)
    acc2d = torch.tensor(acc2d).cuda(non_blocking=True)
    return epe2D, acc2d




def compute_epe_train(est_flow, batch):
    """
    Compute EPE during training.

    Parameters
    ----------
    est_flow : torch.Tensor
        Estimated flow.
    batch : flot.datasets.generic.Batch
        Contains ground truth flow and mask.

    Returns
    -------
    epe : torch.Tensor
        Mean EPE for current batch.

    """

    mask = batch["ground_truth"][0][..., 0]
    true_flow = batch["ground_truth"][1]
    error = est_flow - true_flow
    error = error[mask > 0]
    epe_per_point = torch.sqrt(torch.sum(torch.pow(error, 2.0), -1))
    epe = epe_per_point.mean()

    return epe


def compute_epe(est_flow, batch):
    """
    Compute EPE, accuracy and number of outliers.

    Parameters
    ----------
    est_flow : torch.Tensor
        Estimated flow.
    batch : flot.datasets.generic.Batch
        Contains ground truth flow and mask.

    Returns
    -------
    EPE3D : float
        End point error.
    acc3d_strict : float
        Strict accuracy.
    acc3d_relax : float
        Relax accuracy.
    outlier : float
        Percentage of outliers.

    """

    # Extract occlusion mask
    mask = batch["ground_truth"][0].cpu().numpy()[..., 0]

    # Flow
    sf_gt = batch["ground_truth"][1].cpu().numpy()[mask > 0]
    sf_pred = est_flow.cpu().numpy()[mask > 0]

    #
    if len(sf_gt)>0 and len(sf_pred)>0:
        l2_norm = np.linalg.norm(sf_gt - sf_pred, axis=-1)
        EPE3D = l2_norm.mean()
    else:
        l2_norm =(np.array([0])).astype(np.float)
        EPE3D =  l2_norm.mean()

    #
    sf_norm = np.linalg.norm(sf_gt, axis=-1)
    relative_err = l2_norm / (sf_norm + 1e-4)

    acc3d_strict_filter = np.logical_or(l2_norm < 0.05, relative_err < 0.05)
    if len(acc3d_strict_filter) != 0 :
        acc3d_strict = (
            acc3d_strict_filter.astype(np.float).mean()
        )
    else:
        acc3d_strict = (np.array([0])).astype(np.float).mean()

    acc3d_relax_filter = np.logical_or(l2_norm < 0.1, relative_err < 0.1)
    if len(acc3d_relax_filter) != 0 :
        acc3d_relax = (
            (np.logical_or(l2_norm < 0.1, relative_err < 0.1)).astype(np.float).mean()
        )
    else:
        acc3d_relax =(np.array([0])).astype(np.float).mean()
    
    outlier_filter = np.logical_or(l2_norm > 0.3, relative_err > 0.1)
    if len(outlier_filter) != 0 :
        outlier = outlier_filter.astype(np.float).mean()
    else:
        outlier= (np.array([0])).astype(np.float).mean()
    
    EPE3D = torch.tensor(EPE3D).cuda(non_blocking=True)
    acc3d_strict = torch.tensor(acc3d_strict).cuda(non_blocking=True)
    acc3d_relax = torch.tensor(acc3d_relax).cuda(non_blocking=True)
    outlier = torch.tensor(outlier).cuda(non_blocking=True)

    return EPE3D, acc3d_strict, acc3d_relax, outlier






def compute_epe_nomask(est_flow, batch):
    """
    Compute EPE, accuracy and number of outliers.

    Parameters
    ----------
    est_flow : torch.Tensor
        Estimated flow.
    batch : flot.datasets.generic.Batch
        Contains ground truth flow and mask.

    Returns
    -------
    EPE3D : float
        End point error.
    acc3d_strict : float
        Strict accuracy.
    acc3d_relax : float
        Relax accuracy.
    outlier : float
        Percentage of outliers.

    """

    # Extract occlusion mask
    # mask = batch["ground_truth"][0].cpu().numpy()[..., 0]

    # Flow
    sf_gt = batch["ground_truth"][1].cpu().numpy()
    sf_pred = est_flow.cpu().numpy()

    #
    if len(sf_gt)>0 and len(sf_pred)>0:
        l2_norm = np.linalg.norm(sf_gt - sf_pred, axis=-1)
        EPE3D = l2_norm.mean()
    else:
        l2_norm =(np.array([0])).astype(np.float)
        EPE3D =  l2_norm.mean()

    #
    sf_norm = np.linalg.norm(sf_gt, axis=-1)
    relative_err = l2_norm / (sf_norm + 1e-4)

    acc3d_strict_filter = np.logical_or(l2_norm < 0.05, relative_err < 0.05)
    if len(acc3d_strict_filter) != 0 :
        acc3d_strict = (
            acc3d_strict_filter.astype(np.float).mean()
        )
    else:
        acc3d_strict = (np.array([0])).astype(np.float).mean()

    acc3d_relax_filter = np.logical_or(l2_norm < 0.1, relative_err < 0.1)
    if len(acc3d_relax_filter) != 0 :
        acc3d_relax = (
            (np.logical_or(l2_norm < 0.1, relative_err < 0.1)).astype(np.float).mean()
        )
    else:
        acc3d_relax =(np.array([0])).astype(np.float).mean()
    
    outlier_filter = np.logical_or(l2_norm > 0.3, relative_err > 0.1)
    if len(outlier_filter) != 0 :
        outlier = outlier_filter.astype(np.float).mean()
    else:
        outlier= (np.array([0])).astype(np.float).mean()
    
    EPE3D = torch.tensor(EPE3D).cuda(non_blocking=True)
    acc3d_strict = torch.tensor(acc3d_strict).cuda(non_blocking=True)
    acc3d_relax = torch.tensor(acc3d_relax).cuda(non_blocking=True)
    outlier = torch.tensor(outlier).cuda(non_blocking=True)

    return EPE3D, acc3d_strict, acc3d_relax, outlier