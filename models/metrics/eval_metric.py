import torch
import torch.nn.functional as F
import numpy as np

@torch.no_grad()
def compute_depth_errors(gt, pred, align=True, dataset='MS2'):
    # pred : b c h w
    # gt: b h w

    abs_diff = abs_rel = sq_rel = log10 = rmse = rmse_log = a1 = a2 = a3 = 0.0

    batch_size, h, w = gt.size()

    if pred.nelement() != gt.nelement():
        pred = F.interpolate(pred, [h, w], mode='nearest')

    pred = pred.view(batch_size, h, w)

    if (dataset == 'ViViD') or (dataset == 'MS2'):
        crop_mask = gt[0] != gt[0]
        crop_mask[:, :] = 1
        max_depth = 80
    else:
        crop_mask = gt[0] != gt[0]
        crop_mask[:, :] = 1
        max_depth = 100

    min_depth = 1e-3
    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > min_depth) & (current_gt < max_depth)
        valid = valid & crop_mask

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid]

        # align scale
        if align:
            valid_pred = valid_pred * \
                torch.median(valid_gt)/torch.median(valid_pred)

        valid_pred = valid_pred.clamp(min_depth, max_depth)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        diff_i = valid_gt - valid_pred
        abs_diff += torch.mean(torch.abs(diff_i))
        abs_rel += torch.mean(torch.abs(diff_i) / valid_gt)
        sq_rel += torch.mean(((diff_i)**2) / valid_gt)
        rmse += torch.sqrt(torch.mean(diff_i ** 2))
        rmse_log += torch.sqrt(torch.mean((torch.log(valid_gt) -
                               torch.log(valid_pred)) ** 2))
        log10 += torch.mean(torch.abs((torch.log10(valid_gt) -
                            torch.log10(valid_pred))))

    return [metric.item() / batch_size for metric in [abs_diff, abs_rel, sq_rel, log10, rmse, rmse_log, a1, a2, a3]]
