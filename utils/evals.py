import numpy as np
from collections import defaultdict
from models.utils.boundingbox import compute_iou_xp


def compute_voc_detection_precision_recall(
        pred_bbox, pred_label, pred_cls, gt_bbox, gt_label, iou_thresh=0.5
):
    # iterable of np.ndarray, ~, ~, ~, ~, float
    # B batches of sets (N_bat, 4), bat varies from 0 to B-1
    pred_bbox  = iter(pred_bbox)
    pred_label = iter(pred_label)
    pred_cls   = iter(pred_cls)
    gt_bbox    = iter(gt_bbox)
    gt_label   = iter(gt_label)

    n_pos = defaultdict(int)
    cls   = defaultdict(list)
    match = defaultdict(list)

    for pred_bbox_, pred_label_, pred_cls_, gt_bbox_, gt_label_ in zip(
        pred_bbox,  pred_label,  pred_cls,  gt_bbox,  gt_label
    ):
        gt_diff = np.zeros(gt_bbox_.shape[0], dtype=bool)
        for l in np.unique(np.concatenate((pred_label_, gt_label_)).astype(int)):
            pred_mask_l = pred_label_ == l  # Lp, say, labels are l
            pred_bbox_l = pred_bbox_[pred_mask_l]  # (Lp, 4)
            pred_cls_l  = pred_cls_[pred_mask_l]   # (Lp, )

            order = pred_cls_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_cls_l  = pred_cls_l[order]

            gt_mask_l = gt_label_ == l  # Lg, say, labels are l
            gt_bbox_l = gt_bbox_[gt_mask_l]  # (Lg, 4)
            gt_diff_l = gt_diff[gt_mask_l]

            n_pos[l] += np.logical_not(gt_diff[gt_mask_l]).sum()
            cls[l].extend(pred_cls_l)

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                # 0 means no match for all bbox in this img, with amount of p_b_l.shape[0] (N_bat)
                match[l].extend((0,) * pred_bbox_l.shape[0])
                continue

            pred_bbox_l = pred_bbox_l.copy()
            gt_bbox_l   = gt_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1
            gt_bbox_l[:, 2:]   += 1

            iou = compute_iou_xp(pred_bbox_l, gt_bbox_l)  # (Lp, Lg) returned
            gt_idx = iou.argmax(axis=1)  # (Lp, ), gt index for each pred_bbox_l
            gt_idx[iou.max(axis=1) < iou_thresh] = -1  # set those low confidence bbox to -1
            del iou

            select = np.zeros(gt_bbox_l.shape[0], dtype=bool)  # (Lp, )
            for gt_idx_ in gt_idx:
                if gt_idx_ >= 0:  # matching gt bbox exists
                    if gt_diff_l[gt_idx_]:  # if too difficult, set to -1
                        match[l].append(-1)
                    else:
                        if not select[gt_idx_]:  # if gt bbox not selected yet
                            match[l].append(1)
                        else:
                            match[l].append(0)  # already selected
                    select[gt_idx_] = True  # this gt bbox is occupied
                else:
                    match[l].append(0)  # -1, means low confidence, no matching
