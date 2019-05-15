import numpy as np
from collections import defaultdict
from models.utils.boundingbox import compute_iou_xp


def compare_iterable_length(iterables):
    # TODO
    pass

def compute_voc_detection_precision_recall(
        pred_bbox, pred_label, pred_cls, gt_bbox, gt_label, iou_thresh=0.5
):
    # iterable of np.ndarray, ~, ~, ~, ~, float
    # B batches of sets (*_bat, 4), bat varies from 0 to B-1 (*_0, *_1, ...), *_bat bbox
    pred_bbox  = iter(pred_bbox)   # B * (N_bat, 4), (x_min, y_min, x_max, y_max)
    pred_label = iter(pred_label)  # B * (N_bat,  )
    pred_cls   = iter(pred_cls)    # B * (N_bat,  )
    gt_bbox    = iter(gt_bbox)     # B * (K_bat, 4), (x_min, y_min, x_max, y_max)
    gt_label   = iter(gt_label)    # B * (K_bat,  )

    n_pos = defaultdict(int)
    cls   = defaultdict(list)
    match = defaultdict(list)

    for pred_bbox_, pred_label_, pred_cls_, gt_bbox_, gt_label_ in zip(
        pred_bbox,  pred_label,  pred_cls,  gt_bbox,  gt_label
    ):
        for l in np.unique(np.concatenate((pred_label_, gt_label_)).astype(int)):
            pred_mask_l = pred_label_ == l  # Lp, say, labels are l
            pred_bbox_l = pred_bbox_[pred_mask_l]  # (Lp, 4)
            pred_cls_l  = pred_cls_[pred_mask_l]   # (Lp, )

            order = pred_cls_l.argsort()[::-1]
            pred_bbox_l = pred_bbox_l[order]
            pred_cls_l  = pred_cls_l[order]

            gt_mask_l = gt_label_ == l  # Lg, say, labels are l
            gt_bbox_l = gt_bbox_[gt_mask_l]  # (Lg, 4)

            n_pos[l] += gt_mask_l.sum()  # number of positive gt, Lg
            cls[l].extend(pred_cls_l)  # add p_c_l to list cls[l]

            if len(pred_bbox_l) == 0:
                continue
            if len(gt_bbox_l) == 0:
                # 0 means no match for all bbox in this img, with amount of p_b_l.shape[0] (N)
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
                    if not select[gt_idx_]:  # if gt bbox not selected yet
                        match[l].append(1)
                    else:
                        match[l].append(0)  # already selected
                    select[gt_idx_] = True  # indicate this gt bbox is occupied
                else:
                    match[l].append(0)  # -1, low confidence

    # call the next of each iterable to determine whether there are elements left
    for iter_ in (pred_bbox, pred_label, pred_cls, gt_bbox, gt_label):
        if next(iter_, None) is not None:
            raise ValueError('lengths of iterables do not match.')

    n_fg_class = max(n_pos.keys()) + 1
    precision = [None] * n_fg_class
    recall = [None] * n_fg_class

    for l in n_pos.keys():
        # all matches of label l from all images
        cls_l   = np.array(cls[l])
        match_l = np.array(match[l], dtype=np.int8)

        order = cls_l.argsort()[::-1]  # from high to low
        match_l = match_l[order]

        # not the same as the usual meanings of TP and FP, positives here means positive bbox
        true_positives  = np.cumsum(match_l == 1)
        false_positives = np.cumsum(match_l == 0)
        precision[l] = true_positives / (true_positives + false_positives)
        if n_pos[l] > 0:
            # n_pos[l] indicates number of total gts of class l(real TP + FN)
            recall[l] = true_positives / n_pos[l]

    return precision, recall
