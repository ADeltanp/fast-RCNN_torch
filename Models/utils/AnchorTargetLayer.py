import torch as t
import numpy as np
import cupy as cp


class AnchorTargetLayer:
    def __init__(self,
                 n_sample=256,
                 positive_thresh=0.7,
                 negative_thresh=0.3,
                 pn_ratio=0.5):
        self.n_sample = n_sample
        self.positive_thresh = 0.7
        self.negative_thresh = 0.3
        self.pn_ratio = pn_ratio

    def __call__(self, gt_bbox, anchor, img_size):
        img_h, img_w = img_size
        n_anchor = len(anchor)
        max_iou_id, labels = self._label(anchor, gt_bbox)


    def _label(self, anchor, gt_bbox):
        labels = np.empty((len(anchor),), dtype=np.int32)
        labels.fill(-1)

        max_iou_id, max_iou, gt_max_iou_id = self._ious(anchor, gt_bbox)
        # label those < neg_thresh, > pos_thresh, highest iou wrt any gt
        labels[max_iou < self.negative_thresh] = 0
        labels[gt_max_iou_id] = 1
        labels[max_iou > self.positive_thresh] = 1

        n_positives = int(self.pn_ratio * self.n_sample)
        positive_id = np.where(labels == 1)[0]
        if len(positive_id) > n_positives:
            discard_id = np.random.choice(positive_id,
                                          size=(len(positive_id) - n_positives),
                                          replace=False)
            labels[discard_id] = -1

        n_negatives = self.n_sample - n_positives
        negative_id = np.where(labels == 0)[0]
        if len(negative_id) > n_negatives:
            discard_id = np.random.choice(negative_id,
                                          size=(len(negative_id) - n_negatives),
                                          replace=False)
            labels[discard_id] = -1

        return max_iou_id, labels

    def _ious(self, anchor, gt_bbox):
        ious = iou(anchor, gt_bbox)  # shape (N, K)

        # axis 0 is every anchor, axis 1 is every gt_bbox
        max_iou_id = ious.argmax(axis=1)
        gt_max_iou_id = ious.argmax(axis=0)

        max_iou = ious[np.arange(len(anchor)), max_iou_id]  # shape (N, ), max iou of each anchor
        gt_max_iou = ious[gt_max_iou_id, np.arange(ious.shape[1])]  # shape (K, ), max of each gt
        gt_max_iou_id = np.where(ious == gt_max_iou[:, np.newaxis])[0]  # retrieve which anchor has max iou with gt

        return max_iou_id, max_iou, gt_max_iou_id






def iou(bbox_a, bbox_b):
    # xp stands for either np or cp according to the type of inputs
    xp = cp.get_array_module(bbox_a)

    # newaxis here is to utilize broadcasting to compare between each element in a and b
    top_left     = xp.maximum(bbox_a[:, xp.newaxis, :2], bbox_b[:, :2])  # shape (N, K, 2)
    bottom_right = xp.minimum(bbox_a[:, xp.newaxis, 2:], bbox_b[:, 2:])  # shape (N, K, 2)

    # shapes: (N, K), (N, ), (K, ), returns (N, K)
    iou_area = xp.prod(bottom_right - top_left, axis=2) * (top_left < bottom_right).all(axis=2)
    area_a = xp.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = xp.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return iou_area / (area_a[:, xp.newaxis] + area_b - iou_area)




