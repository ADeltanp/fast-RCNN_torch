import numpy as np
import cupy as cp
from .boundingbox import bbox2t_encoded, iou


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

    def __call__(self, gt_bbox, anchor):
        max_iou_id, labels = self._label(anchor, gt_bbox)

        t_star = bbox2t_encoded(anchor, gt_bbox[max_iou_id, :])
        return t_star, labels

    def _label(self, anchor, gt_bbox):
        # cupy compatible TODO Compatibility Not Tested
        xp = cp.get_array_module(anchor)
        labels = xp.empty((len(anchor),), dtype=xp.int32)  # shape (N, )
        labels.fill(-1)

        max_iou_id, max_iou, gt_max_iou_id = self._ious(anchor, gt_bbox)
        # label those < neg_thresh, > pos_thresh, highest iou wrt any gt
        labels[max_iou < self.negative_thresh] = 0
        labels[gt_max_iou_id] = 1
        labels[max_iou > self.positive_thresh] = 1

        n_positives = int(self.pn_ratio * self.n_sample)
        positive_id = xp.where(labels == 1)[0]
        if len(positive_id) > n_positives:
            discard_id = xp.random.choice(positive_id,
                                          size=(len(positive_id) - n_positives),
                                          replace=False)
            labels[discard_id] = -1

        n_negatives = self.n_sample - n_positives
        negative_id = xp.where(labels == 0)[0]
        if len(negative_id) > n_negatives:
            discard_id = xp.random.choice(negative_id,
                                          size=(len(negative_id) - n_negatives),
                                          replace=False)
            labels[discard_id] = -1

        return max_iou_id, labels

    def _ious(self, anchor, gt_bbox):
        # cupy compatible TODO Compatibility Not Tested
        xp = cp.get_array_module(anchor)
        ious = iou(anchor, gt_bbox)  # shape (N, K), N anchors, K gt

        # axis 0 is the max anchor of every gt_bbox, axis 1 is gt_bbox of every anchor
        max_iou_id = ious.argmax(axis=1)  # shape (N, )
        gt_max_iou_id = ious.argmax(axis=0)  # shape (K, )

        max_iou = ious[xp.arange(len(anchor)), max_iou_id]  # shape (N, ), max iou of each anchor
        gt_max_iou = ious[gt_max_iou_id, xp.arange(ious.shape[1])]  # shape (K, ), max of each gt
        gt_max_iou_id = xp.where(ious == gt_max_iou[:, xp.newaxis])[0]  # retrieve which anchor has max iou with gt

        return max_iou_id, max_iou, gt_max_iou_id
