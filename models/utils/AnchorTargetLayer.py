import cupy as cp
from .boundingbox import bbox2t_encoded, compute_iou_xp


class AnchorTargetLayer:
    def __init__(self,
                 n_sample=256,
                 positive_thresh=0.7,
                 negative_thresh=0.3,
                 positive_ratio=0.5):
        self.n_sample = n_sample
        self.positive_thresh = positive_thresh
        self.negative_thresh = negative_thresh
        self.positive_ratio = positive_ratio

    def __call__(self, gt_bbox, anchor):
        max_iou_id, labels = self._label(anchor, gt_bbox)

        t_star = bbox2t_encoded(anchor, gt_bbox[max_iou_id, :])
        return t_star, labels  # (N, 4), (N, ), xp.ndarray

    def _label(self, anchor, gt_bbox):
        # label: invalid -> -1; negative -> 0; positive -> 1
        # cupy compatible TODO Compatibility Not Tested
        xp = cp.get_array_module(anchor)
        labels = xp.empty((len(anchor),), dtype=xp.int32)  # shape (N, )
        labels.fill(-1)

        max_iou_id, max_iou, gt_max_iou_id = self._ious(anchor, gt_bbox)
        # label those < neg_thresh, > pos_thresh, highest iou wrt any gt
        labels[max_iou < self.negative_thresh] = 0
        labels[gt_max_iou_id] = 1
        labels[max_iou > self.positive_thresh] = 1

        n_positives = int(self.positive_ratio * self.n_sample)
        positive_id = xp.where(labels == 1)[0]
        if len(positive_id) > n_positives:
            discard_id = xp.random.choice(positive_id,
                                          size=(len(positive_id) - n_positives),
                                          replace=False)
            labels[discard_id] = -1

        n_negatives = self.n_sample - xp.sum(labels == 1)
        negative_id = xp.where(labels == 0)[0]
        if len(negative_id) > n_negatives:
            discard_id = xp.random.choice(negative_id,
                                          size=(len(negative_id) - n_negatives),
                                          replace=False)
            labels[discard_id] = -1

        return max_iou_id, labels  # (N, ), (N, ), xp.ndarray

    def _ious(self, anchor, gt_bbox):
        # cupy compatible TODO Compatibility Not Tested
        xp = cp.get_array_module(anchor)
        ious = compute_iou_xp(anchor, gt_bbox)  # shape (N, K), N anchors, K gt

        # axis 0 gets the max anchor of every gt_bbox, axis 1 gets gt_bbox of every anchor
        max_iou_id = ious.argmax(axis=1)  # shape (N, ), assign N anchor with gt id
        gt_max_iou_id = ious.argmax(axis=0)  # shape (K, ), assign K gt with anchor id

        # shape (N, ), max iou of each anchor (max within row, not id but data
        anchor_max_iou = ious[xp.arange(ious.shape[0]), max_iou_id]
        # shape (K, ), max iou of each gt (max within column, not id but data)
        gt_max_iou = ious[gt_max_iou_id, xp.arange(ious.shape[1])]
        # retrieve which anchor has max iou with gt, (N, K) == (K, ),
        # as only the top K anchors have identical iou with one of gt_max_iou,
        # hence output of shape (K, )
        gt_max_iou_id = xp.where(ious == gt_max_iou)[0]  # (K, )

        return max_iou_id, anchor_max_iou, gt_max_iou_id  # (N, ), (K, ), (K, ), xp.ndarray
