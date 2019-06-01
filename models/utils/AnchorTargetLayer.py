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

    def __call__(self, gt_bbox, anchor, img_size):
        xp = cp.get_array_module(anchor)

        n_anchor = len(anchor)
        in_index = xp.where(
            (anchor[:, 0] >= 0) &
            (anchor[:, 1] >= 0) &
            (anchor[:, 2] <= img_size[1]) &
            (anchor[:, 3] <= img_size[0])
        )[0]
        in_anchor = anchor[in_index, :]  # (I, 4), I anchors inside the image

        max_iou_id, v_label = self._label(in_anchor, gt_bbox, in_index)
        t_star = bbox2t_encoded(in_anchor, gt_bbox[max_iou_id, :])

        v_label = _unmap(v_label, n_anchor, in_index, fill=-1)
        t_star = _unmap(t_star, n_anchor, in_index, fill=0)
        return t_star, v_label  # (N, 4), (N, ), xp.ndarray

    def _label(self, anchor, gt_bbox, in_index):
        # v_label: invalid(ignore) -> -1; negative -> 0; positive -> 1
        # cupy compatible TODO Compatibility Not Tested
        xp = cp.get_array_module(anchor)
        v_label = xp.empty((len(in_index),), dtype=xp.int32)  # shape (I, )
        v_label.fill(-1)

        max_iou_id, max_iou, gt_max_iou_id = self._ious(anchor, gt_bbox, in_index)
        # label those < neg_thresh, > pos_thresh, highest iou wrt any gt
        v_label[max_iou < self.negative_thresh] = 0
        v_label[gt_max_iou_id] = 1
        v_label[max_iou > self.positive_thresh] = 1

        n_positives = int(self.positive_ratio * self.n_sample)
        positive_id = xp.where(v_label == 1)[0]
        if len(positive_id) > n_positives:
            discard_id = xp.random.choice(positive_id,
                                          size=(len(positive_id) - n_positives),
                                          replace=False)
            v_label[discard_id] = -1

        n_negatives = self.n_sample - xp.sum(v_label == 1)
        negative_id = xp.where(v_label == 0)[0]
        if len(negative_id) > n_negatives:
            discard_id = xp.random.choice(negative_id,
                                          size=(len(negative_id) - n_negatives),
                                          replace=False)
            v_label[discard_id] = -1

        return max_iou_id, v_label  # (I, ), (I, ), xp.ndarray

    def _ious(self, anchor, gt_bbox, i_idx):
        # cupy compatible TODO Compatibility Not Tested
        xp = cp.get_array_module(anchor)
        ious = compute_iou_xp(anchor, gt_bbox)  # shape (I, K), I anchors, K gt

        # axis 0 gets the max anchor w.r.t. each gt_bbox, axis 1 gets gt_bbox w.r.t. each anchor
        max_iou_id = ious.argmax(axis=1)  # shape (I, ), assign each anchor with gt id
        gt_max_iou_id = ious.argmax(axis=0)  # shape (K, ), assign each gt with anchor id

        # shape (I, ), max iou of each anchor (max within row, not id but data)
        anchor_max_iou = ious[xp.arange(len(i_idx)), max_iou_id]

        # shape (K, ), max iou of each gt (max within column, not id but data)
        gt_max_iou = ious[gt_max_iou_id, xp.arange(ious.shape[1])]

        # retrieve which anchor has max iou with gt, (I, K) == (K, ),
        # as only the top K anchors have identical iou with one of gt_max_iou,
        # hence output of shape (K, )
        gt_max_iou_id = xp.where(ious == gt_max_iou)[0]  # (K, )

        return max_iou_id, anchor_max_iou, gt_max_iou_id  # (I, ), (I, ), (K, ), xp.ndarray


def _unmap(data, size, index, fill=0):
    xp = cp.get_array_module(data)
    if len(data.shape) == 1:
        result = xp.empty((size,), dtype=data.dtype)
        result.fill(fill)
        result[index] = data
    else:
        result = xp.empty((size,) + data.shape[1:], dtype=data.dtype)
        result.fill(fill)
        result[index, :] = data
    return result
