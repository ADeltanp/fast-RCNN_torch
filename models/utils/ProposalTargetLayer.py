import cupy as cp
from .boundingbox import compute_iou_xp, bbox2t_encoded


class ProposalTargetLayer:
    def __init__(self,
                 n_sample=128,
                 positive_ratio=0.25,
                 positive_iou_thresh=0.5,
                 negative_iou_thresh_hi=0.5,
                 negative_iou_thresh_lo=0.0):
        self.n_sample = n_sample
        self.positive_ratio = positive_ratio
        self.positive_iou_thresh = positive_iou_thresh
        self.negative_iou_thresh_hi = negative_iou_thresh_hi
        self.negative_iou_thresh_lo = negative_iou_thresh_lo

    def __call__(self, roi, gt_bbox, gt_label,
                 t_normalize_mean=(0.0, 0.0, 0.0, 0.0),
                 t_normalize_std=(0.1, 0.1, 0.2, 0.2)):
        '''
        :param roi: of shape (N, 4), with (x, y, x, y)
        :param gt_bbox: of shape (K, 4), with (x, y, x, y)
        :param gt_label: of shape (K, )
        :param t_normalize_mean:
        :param t_normalize_std:
        :return:
        '''
        # cupy compatible TODO Compatibility Not Tested
        xp = cp.get_array_module(roi)

        n_positives = xp.round(self.positive_ratio * self.n_sample)
        iou = compute_iou_xp(roi, gt_bbox)       # (N, K), N roi, K gt
        max_iou_gt_id = iou.argmax(axis=1)       # (N,  ), gt id of max iou w.r.t each roi
        max_iou = iou.max(axis=1)                # (N,  ), get the max iou w.r.t each roi
        roi_label = gt_label[max_iou_gt_id] + 1  # assign gt label to each roi, 0 reserved for bg

        positive_id = xp.where(max_iou >= self.positive_iou_thresh)[0]
        positives = int(min(n_positives, len(positive_id)))
        if len(positive_id) > 0:
            positive_id = xp.random.choice(positive_id, size=positives, replace=False)

        negative_id = xp.where((max_iou < self.negative_iou_thresh_hi) &
                               (max_iou >= self.negative_iou_thresh_lo))[0]
        negatives = self.n_sample - positives
        negatives = int(min(negatives, len(negative_id)))
        if len(negative_id) > 0:
            negative_id = xp.random.choice(negative_id, size=negatives, replace=False)

        keep = xp.append(positive_id, negative_id)
        roi_label = roi_label[keep]  # sort rois to pos:neg
        roi_label[positives:] = 0  # all negatives are considered as bg
        sample_roi = roi[keep]  # hence sample_roi[i]'s label is roi_label[i]

        # max_iou_gt_id[keep] is sorted id of [pos:neg],
        # thus gt_bbox[...] is corresponding bbox of sample
        # here we compute t_star of the original paper
        t_star = bbox2t_encoded(sample_roi, gt_bbox[max_iou_gt_id[keep]])
        t_star = ((t_star - xp.array(t_normalize_mean, xp.float32)) /
                  xp.array(t_normalize_std, xp.float32))

        # the first two are (x, y, h, w)
        # (n_sample, 4), (n_sample, 4), (n_sample, ), xp.ndarray
        # reminds that label here is ACTUAL target label, from 0 to n_class,
        # rather than that 'v_label' used in anchor target layer to denote the validity
        return sample_roi, t_star, roi_label
