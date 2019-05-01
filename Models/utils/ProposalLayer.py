import cupy as cp
from Models.utils.boundingbox import t_encoded2bbox
from Models.utils.nms.non_maximum_suppression import non_maximum_suppression as NMS


class ProposalLayer:
    def __init__(self,
                 extractor,
                 nms_thresh=0.7,
                 n_pre_nms_train=12000,
                 n_post_nms_train=2000,
                 n_pre_nms_test=6000,
                 n_post_nms_test=300,
                 min_bbox_size=16):

        self.extractor = extractor
        self.nms_thresh = nms_thresh
        self.n_pre_nms_train  = n_pre_nms_train
        self.n_post_nms_train = n_post_nms_train
        self.n_pre_nms_test   = n_pre_nms_test
        self.n_post_nms_test  = n_post_nms_test
        self.min_bbox_size = min_bbox_size

    def __call__(self, cls, reg, anchor, img_size, img_scale, phase):
        '''
        :param cls: (numpy array) cls output by RPN
        :param reg: (numpy array) reg output by RPN
        :param anchor: (xp array) generated in RPN
        :param img_size: (tuple of ints) array of original image (h, w)
        :param img_scale: (float) image scaling factor
                          in data processing section
        :param phase: (string) either 'train' or 'test'
        :return: region of interest of shape (N, 4), N rois in total,
                 the 4 is (x_min, y_min, x_max, y_max)
        '''
        # cupy compatible TODO Compatibility Not Tested
        xp = cp.get_array_module(anchor)
        if xp is cp:
            cls = cp.asarray(cls)
            reg = cp.asarray(reg)

        if phase is 'train':
            n_pre_nms  = self.n_pre_nms_train
            n_post_nms = self.n_post_nms_train
        else:
            n_pre_nms  = self.n_pre_nms_test
            n_post_nms = self.n_post_nms_test

        rois = t_encoded2bbox(anchor, reg)
        rois[:, slice(0, 4, 2)] = xp.clip(rois[:, slice(0, 4, 2)], 0, img_size[0])  # clipping x-axis
        rois[:, slice(1, 4, 2)] = xp.clip(rois[:, slice(1, 4, 2)], 0, img_size[1])  # clipping y-axis

        # discard bbox whose size < min_bbox_size
        min_bbox_size = self.min_bbox_size * img_scale
        w = rois[:, 2] - rois[:, 0]
        h = rois[:, 3] - rois[:, 1]
        keep = xp.where((w >= min_bbox_size) & (h >= min_bbox_size))[0]
        rois = rois[keep, :]
        cls = cls[keep, :]

        # sort bbox according to cls probability
        order = xp.argsort(-cls[:, 0])
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        rois = rois[order, :]

        # TODO Implement NMS by myself
        keep = NMS(rois, thresh=self.nms_thresh)
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        rois = rois[keep]
        return rois
