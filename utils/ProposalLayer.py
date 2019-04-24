import numpy as np
import torch as t
import torch.nn as nn


class ProposalLayer:
    def __init__(self,
                 extractor,
                 nms_thresh=0.7,
                 n_pre_nms_train = 12000,
                 n_post_nms_train = 2000,
                 n_pre_nms_test = 6000,
                 n_post_nms_test = 300,
                 min_bbox_size = 16):
        self.extractor = extractor
        self.nms_thresh  = nms_thresh
        self.n_pre_nms_train = n_pre_nms_train
        self.n_post_nms_train = n_post_nms_train
        self.n_pre_nms_test = n_pre_nms_test
        self.n_post_nms_test = n_post_nms_test
        self.min_bbox_size = min_bbox_size

    def __call__(self, cls, reg, anchor, img_size, img_scale):
        if self.extractor.training:
            n_pre_nms = self.n_pre_nms_train
            n_post_nms = self.n_post_nms_train
        else:
            n_pre_nms = self.n_pre_nms_test
            n_post_nms = self.n_post_nms_test

        


