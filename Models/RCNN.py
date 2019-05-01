import torch as t
import torch.nn as nn
import numpy as np
from .utils.roi_pooling.roi_pooling_layer import RoIPooling2D


class RCNN(nn.Module):
    def __init__(self,
                 n_class,
                 roi_size,
                 roi_pooled_shape=7,
                 feat_receptive_len=16):
        super(RCNN, self).__init__()
        self.n_class = n_class + 1
        self.roi_size = roi_size

        self.roi_pool = RoIPooling2D(roi_pooled_shape, feat_receptive_len)
        self.fcs = nn.Sequential(  # fc6 & fc7 of original VGG16
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )
        self.cls = nn.Linear(4096, self.n_class)
        self.reg = nn.Linear(4096, self.n_class * 4)

    def forward(self, feat, rois, roi_batch_id):
        '''
        :param feat: (torch.Tensor) feature map output by extractor
        :param rois: (numpy array) rois of shape (bat * roi_per_image, 4)
        :param roi_batch_id: (numpy array) idx indicating the batch of roi
                             of shape (bat * roi_per_image, )
        :return: (torch.Tensor) cls and reg output by RCNN
        '''

        # not requires grad as no backprop at RPN
        rois = t.from_numpy(rois)
        roi_batch_id = t.from_numpy(roi_batch_id[:, np.newaxis])
        id_rois = t.cat((roi_batch_id, rois), dim=1)

        id_rois = id_rois.contiguous()
        feat = feat.contiguous()

        pooled_feat = self.roi_pool(feat, id_rois)
        pooled_feat = pooled_feat.view(pooled_feat.size(0), -1)
        shared = self.fcs(pooled_feat)
        cls = self.cls(shared)
        reg = self.reg(shared)
        return cls, reg
