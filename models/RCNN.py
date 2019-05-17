import torch as t
import torch.nn as nn
import numpy as np
from torchvision import models
from utils import converter
from .utils.roi_pooling.roi_pooling_layer import RoIPooling2D


class RCNN(nn.Module):
    def __init__(self,
                 n_class,
                 init_mean,
                 init_std,
                 roi_pooled_shape=7,
                 feat_receptive_len=16):
        super().__init__()
        self.n_class = n_class + 1

        self.roi_pool = RoIPooling2D(roi_pooled_shape, feat_receptive_len)
        self.fcs = nn.Sequential(  # fc6 & fc7 of original VGG16
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU()
        )
        self.cls = nn.Linear(4096, self.n_class)
        self.reg = nn.Linear(4096, self.n_class * 4)

        self._init_params(init_mean, init_std)

    def forward(self, feat, rois, roi_batch_id):
        '''
        :param feat: (torch.Tensor) feature map output by extractor
        :param rois: (numpy array) rois of shape (bat * roi_per_image, 4)
        :param roi_batch_id: (numpy array) idx indicating the batch of roi
                             of shape (bat * roi_per_image, )
        :return: (torch.Tensor) cls and reg output by RCNN
        '''

        # not requires grad as no backprop at RPN
        rois = converter.to_tensor(rois)
        roi_batch_id = converter.to_tensor(roi_batch_id[:, np.newaxis])
        id_rois = t.cat((roi_batch_id, rois), dim=1)

        id_rois = id_rois.contiguous()
        feat = feat.contiguous()

        pooled_feat = self.roi_pool(feat, id_rois)
        pooled_feat = pooled_feat.view(pooled_feat.size(0), -1)
        shared = self.fcs(pooled_feat)
        cls = self.cls(shared)
        reg = self.reg(shared)
        return cls, reg  # (B * roi_per_image, n_class \and\ n_class * 4), torch.Tensor

    def _init_params(self, mean, std):
        temp = models.vgg16(pretrained=True).classifier
        self.fcs[0].weight.data = temp[0].weight.data
        self.fcs[2].weight.data = temp[3].weight.data

        self.fcs[0].bias.data = temp[0].bias.data
        self.fcs[2].bias.data = temp[3].bias.data

        self.cls.weight.data.normal_(mean, std)
        self.reg.weight.data.normal_(mean, std)

        self.cls.bias.data.zero_()
        self.reg.bias.data.zero_()
