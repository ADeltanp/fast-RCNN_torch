import os
import torch as t
import torch.nn as nn
import utils.converter as converter

from collections import namedtuple
from torchnet.meter import ConfusionMeter, AverageValueMeter
from utils.config import config
from models.utils.AnchorTargetLayer import AnchorTargetLayer
from models.utils.ProposalTargetLayer import ProposalTargetLayer

LossTuple = namedtuple('LossTuple',
                       ['rpn_reg_loss',
                        'rpn_cls_loss',
                        'roi_reg_loss',
                        'roi_cls_loss',
                        'total_loss'])


class TrainHelper(nn.Module):
    def __init__(self, faster_rcnn):
        super(TrainHelper, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = config.rpn_sigma
        self.roi_sigma = config.roi_sigma

        self.anchor_target_layer = AnchorTargetLayer()
        self.proposal_target_layer = ProposalTargetLayer()

        self.reg_normalize_mean = faster_rcnn.reg_normalize_mean
        self.reg_normalize_std = faster_rcnn.reg_normalize_std

        self.optimizer = self.faster_rcnn.get_optimizer()

        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(config.n_class + 1)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}

    def forward(self, img, bbox, label, scale):
        '''
        :param img: (torch.autograd.Variable) a batch of images
        :param bbox: (torch.autograd.Variable) a batch of bbox of shape (B, N, 4)
        :param label: (torch.autograd.Variable) a batch of labels of shape (B, N)
                      background is excluded
        :param scale: (float) scale factor during preprocess
        :return: namedtuple of 5 losses
        '''
        n = bbox.shape[0]
        if n != 1:
            raise ValueError('Only support batch size of 1')

        _, _, h, w = img.shape
        img_size = (h, w)

        feat = self.faster_rcnn.extractor(img)
        rpn_reg, rpn_cls, rois, roi_id, anchor = self.faster_rcnn.RPN(feat, img_size, scale)

        bbox = bbox[0]
        label = label[0]
        rpn_cls = rpn_cls[0]
        rpn_reg = rpn_reg[0]
        roi = rois

        sample_roi, gt_roi_reg, gt_roi_label = self.proposal_target_layer(
            roi,
            converter.to_numpy(bbox),
            converter.to_numpy(label),
            self.reg_normalize_mean,
            self.reg_normalize_std
        )
        sample_roi_idx = t.zeros(len(sample_roi))
        roi_reg, roi_cls = self.faster_rcnn.RCNN(feat, sample_roi, sample_roi_idx)

        # RPN losses
        gt_rpn_reg, gt_rpn_label = self.anchor_target_layer(
            converter.to_numpy(bbox),
            anchor,
        )
        gt_rpn_label = converter.to_tensor(gt_rpn_label).long()
        gt_rpn_reg = converter.to_tensor(gt_roi_reg)
        rpn_reg_loss = # TODO

def _smooth_l1_loss(pred_t, gt_t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (pred_t - gt_t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1.0 / sigma2)).float()
    y = (flag * (sigma2 / 2.0) * (diff ** 2) + (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()

def _fast_rcnn_reg_loss(pred_reg, gt_reg, gt_label, sigma):
    in_weight = t.zeros(gt_reg.shape).cuda()
    # make those rows of positive rois 1 and others 0
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1

    reg_loss = _smooth_l1_loss(pred_reg, gt_reg, in_weight.detach(), sigma)

    reg_loss /= ((gt_label >= 0).sum().float())
    return reg_loss
