import os
import torch as t
import torch.nn as nn
import utils.converter as converter
from torch.nn import functional as F
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

        # RPN returns (B, f_h * f_w * n_a, 2), (B, f_h * f_w * n_a, 4),
        # (N_pos_nms, 4), (N_p_n, 4), (f_h * f_w * n_a, 4)
        rpn_cls, rpn_reg, rois, roi_id, anchor = self.faster_rcnn.RPN(feat, img_size, scale)

        bbox = bbox[0]
        label = label[0]
        rpn_cls = rpn_cls[0]
        rpn_reg = rpn_reg[0]
        roi = rois

        # proposal target layer returns (n_pos + n_neg, 4 \and\ 4 \and\ (NA)), xp.ndarray
        # label here is actual target label(gt label), not denote of validity
        sample_roi, gt_roi_reg, gt_roi_label = self.proposal_target_layer(
            roi,
            converter.to_numpy(bbox),
            converter.to_numpy(label),
            self.reg_normalize_mean,
            self.reg_normalize_std
        )
        sample_roi_idx = t.zeros(len(sample_roi))  # all samples are batch 0

        # RCNN returns (B * roi_per_image, n_class \and\ n_class * 4), torch.Tensor,
        # where B is 1 as only supports batch size of 1
        roi_cls, roi_reg = self.faster_rcnn.RCNN(feat, sample_roi, sample_roi_idx)

        # ------------------ RPN losses
        # label identifies whether the rpn output is valid or not
        # label: invalid -> -1; negative(bg) -> 0; positive(fg) -> 1
        # anchor target layer returns (N, 4), (N, ), xp.ndarray
        # where gt_rpn_reg is the id of gt assigned to that proposal
        gt_rpn_reg, gt_rpn_label = self.anchor_target_layer(
            converter.to_numpy(bbox),
            anchor,
        )
        gt_rpn_label = converter.to_tensor(gt_rpn_label).long()  # already on cuda
        gt_rpn_reg = converter.to_tensor(gt_roi_reg)  # already on cuda
        rpn_reg_loss = _fast_rcnn_reg_loss(
            rpn_reg,
            gt_rpn_reg,
            gt_rpn_label.data,
            self.rpn_sigma
        )
        rpn_cls_loss = F.cross_entropy(rpn_cls, gt_rpn_label, ignore_index=-1)
        gt_rpn_valid_label = gt_rpn_label[gt_rpn_label > -1]
        rpn_valid_cls = converter.to_numpy(rpn_cls)[converter.to_numpy(gt_rpn_label) > -1]
        self.rpn_cm.add(converter.to_tensor(rpn_valid_cls, False), gt_rpn_valid_label)

        # ------------------ roi losses (RCNN losses)
        n_sample = roi_cls.shape[0]
        roi_reg = roi_reg.view(n_sample, -1, 4)  # (roi_per_image, n_class, 4)
        sample_roi_reg = roi_reg[t.arange(0, n_sample).long().cuda(),
                                 converter.to_tensor(gt_roi_label).long()]


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
