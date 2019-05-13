import os, time
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
        self.rcnn_cm = ConfusionMeter(config.n_class + 1)
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
        assert bbox.shape[0] == 1, 'currently only support batch size of 1.'

        _, _, h, w = img.shape
        img_size = (h, w)

        feat = self.faster_rcnn.extractor(img)

        # RPN returns (B, f_h * f_w * n_a, 2), (B, f_h * f_w * n_a, 4),
        # (N_pos_nms, 4), (N_p_n, 4), (f_h * f_w * n_a, 4), N_pos_nms = roi_per_img
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
        rcnn_cls, rcnn_reg = self.faster_rcnn.RCNN(feat, sample_roi, sample_roi_idx)

        # ------------------ RPN losses
        #
        # label identifies whether the rpn output is valid or not
        # label: invalid -> -1; negative(bg) -> 0; positive(fg) -> 1
        #
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
        gt_rpn_valid_label = gt_rpn_label[gt_rpn_label > -1].cpu()
        rpn_valid_cls = rpn_cls[gt_rpn_label > -1].cpu()
        self.rpn_cm.add(converter.to_tensor(rpn_valid_cls, False),
                        gt_rpn_valid_label.data.long())

        # ------------------ roi losses (RCNN losses)
        n_sample = rcnn_cls.shape[0]
        rcnn_reg = rcnn_reg.view(n_sample, -1, 4)  # (roi_per_image, n_class, 4)

        # as RCNN is fed with output of proposal target layer(PTL),
        # use gt_roi_label, output of PTL, to choose the bbox assoc. with gt bbox
        # (n_sample, 4)
        sample_rcnn_reg = rcnn_reg[t.arange(0, n_sample).long().cuda(),
                                   converter.to_tensor(gt_roi_label).long()]
        # it doesn't matter to give it another name as they share the same memory
        gt_rcnn_label = converter.to_tensor(gt_roi_label).long()
        gt_rcnn_reg = converter.to_tensor(gt_roi_reg)

        rcnn_reg_loss = _fast_rcnn_reg_loss(
            sample_rcnn_reg.continuous(),
            gt_rcnn_reg,
            gt_rcnn_label.data,
            self.roi_sigma
        )
        rcnn_cls_loss = nn.CrossEntropyLoss()(rcnn_cls, gt_rcnn_label)
        self.rcnn_cm.add(converter.to_tensor(rcnn_cls, False),
                         gt_rcnn_label.data.long())

        losses = [rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss, rcnn_reg_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses)

    def train_step(self, img, bbox, label, scale):
        self.optimizer.zero_grad()
        losses = self.forward(img, bbox, label, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        self.update_meters(losses)
        return losses

    def save(self, save_optimizer=False, save_path=None, **kwargs):
        save_dict = dict()
        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['config'] = config._state_dict()
        save_dict['miscellaneous'] = kwargs
        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            time_ = time.strftime('%y%m%d%H%M')
            save_path = 'checkpoints/faster_rcnn_%s' % time_
            for k, v in kwargs.items():
                save_path += '_%s' % v

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        t.save(save_dict, save_path)
        return save_path

    def load(self, path, load_optimizer=True, parse_config=False):
        state_dict = t.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:
            self.faster_rcnn.load_state_dict(state_dict)
        if parse_config:
            config._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

    def update_meters(self, losses):
        loss_dict = {k: converter.to_tensor(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_dict[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.rcnn_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}


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
