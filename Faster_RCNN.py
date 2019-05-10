import torch as t
import torch.nn as nn
from models.VGG16 import VGG16
from models.RPN import RPN
from models.RCNN import RCNN
from utils.config import config


class Faster_RCNN(nn.Module):
    def __init__(self, n_class=20,
                 extractor_pretrained=True,
                 anchor_ratio=[0.5, 1, 2],
                 anchor_scale=[8, 16, 32],
                 rcnn_init_mean=0, rcnn_init_std=0.01,
                 reg_normalize_mean = (0.0, 0.0, 0.0, 0.0),
                 reg_normalize_std = (0.1, 0.1, 0.2, 0.2)):
        self.n_class = n_class

        self.extractor = VGG16(extractor_pretrained)
        self.RPN = RPN('VGG16', anchor_ratio, anchor_scale)
        self.RCNN = RCNN(n_class, rcnn_init_mean, rcnn_init_std)

        self.reg_normalize_mean = reg_normalize_mean
        self.reg_normalize_std = reg_normalize_std

        super(Faster_RCNN, self).__init__()

    def forward(self, img, img_size, img_scale, phase):
        feat = self.extractor(img)
        rpn_cls, rpn_reg, roi_list, roi_id, anchors = self.RPN(feat, img_size, img_scale)
        cls, reg = self.RCNN(feat, roi_list, roi_id)
        return cls, reg, roi_list, roi_id

    def get_optimizer(self):
        lr = config.lr
        params = []
        for name, param in dict(self.named_parameters()).items():
            if param.requires_grad:
                if 'bias' in name:
                    params += [{'params': [param], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [param], 'lr': lr, 'weight_decay': config.weight_decay}]

        if config.use_adam:
            self.optimizer = t.optim.Adam(params)
        else:
            self.optimizer = t.optim.SGD(params, momentum=0.9)
        return self.optimizer

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer
