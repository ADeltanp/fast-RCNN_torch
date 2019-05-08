import torch as t
import torch.nn as nn
from models.VGG16 import VGG16
from models.RPN import RPN
from models.RCNN import RCNN
from models.utils.ProposalTargetLayer import ProposalTargetLayer
from models.utils.AnchorTargetLayer import AnchorTargetLayer


class Faster_RCNN(nn.Module):
    def __init__(self, img_size, n_class):
        self.img_size = img_size
        self.n_class = n_class

        self.extractor = VGG16(pretrained=True)
        self.RPN = RPN('VGG16', img_size)
        self.RCNN = RCNN(n_class, init_mean=0, init_std=0.01)

        super(Faster_RCNN, self).__init__()

    def forward(self, img, img_size, img_scale, phase):
        feat = self.extractor(img)
        rpn_cls, rpn_reg, roi_list, roi_id, anchors = self.RPN(feat, img_size, img_scale)
        cls, reg = self.RCNN(feat, roi_list, roi_id)
        return cls, reg, roi_list, roi_id
