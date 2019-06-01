import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cupy as cp
from models.utils.ProposalLayer import ProposalLayer
from models.utils.boundingbox import generate_anchor_base, all_anchors


class RPN(nn.Module):
    def __init__(self, extractor,
                 anchor_ratio=[0.5, 1, 2], anchor_scale=[8, 16, 32],
                 init_mean=0, init_std=0.01, cp_enable=False):
        super().__init__()
        self.extractor = extractor
        self.cp_enable = cp_enable
        self.anchor_base = generate_anchor_base(scale=anchor_scale,
                                                ratio=anchor_ratio,
                                                cp_enable=self.cp_enable)
        if extractor is "VGG16":
            self.feat_receptive_len = 16

        self.share = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU()
        )
        self.cls = nn.Conv2d(512, 18, 1, padding=0)
        self.reg = nn.Conv2d(512, 36, 1, padding=0)
        self.ProposalLayer = ProposalLayer(self.extractor)
        self._initialize_params(init_mean, init_std)

    def forward(self, feat, img_size, img_scale=1.0, phase='train'):
        '''
        :param feat: (torch.Tensor) feature map output by extractor (B, C, f_h, f_w)
        :param img_size: (tuple of ints) original image size (h, w),
                         used in proposal layer to clip down rois
        :param img_scale: (float) image scaling factor during data processing
        :param phase: (string) either 'train' or 'test',
                      determine whether anchors should be within image
        :return:reg (torch.Tensor) regression output by rpn
                cls (torch.Tensor) classification output by rpn
                roi_list (numpy ndarray) list of rois output by rpn
                roi_id (numpy ndarray) list of batch id of corresponding roi
                anchors (xp ndarray) all anchors on the feature map
        '''
        # cupy compatible TODO Compatibility Not Tested
        if self.cp_enable:
            xp = cp
        else:
            xp = np
        bat, _, h, w = feat.shape
        anchors, valid_idx = all_anchors(self.anchor_base, img_size,
                                         self.feat_receptive_len, h, w)
        num_anchors = self.anchor_base.shape[0]

        shared = self.share(feat)
        cls = self.cls(shared)
        cls = cls.permute(0, 2, 3, 1).contiguous().view(-1, 2)
        cls = cls[valid_idx].contiguous().view(bat, -1, 2)  # shape (B, all_anchors, 2)
        cls = F.softmax(cls, dim=-1)
        cls = cls.view(bat, -1, 2)  # (B, v_h * v_w * n_a, 2)

        reg = self.reg(shared)
        reg = reg.permute(0, 2, 3, 1).contiguous().view(-1, 4)
        reg = reg[valid_idx].contiguous().view(bat, -1, 4)  # shape (B, h * w * n_a, 4)

        # roi is not (x, y, h, w) but (x, y, x, y)
        roi_list = list()
        roi_id = list()
        for i in range(bat):
            roi = self.ProposalLayer(
                cls[i].cpu().data.numpy(),
                reg[i].cpu().data.numpy(),
                anchors,
                img_size,
                img_scale,
                phase
            )
            batch_id = i * np.ones((len(roi),), dtype=xp.int32)
            roi_list.append(roi)
            roi_id.append(batch_id)

        roi_list = xp.concatenate(roi_list, axis=0)
        roi_id = np.concatenate(roi_id, axis=0)

        if xp is cp:
            roi_list = cp.asnumpy(roi_list)

        # (B, h * w * n_a, 2 \and\ 4), (B * N_pos_nms, 4 \and\ (NA)), (h * w * n_a, 4)
        # only support batch size 1 b/c feat.shape must be constant over one forward
        return cls, reg, roi_list, roi_id, anchors

    def _initialize_params(self, mean, std):
        self.share[0].weight.data.normal_(mean, std)
        self.cls.weight.data.normal_(mean, std)
        self.reg.weight.data.normal_(mean, std)

