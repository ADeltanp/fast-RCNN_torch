import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Models.utils.ProposalLayer import ProposalLayer
from Models.utils.anchors import generate_anchor_base, all_anchors


class RPN(nn.Module):
    def __init__(self, RPN, extractor, img_size, img_scale):
        super(RPN, self).__init__()
        self.extractor = extractor
        self.img_size = img_size
        self.img_scale = img_scale
        self.anchor_base = generate_anchor_base()
        if extractor is "VGG16":
            self.feat_receptive_len = 16

        self.share = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU()
        )
        self.cls = nn.Conv2d(512, 18, 1, padding=0)
        self.reg = nn.Conv2d(512, 36, 1, padding=0)
        self.ProposalLayer = ProposalLayer(self.extractor)

    def _reshape(self, x, size):
        # B * C * H * W --> B * size * (H * C / size) * W
        shape = x.size()
        x.view(
            shape[0],
            int(float(shape[1]) * float(shape[3]) / float(size)),
            shape[2],
            int(size)
        )
        return x

    def forward(self, x, img_size, img_scale=1.0, phase='test'):
        bat, _, h, w = x.shape
        anchors = all_anchors(self.anchor_base, self.feat_receptive_len, h, w, phase=phase)
        num_anchors = anchors[0]

        shared = self.share(x)
        cls = self.cls(shared)
        cls = cls.permute(0, 2, 3, 1).contiguous().view(bat, h, w, num_anchors, 2)  # shape (B, h, w, n_a, 2)
        cls = F.softmax(cls, dim=4)
        cls = cls.view(bat, -1, 2)

        reg = self.reg(shared)
        reg = reg.permute(0, 2, 3, 1).contiguous().view(bat, -1, 4)  # shape (B, h * w * n_a, 4)

        roi_list = list()
        roi_id = list()
        for i in range(bat):
            roi = self.ProposalLayer(
                cls[i].cpu().data.numpy(),
                reg[i].cpu().data.numpy(),
                anchors,
                img_size,
                img_scale
            )
            batch_id = i * np.ones((len(roi),), dtype=np.int32)
            roi_list.append(roi)
            roi_id.append(batch_id)

        np.concatenate(roi_list, axis=0)
        roi_id = np.concatenate(roi_id, axis=0)

        return reg, cls, roi_list, roi_id, anchors

