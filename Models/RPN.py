import torch.nn as nn
import torch.nn.functional as F
from utils.ProposalLayer import ProposalLayer
from utils.anchors import generate_anchor_base, all_anchors


class RPN(nn.Module):
    def __init__(self, RPN, extractor, img_size, img_scale):
        super(RPN, self).__init__()
        self.extractor = extractor
        self.img_size = img_size
        self.img_scale = img_scale
        self.anchor = generate_anchor_base()
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

    def forward(self, x):
        bat, _, h, w = x.shape
        anchors = all_anchors(self.anchor, self.feat_receptive_len, h, w)

        shared = self.share(x)
        cls = self.cls(shared)
        cls = self._reshape(cls, 2)  # shape (B, 2, (H * 9), W)
        cls = F.softmax(cls, 1)
        cls = self._reshape(cls, 18)  # shape (B, 18, H, W)
        reg = self.reg(shared)
        out = self.ProposalLayer(cls, reg, self.anchor, self.img_size, self.img_scale)
        # TODO Proposal Layer & Proposal Target Layer

