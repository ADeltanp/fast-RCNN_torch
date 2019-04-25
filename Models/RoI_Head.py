import torch as t
import torch.nn as nn


class RoI_Head(nn.Module):
    def __init__(self, n_class, roi_size):
        super(RoI_Head, self).__init__()
