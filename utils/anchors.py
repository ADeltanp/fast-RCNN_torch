import torch as t
import numpy as np


def generate_anchor_base(base_size=16, scale=[8, 16, 32],  ratio=[0.5, 1.0, 2.0]):
    anchor = np.zeros((len(scale) * len(ratio), 4), dtype=np.float32)
    ctr_x = base_size / 2.0
    ctr_y = base_size / 2.0
    for i in range(len(ratio)):
        for j in range(len(scale)):
            h = base_size * scale[j] * np.sqrt(ratio[i])
            w = base_size * scale[j] * np.sqrt(1.0 / ratio[i])
            idx = i * len(scale) + j
            anchor[idx, 0] = ctr_x - w / 2.0  # x_min of ratio idx
            anchor[idx, 1] = ctr_y - h / 2.0  # y_min of ratio idx
            anchor[idx, 2] = ctr_x + w / 2.0  # x_max of ratio idx
            anchor[idx, 3] = ctr_y + h / 2.0  # y_max of ratio idx
    return anchor


def all_anchors(anchor_base, feat_receptive_len, h, w):
    x = np.arange(0, w * feat_receptive_len, feat_receptive_len)
    y = np.arange(0, h * feat_receptive_len, feat_receptive_len)
    x, y = np.meshgrid(x, y)
    ctr_shift = np.stack((x.ravel(), y.ravel(), x.ravel(), y.ravel()), axis=1)
    num_base = anchor_base.shape[0]
    num_shift = ctr_shift.shape[0]
    anchors = anchor_base.reshape((1, num_base, 4)) + ctr_shift.reshape((1, num_shift, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((num_base * num_shift, 4)).astype(np.float32)
    return anchors
