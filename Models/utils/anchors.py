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
            anchor[idx, 0] = ctr_x - w / 2.0  # x_min of idx th ratio
            anchor[idx, 1] = ctr_y - h / 2.0  # y_min of idx th ratio
            anchor[idx, 2] = ctr_x + w / 2.0  # x_max of idx th ratio
            anchor[idx, 3] = ctr_y + h / 2.0  # y_max of idx th ratio
    return anchor


def all_anchors(anchor_base, feat_receptive_len, h, w, phase='test'):
    x = np.arange(0, w * feat_receptive_len, feat_receptive_len)
    y = np.arange(0, h * feat_receptive_len, feat_receptive_len)
    x, y = np.meshgrid(x, y)

    ctr_shift = np.stack((x.ravel(), y.ravel(), x.ravel(), y.ravel()), axis=1)
    num_base = anchor_base.shape[0]
    num_shift = ctr_shift.shape[0]
    anchors = anchor_base.reshape((1, num_base, 4)) + ctr_shift.reshape((1, num_shift, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((num_base * num_shift, 4)).astype(np.float32)
    if phase is 'train':
        valid_index = np.where(
            (anchors[:, 0] >= 0) &
            (anchors[:, 1] >= 0) &
            (anchors[:, 2] <= 800) &
            (anchors[:, 3] <= 800)
        )
        anchors = anchors[valid_index, :]
    return anchors


def decode_to_bbox(anchor, reg):
    '''
    decode t_x, t_y, t_h, t_w to bounding box, x_min, x_max, y_min, y_max

    anchor[i, :] = x_a_min, y_a_min, x_a_max, y_a_max, offsets of i th anchor
    reg[i, :] = t_x, t_y, t_w, t_h, output of reg layer

    :param anchor: all anchor boxes over the image of shape (N, 4)
    :param reg: regression output by RPN of shape (N, 4)
    :return: the predicted bounding box of shape (N, 4)

    N is the number of anchors in total
    w_a, h_a, x_a, y_a, t_x, t_y, t_w, t_h are the notations used in the original paper
    pred_x, pred_y, pred_w, pred_h stands for x, y, w, h in the original paper respectively
    '''
    if anchor[0] == 0:
        return np.zeros((0, 4), dtype=reg.dtype)

    w_a = anchor[:, 2] - anchor[:, 0]  # shape (N, )
    h_a = anchor[:, 3] - anchor[:, 1]  # shape (N, )
    x_a = anchor[:, 0] + 0.5 * w_a     # shape (N, )
    y_a = anchor[:, 1] + 0.5 * h_a     # shape (N, )

    t_x = reg[, 0::4]  # shape (N, 1)
    t_y = reg[, 1::4]  # shape (N, 1)
    t_w = reg[, 2::4]  # shape (N, 1)
    t_h = reg[, 3::4]  # shape (N, 1)

    pred_x = t_x * w_a[:, np.newaxis] + x_a[:, np.newaxis]  # shape (N, 1), x = t_x * w_a + x_a
    pred_y = t_y * h_a[:, np.newaxis] + y_a[:, np.newaxis]  # shape (N, 1), y = t_y * h_a + y_a
    pred_w = np.exp(t_w) * w_a[:, np.newaxis]  # shape (N, 1), w = exp(t_w) * w_a
    pred_h = np.exp(t_h) * h_a[:, np.newaxis]  # shape (N, 1), h = exp(t_h) * h_a

    pred_bbox = np.zeros(reg.shape, dtype=reg.dtype)
    pred_bbox[:, 0::4] = pred_x - pred_w * 0.5  # min_x of bbox
    pred_bbox[:, 1::4] = pred_y - pred_h * 0.5  # min_y of bbox
    pred_bbox[:, 2::4] = pred_x + pred_w * 0.5  # max_x of bbox
    pred_bbox[:, 3::4] = pred_y + pred_h * 0.5  # max_y of bbox

    return pred_bbox

def encode_from_bbox(anchor, pred):
    
