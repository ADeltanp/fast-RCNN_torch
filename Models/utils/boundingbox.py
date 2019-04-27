import numpy as np
import cupy as cp


def generate_anchor_base(base_size=16,
                         scale=[8, 16, 32],
                         ratio=[0.5, 1.0, 2.0],
                         cp_enable=False):
    # cupy compatible TODO Compatibility Not Tested
    if cp_enable:
        xp = cp
    else:
        xp = np

    anchor = xp.zeros((len(scale) * len(ratio), 4), dtype=xp.float32)
    ctr_x = base_size / 2.0
    ctr_y = base_size / 2.0
    for i in range(len(ratio)):
        for j in range(len(scale)):
            h = base_size * scale[j] * xp.sqrt(ratio[i])
            w = base_size * scale[j] * xp.sqrt(1.0 / ratio[i])
            idx = i * len(scale) + j
            anchor[idx, 0] = ctr_x - w / 2.0  # x_min of idx th ratio
            anchor[idx, 1] = ctr_y - h / 2.0  # y_min of idx th ratio
            anchor[idx, 2] = ctr_x + w / 2.0  # x_max of idx th ratio
            anchor[idx, 3] = ctr_y + h / 2.0  # y_max of idx th ratio
    return anchor


def all_anchors(anchor_base, feat_receptive_len, h, w, phase='test'):
    # cupy compatible TODO Compatibility Not Tested
    xp = cp.get_array_module(anchor_base)
    x = xp.arange(0, w * feat_receptive_len, feat_receptive_len)
    y = xp.arange(0, h * feat_receptive_len, feat_receptive_len)
    x, y = xp.meshgrid(x, y)

    ctr_shift = xp.stack((x.ravel(), y.ravel(), x.ravel(), y.ravel()), axis=1)
    num_base = anchor_base.shape[0]
    num_shift = ctr_shift.shape[0]
    anchors = anchor_base.reshape((1, num_base, 4)) + ctr_shift.reshape((1, num_shift, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((num_base * num_shift, 4)).astype(xp.float32)
    if phase is 'train':
        valid_index = xp.where(
            (anchors[:, 0] >= 0) &
            (anchors[:, 1] >= 0) &
            (anchors[:, 2] <= 800) &
            (anchors[:, 3] <= 800)
        )
        anchors = anchors[valid_index, :]
    return anchors


def t_encoded2bbox(anchor, t_encoded):
    '''
    cupy compatible TODO Compatibility Not Tested
    decode t_x, t_y, t_h, t_w to bounding box, x_min, x_max, y_min, y_max

    anchor[i, :] = x_a_min, y_a_min, x_a_max, y_a_max, offset of i th anchor
    t_encoded[i, :] = t_x, t_y, t_w, t_h, output of reg layer or encoded gt

    :param anchor: all anchor boxes over the image of shape (N, 4)
    :param t_encoded: regression output by RPN of shape (N, 4)
    :return: the predicted bounding box of shape (N, 4)

    N is the number of anchors in total
    w_a, h_a, x_a, y_a, t_x, t_y, t_w, t_h are the notations used in the original paper
    tgt_x, tgt_y, tgt_w, tgt_h stands for x, y, w, h in the original paper respectively
    x_a and y_a, tgt_x and tgt_y are coordinates of the according center

    the function following is the reverse of this one
    '''
    xp = cp.get_array_module(anchor)

    if anchor[0] == 0:
        return xp.zeros((0, 4), dtype=t_encoded.dtype)

    w_a = anchor[:, 2] - anchor[:, 0]  # shape (N, )
    h_a = anchor[:, 3] - anchor[:, 1]  # shape (N, )
    x_a = anchor[:, 0] + 0.5 * w_a     # shape (N, )
    y_a = anchor[:, 1] + 0.5 * h_a     # shape (N, )

    t_x = t_encoded[:, 0]  # shape (N, )
    t_y = t_encoded[:, 1]  # shape (N, )
    t_w = t_encoded[:, 2]  # shape (N, )
    t_h = t_encoded[:, 3]  # shape (N, )

    tgt_x = t_x * w_a + x_a    # shape (N, )
    tgt_y = t_y * h_a + y_a    # shape (N, )
    tgt_w = xp.exp(t_w) * w_a  # shape (N, )
    tgt_h = xp.exp(t_h) * h_a  # shape (N, )

    target_bbox = xp.vstack((tgt_x, tgt_y, tgt_w, tgt_h)).transpose()
    return target_bbox


def bbox2t_encoded(anchor, target_bbox):
    '''
    cupy compatible TODO Compatibility Not Tested
    :param anchor: anchors over image of shape (N, 4)
    :param target_bbox: bbox over image of shape (N, 4)
    :return: encoded offsets (denoted by t in the paper) of shape (N, 4)

    anchor and target_bbox must both be either numpy or cupy objects
    '''
    xp = cp.get_array_module(anchor)

    w_a = anchor[:, 2] - anchor[:, 0]
    h_a = anchor[:, 3] - anchor[:, 1]
    x_a = anchor[:, 0] + 0.5 * w_a
    y_a = anchor[:, 1] + 0.5 * h_a

    tgt_w = target_bbox[:, 2] - target_bbox[:, 0]
    tgt_h = target_bbox[:, 3] - target_bbox[:, 1]
    tgt_x = target_bbox[:, 0] + 0.5 * tgt_w
    tgt_y = target_bbox[:, 1] + 0.5 * tgt_h

    # ensure not divided by zero
    eps = xp.finfo(w_a.dtype).eps
    h_a = xp.maximum(h_a, eps)
    w_a = xp.maximum(w_a, eps)

    t_x = (tgt_x - x_a) / w_a
    t_y = (tgt_y - y_a) / h_a
    t_w = xp.log(tgt_w / w_a)
    t_h = xp.log(tgt_h / h_a)

    t_encoded = xp.vstack((t_x, t_y, t_w, t_h)).transpose()
    return t_encoded

def iou(bbox_a, bbox_b):
    # cupy compatible TODO Compatibility Not Tested
    xp = cp.get_array_module(bbox_a)

    # newaxis here is to utilize broadcasting to compare between each element in a and b
    top_left     = xp.maximum(bbox_a[:, xp.newaxis, :2], bbox_b[:, :2])  # shape (N, K, 2)
    bottom_right = xp.minimum(bbox_a[:, xp.newaxis, 2:], bbox_b[:, 2:])  # shape (N, K, 2)

    # shapes: (N, K), (N, ), (K, ), returns (N, K)
    iou_area = xp.prod(bottom_right - top_left, axis=2) * (top_left < bottom_right).all(axis=2)
    area_a = xp.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = xp.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return iou_area / (area_a[:, xp.newaxis] + area_b - iou_area)
