import numpy as np
import cupy as cp
import torch as t


def generate_anchor_base(base_size=16,
                         scale=[8, 16, 32],
                         ratio=[0.5, 1.0, 2.0],
                         cp_enable=False):
    # cupy compatible TODO Compatibility Not Tested
    if cp_enable:
        xp = cp
    else:
        xp = np

    scale = xp.array(scale, dtype=xp.float32)[:, np.newaxis]  # shape (3, 1)
    ratio = xp.array(ratio, dtype=xp.float32)[:, np.newaxis]  # shape (3, 1)
    anchor = xp.zeros((len(scale) * len(ratio), 4), dtype=xp.float32)
    ctr_x = base_size / 2.0
    ctr_y = base_size / 2.0

    anchor[:, 0].fill(ctr_x)
    anchor[:, 1].fill(ctr_y)
    anchor[:, 2].fill(ctr_x)
    anchor[:, 3].fill(ctr_y)

    # matmul produces [3, 3] matrix, reshape into (9, 1) array
    h = base_size * xp.matmul(scale, xp.sqrt(ratio).transpose()).reshape(-1, 1)
    w = base_size * xp.matmul(scale, xp.sqrt(1 / ratio).transpose()).reshape(-1, 1)

    anchor[:, 0::4] -= w / 2.0  # x_min of idx th ratio
    anchor[:, 1::4] -= h / 2.0  # y_min of idx th ratio
    anchor[:, 2::4] += w / 2.0  # x_max of idx th ratio
    anchor[:, 3::4] += h / 2.0  # y_max of idx th ratio

    return anchor


def all_anchors(anchor_base, img_size, feat_receptive_len, h, w):
    # cupy compatible TODO Compatibility Not Tested
    assert anchor_base.shape[1] == 4

    xp = cp.get_array_module(anchor_base)
    x = xp.arange(0, w * feat_receptive_len, feat_receptive_len)
    y = xp.arange(0, h * feat_receptive_len, feat_receptive_len)
    x, y = xp.meshgrid(x, y)

    ctr_shift = xp.stack((x.ravel(), y.ravel(), x.ravel(), y.ravel()), axis=1)
    num_base = anchor_base.shape[0]
    num_shift = ctr_shift.shape[0]

    # shape (1, 9, 4) + (num_shift, 1, 4) = (num_shift, 9, 4), num_base = 9 by default
    anchors = anchor_base[xp.newaxis, :] + ctr_shift[xp.newaxis, :].transpose((1, 0, 2))
    anchors = anchors.reshape((num_base * num_shift, 4)).astype(xp.float32)
    valid_index = np.arange(anchors.shape[0])
    # if phase is 'train':
    #     assert len(img_size) == 2
    #     valid_index = xp.where(
    #         (anchors[:, 0] >= 0) &
    #         (anchors[:, 1] >= 0) &
    #         (anchors[:, 2] <= img_size[1]) &
    #         (anchors[:, 3] <= img_size[0])
    #     )[0]
    #     anchors = anchors[valid_index, :]

    return anchors, valid_index


def t_encoded2bbox(anchor, t_encoded):
    '''
    torch, cupy compatible TODO Compatibility Not Tested
    decode t_x, t_y, t_h, t_w to bounding box, x_min, x_max, y_min, y_max

    anchor[i, :] = x_a_min, y_a_min, x_a_max, y_a_max, offset of i th anchor
    t_encoded[i, :] = t_x, t_y, t_h, t_w, output of reg layer or encoded gt

    :param anchor: all anchor boxes over the image of shape (N, 4)
    :param t_encoded: regression output by RPN of shape (N, 4)
    :return: the predicted bounding box of shape (N, 4)

    N is the number of anchors in total
    w_a, h_a, x_a, y_a, t_x, t_y, t_w, t_h are the notations used in the original paper
    tgt_* stands for * or *_star in the original paper resp.
    x_a and y_a, tgt_x and tgt_y are coordinates of the according center

    the function following is the reverse of this one
    '''
    assert anchor.shape[0] == t_encoded.shape[0]
    assert anchor.shape[1] == 4
    assert t_encoded.shape[1] == 4
    assert isinstance(anchor, type(t_encoded))
    assert anchor.dtype == t_encoded.dtype

    if isinstance(anchor, t.Tensor):
        txp = t
    elif type(anchor) in (cp.ndarray, np.ndarray):
        txp = cp.get_array_module(anchor)
    else:
        raise TypeError('only accept torch.Tensor, cp.ndarray or np.ndarray.')

    # if anchor[0] == 0:
    #    return txp.zeros((0, 4), dtype=t_encoded.dtype)

    w_a = anchor[:, 2] - anchor[:, 0]  # shape (N, )
    h_a = anchor[:, 3] - anchor[:, 1]  # shape (N, )
    x_a = anchor[:, 0] + 0.5 * w_a     # shape (N, )
    y_a = anchor[:, 1] + 0.5 * h_a     # shape (N, )

    t_x = t_encoded[:, 0]  # shape (N, )
    t_y = t_encoded[:, 1]  # shape (N, )
    t_h = t_encoded[:, 2]  # shape (N, )
    t_w = t_encoded[:, 3]  # shape (N, )

    tgt_x = t_x * w_a + x_a    # shape (N, )
    tgt_y = t_y * h_a + y_a    # shape (N, )
    tgt_h = txp.exp(t_h) * h_a  # shape (N, )
    tgt_w = txp.exp(t_w) * w_a  # shape (N, )

    # stacked (4, N) -> transpose(N, 4)
    target_bbox = txp.stack((tgt_x, tgt_y, tgt_h, tgt_w)).transpose(1, 0)
    return target_bbox


def bbox2t_encoded(anchor, target_bbox):
    '''
    torch, cupy compatible TODO Compatibility Not Tested
    :param anchor: anchors over image of shape (N, 4)
    :param target_bbox: bbox over image of shape (N, 4)
    :return: encoded offsets (denoted by t in the paper) of shape (N, 4)

    anchor and target_bbox must both be either t.Tensor, np.ndarray or cp.ndarray
    '''
    assert anchor.shape[0] == target_bbox.shape[0]
    assert anchor.shape[1] == 4
    assert target_bbox.shape[1] == 4
    assert isinstance(anchor, type(target_bbox))
    assert anchor.dtype == target_bbox.dtype

    if isinstance(anchor, t.Tensor):
        txp = t
    elif type(anchor) in (cp.ndarray, np.ndarray):
        txp = cp.get_array_module(anchor)
    else:
        raise TypeError('only accept torch.Tensor, cp.ndarray or np.ndarray.')

    w_a = anchor[:, 2] - anchor[:, 0]  # (N, )
    h_a = anchor[:, 3] - anchor[:, 1]
    x_a = anchor[:, 0] + 0.5 * w_a
    y_a = anchor[:, 1] + 0.5 * h_a

    tgt_w = target_bbox[:, 2] - target_bbox[:, 0]  # (N, )
    tgt_h = target_bbox[:, 3] - target_bbox[:, 1]
    tgt_x = target_bbox[:, 0] + 0.5 * tgt_w
    tgt_y = target_bbox[:, 1] + 0.5 * tgt_h

    # ensure not divided by zero
    eps = txp.finfo(w_a.dtype).eps
    if txp is t:
        h_a = txp.max(h_a, eps)
        w_a = txp.max(w_a, eps)
    else:
        h_a = txp.maximum(h_a, eps)
        w_a = txp.maximum(w_a, eps)

    t_x = (tgt_x - x_a) / w_a  # (N, )
    t_y = (tgt_y - y_a) / h_a
    t_h = txp.log(tgt_h / h_a)
    t_w = txp.log(tgt_w / w_a)

    t_encoded = txp.stack((t_x, t_y, t_h, t_w)).transpose(1, 0)
    return t_encoded


def compute_iou_xp(bbox_a, bbox_b):
    # cupy compatible TODO Compatibility Not Tested
    # (N, 4), (K, 4)
    # bbox of (x, y, x, y), not (x, y, h, w)
    assert bbox_a.shape[1] == bbox_b.shape[1]
    assert type(bbox_a) == type(bbox_b)

    xp = cp.get_array_module(bbox_a)

    # newaxis here is to utilize broadcasting to compare between each element in a and b
    top_left     = xp.maximum(bbox_a[:, xp.newaxis, :2], bbox_b[:, :2])  # shape (N, K, 2)
    bottom_right = xp.minimum(bbox_a[:, xp.newaxis, 2:], bbox_b[:, 2:])  # shape (N, K, 2)

    # shapes: (N, K), (N, ), (K, ), returns (N, K)
    iou_area = xp.prod(bottom_right - top_left, axis=2) * (top_left < bottom_right).all(axis=2)
    area_a = xp.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = xp.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return iou_area / (area_a[:, xp.newaxis] + area_b - iou_area)
