import numpy as np
from PIL import Image


def read_image(path, dtype=np.float32):
    img = Image.open(path)
    np_img = np.asarray(img, dtype=dtype)
    return np_img.transpose((2, 0, 1))  # (h, w, c) -> (c, h, w)


def resize_bbox(bbox, in_size, out_size):
    bbox = bbox.copy()
    x_scale = float(out_size[1]) / in_size[1]
    y_scale = float(out_size[0]) / in_size[0]
    # for one bbox, it's (x_min, y_mix, x_max, y_max)
    scale = np.array([x_scale, y_scale, x_scale, y_scale])
    bbox = bbox * scale
    return bbox.astype(np.float32)
