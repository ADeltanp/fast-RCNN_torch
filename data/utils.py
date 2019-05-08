import numpy as np
from PIL import Image


def read_image(path, dtype=np.float32):
    img = Image.open(path)
    np_img = np.asarray(img, dtype=dtype)
    return np_img.transpose((2, 0, 1))  # (h, w, c) -> (c, h, w)

