import numpy as np
import torch as t
from skimage import transform as skt
from torchvision import transforms as tvt
from data.utils import resize_bbox
from .voc_dataset import VOC_detection


def un_normalize(img):
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


def preprocess(img, min_size=600, max_size=1000):
    c, h, w = img.shape

    scale1 = min_size / min(h, w)
    scale2 = max_size / max(h, w)
    scale = min(scale1, scale2)

    img = img / 255.0
    img = skt.resize(img, (c, h * scale, w * scale),
                     mode='reflect', anti_aliasing=False)

    normalize = tvt.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    img = normalize(t.from_numpy(img)).numpy()

    return img


class Transform:
    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        '''
        :param in_data: (img : numpy.ndarray, (c, h, w)
                         bbox : numpy.ndarray, (N, 4)
                         label : numpy.ndarray, (N, 4)
        :return:
        '''
        img, bbox, label = in_data
        _, h, w = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, H, W = img.shape
        scale = H / h
        bbox = resize_bbox(bbox, (h, w), (H, W))

        # TODO Add More Data Augmentation Methods

        # img = np.ascontiguousarray(img)
        # bbox = np.ascontiguousarray(bbox)
        # label = np.ascontiguousarray(label)

        return img, bbox, label, scale


class Dataset:
    def __init__(self, config):
        self.config = config
        self.db = VOC_detection(config.voc_data_dir)
        self.trans = Transform(config.min_size, config.max_size)

    def __getitem__(self, idx):
        original_img, bbox, label = self.db[idx]
        img, bbox, label, scale = self.trans((original_img, bbox, label))
        return img, bbox, label, scale  # img and bbox all resized

    def __len__(self):
        return len(self.db)


class TestDataset:
    def __init__(self, config, branch='test'):
        self.config = config
        self.db = VOC_detection(config.voc_data_dir, branch=branch)

    def __getitem__(self, idx):
        original_img, bbox, label = self.db[idx]
        img = preprocess(original_img)
        return img, original_img.shape[1:], bbox, label  # bbox not resized

    def __len__(self):
        return len(self.db)
