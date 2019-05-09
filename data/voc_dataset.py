import os
import xml.etree.ElementTree as ET
import numpy as np
from .utils import read_image


VOC_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')


class VOC_detection:
    def __init__(self, data_dir, branch="trainval"):
        id_file = os.path.join(data_dir, 'ImageSets/Main/{0}.txt'.format(branch))
        self.data_dir = data_dir
        self.ids = [img_id.strip() for img_id in open(id_file)]
        self.label_names = VOC_LABEL_NAMES

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        annotation_file = os.path.join(self.data_dir, 'Annotations', id_ + '.xml')
        root = ET.parse(annotation_file)
        bbox = list()
        label = list()

        for obj in root.findall("object"):
            if int(obj.find("difficult").text) == 1:
                continue

            name = obj.find("name").text.lower().strip()
            label.append(self.label_names.index(name))

            bbox_obj = obj.find("bndbox")
            bbox.append([int(bbox_obj.find(coordinate).text) - 1
                        for coordinate in ('xmin', 'ymin', 'xmax', 'ymax')])

        label = np.stack(label).astype(np.int32)
        bbox = np.stack(bbox).astype(np.float32)
        image_file = os.path.join(self.data_dir, 'JPEGImages', id_ + '.jpg')
        img = read_image(image_file)

        return img, bbox, label

