import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.dataset import Dataset, TestDataset, un_normalize
from Faster_RCNN import Faster_RCNN
from utils.config import config

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))


def train(**kwargs):
    config._parse(kwargs)
    dataset = Dataset(config)
    print('loading data')
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=config.num_workers)
    testset = TestDataset(config)
    test_dataloader = DataLoader(testset,
                                 batch_size=1,
                                 num_workers=config.test_num_workers,
                                 shuffle=False,
                                 pin_memory=True)
    faster_rcnn = Faster_RCNN()
    print('constructed Faster-RCNN model')
    
