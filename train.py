import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.dataset import Dataset, TestDataset, un_normalize
from Faster_RCNN import Faster_RCNN
from utils.config import config
from utils import converter
from train_helper import TrainHelper

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))


def eval(testset, faster_rcnn, test_num=10000):
    pred_bbox, pred_label, pred_cls = list(), list(), list()
    gt_bbox, gt_label = list(), list()

    for i, (img, size, gt_bbox, gt_label) in tqdm(enumerate(testset)):
        size = [size[0][0].item(), size[1][0].item()]
        pred_bbox_, pred_label_, pred_cls_ = faster_rcnn.predict(img, [size])


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
    faster_rcnn = Faster_RCNN().cuda()
    print('constructed Faster-RCNN model')
    train_helper = TrainHelper(faster_rcnn).cuda()
    if config.load_path:
        train_helper.load(config.load_path)
        print('load pretrained model from %s' % config.load_path)
    best_map = 0
    # --------------- ---- --- ---- --- ---- lr_ = config.lr
    for epoch in range(config.epoch):
        train_helper.reset_meters()
        for i, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = converter.to_scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            train_helper.train_step(img, bbox, label, scale)

            # TODO Add Support of plot_every

        lr_ = train_helper.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, loss:{}'.format(str(lr_), str(train_helper.get_meter_data()))

