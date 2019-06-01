class Config:
    extractor = 'VGG16'
    voc_data_dir = 'D:\\RtCV\\Data\\VOCdevkit\\VOC2012'
    min_size = 600
    max_size = 1000
    num_workers = 8
    test_num_workers = 8

    rpn_sigma = 3.0
    roi_sigma = 1.0

    weight_decay = 3.0
    lr_decay = 0.1
    lr = 1e-3

    data = 'voc'
    n_class = 20

    epoch = 14

    use_adam = False

    test_num = 10000

    load_path = None

    # using inputs to change default config
    def _parse(self, kwargs):
        self_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in self_dict:
                raise ValueError('No option named \"--%s\"' % k)
            setattr(self, k, v)

    def _state_dict(self):
        return {k : getattr(self, k) for k, _ in Config.__dict__.items()
                if not k.startswith("_")}


config = Config()
