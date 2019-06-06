import time
import visdom
import torch as t
import matplotlib
import numpy as np
from matplotlib import pyplot as ppl

matplotlib.use('Agg')
VOC_BBOX_LABEL_NAMES = (
    'fly',
    'bike',
    'bird',
    'boat',
    'pin',
    'bus',
    'c',
    'cat',
    'chair',
    'cow',
    'table',
    'dog',
    'horse',
    'moto',
    'p',
    'plant',
    'sheep',
    'sofa',
    'train',
    'tv',
)


def vis_image(img, ax=None):
    if ax is None:
        fig = ppl.figure()
        ax = fig.add_subplot(1, 1, 1)
    img = img.transpose((1, 2, 0))
    ax.imshow(img.astype(np.uint8))
    return ax


def vis_bbox(img, bbox, label=None, cls=None, ax=None):
    label_names = list(VOC_BBOX_LABEL_NAMES) + ['bg']
    assert label is None or len(bbox) == len(label), 'Length of label must match that of bbox'
    assert cls is None or len(bbox) == len(cls), 'Length of classification score must match that of bbox'
    ax = vis_image(img, ax=ax)

    if len(bbox) == 0:
        return ax

    for i, bbox_ in enumerate(bbox):
        xy = (bbox_[0], bbox_[1])
        height = bbox_[3] - bbox_[1]
        width = bbox_[2] - bbox_[0]
        ax.add_patch(ppl.Rectangle(xy, width, height, fill=False, edgecolor='red', linewidth=2))

        caption = list()

        if label is not None and label_names is not None:
            lb = label[i]
            assert (-1 <= lb < len(label_names)), 'Label out of range, no corresponding name available'
            caption.append(label_names[lb])
        if cls is not None:
            c = cls[i]
            caption.append('{:.2f}'.format(c))

        if len(caption) > 0:
            ax.text(bbox_[0], bbox_[1], ': '.join(caption),
                    style='italic', bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})

    return ax


def fig2data(fig):
    fig.canvas.draw()

    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    buf = np.roll(buf, 3, axis=2)  # roll from ARGB to RGBA
    return buf.reshape(h, w, 4)


def fig2vis(fig):
    ax = fig.get_figure()
    img_data = fig2data(ax).astype(np.int32)
    ppl.close()
    return img_data[:, :, :3].transpose((2, 0, 1)) / 255.0


def visdom_bbox(*args, **kwargs):
    fig = vis_bbox(*args, **kwargs)  # visualize bbox
    data = fig2vis(fig)
    return data  # return image of shape (3, h, w), RGB


class Visualizer:
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, use_incoming_socket=False, **kwargs)
        self._vis_kw = kwargs
        self.index = {}
        self.log_text = ''

    def configure(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def multi_plot(self, dic):
        # dic: dict (name, value)
        for k, v in dic.items():
            if v is not None:
                self.plot(k, v)

    def multi_img(self, dic):
        for k, v in dic.items:
            if v is not None:
                self.img(k, v)

    def plot(self, name, value, **kwargs):
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([value]),
                      X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append',
                      **kwargs,)
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        self.vis.image(t.Tensor(img_).cpu().numpy(),
                       win=name,
                       opts=dict(title=name),
                       **kwargs,)

    def log(self, info, win='log_text'):
        self.log_text += ('[{time}] {info} <br>'.format(time=time.strftime('%m%d_%H%M%S'),
                                                        info=info,))
        self.vis.text(self.log_text, win)

    def __getattr__(self, item):
        return getattr(self.vis, item)

    def state_dict(self):
        return {'index': self.index,
                'vis_kw': self._vis_kw,
                'log_text': self.log_text,
                'env': self.vis.env, }

    def load_state_dict(self, dic):
        self.vis = visdom.Visdom(env=dic.get('env', self.vis.env), **(self.dic.get('bis_kw')))
        self.log_text = dic.get('log_text', '')
        self.index = dic.get('index', dict())
        return self
