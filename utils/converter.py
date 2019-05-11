import torch as t
import numpy as np
# the return and the input of each function below share the same memory


def to_numpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, t.Tensor):
        return data.detach().cpu().numpy()
    raise TypeError("instance must be either np.ndarray or torch.Tensor")


def to_tensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = t.from_numpy(data)
        if cuda:
            tensor = tensor.cuda()
        return tensor

    if isinstance(data, t.Tensor):
        tensor = data.detach()
        if cuda:
            tensor = tensor.cuda()
        return tensor

    raise TypeError("instance must be either np.ndarray or torch.Tensor")
