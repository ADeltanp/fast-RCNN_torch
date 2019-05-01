import cupy as cp
import torch as t
from collections import namedtuple
from torch.autograd import Function
from .roi_functions import forward_kernel, backward_kernel

Stream = namedtuple('Stream', ['ptr'])


class RoIPooling2D(Function):
    def __init__(self, out_shape, feat_receptive_len, cuda_threads=1024):
        if len(out_shape.shape) is 1:
            self.h = self.w = out_shape
        else:
            self.h = out_shape[0]
            self.w = out_shape[1]
        self.scale = 1.0 / float(feat_receptive_len)
        self.cuda_threads = cuda_threads
        cuda_lib_f = cp.cuda.compile_with_cache(forward_kernel)
        cuda_lib_b = cp.cuda.compile_with_cache(backward_kernel)
        self.forward_fn = cuda_lib_f.get_function('roi_forward')
        self.backward_fn = cuda_lib_f.get_function('roi_backward')
        super(RoIPooling2D, self).__init__()

    def forward(ctx, features, rois):
        '''
        :param features: (torch.Tensor) feature map output by extractor
        :param rois: (torch.Tensor) roi array of shape (N, 5),
                     where N = batch_size * roi_size_per_image,
                     5 dims are (batch_index, x_min, y_min, x_max, y_max)
        :return: (torch.Tensor) feature map after roi pooling
        '''

        # must be continuous as we use pointer in C later
        features = features.continuous()
        rois = rois.continuous()

        feat_size = B, C, H, W = features.size()
        n_rois = rois.size(0)
        out = t.zeros(n_rois, C, ctx.h, ctx.w).cuda()  # store the output feature map
        max_idx = t.zeros(n_rois, C, ctx.h, ctx.w).int().cuda()  # store the max feature indices
        N = out.numel()  # num of out elements

        args = [features.data_ptr(),
                rois.data_ptr(),
                out.data_ptr(),
                max_idx.data_ptr(),
                ctx.scale, C, H, W,
                ctx.h, ctx.w, N]
        stream = Stream(ptr=t.cuda.current_stream().cuda_stream)

        # N <= block.x * grid.x <= N + cuda_threads - 1,
        # thus guarantee idx in kernel wont duplicate
        ctx.forward_fn(args=args,
                       block=(ctx.cuda_threads, 1, 1),
                       grid=((ctx.cuda_threads + N - 1) // ctx.cuda_threads, 1, 1),
                       stream=stream)

        ctx.save_for_backward(feat_size, rois, max_idx)
        # output of shape (batches * rois_per_batch, C=512, H=7, W=7) by default
        return out

    def backward(ctx, grad):
        grad = grad.continuous()
        feat_size, rois, max_idx = ctx.saved_tensors
        B, C, H, W = feat_size
        rois = rois.continuous()
        max_idx = max_idx.continuous()
        out_grad = t.zeros(B, C, H, W).cuda()
        n_rois = rois.size(0)
        N = out_grad.numel()

        args = [grad.data_ptr(),
                max_idx.data_ptr(),
                rois.data_ptr(),
                out_grad.data_ptr(),
                n_rois, ctx.scale,
                C, H, W, ctx.h, ctx.w,
                N]
        stream = Stream(ptr=t.cuda.current_stream().cuda_stream)

        ctx.backward_fn(args=args,
                        block=(ctx.cuda_threads, 1, 1),
                        grid=((ctx.cuda_threads + N - 1) // ctx.cuda_threads, 1, 1),
                        stream=stream)
        # no backprop for rpn here
        return out_grad, None
