import torch
# must import after torch
from . import _C

# using PyTorch api to wrap the CUDA function
class Trilinear_interpolation_cuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feats, points):
        feat_interp = _C.trilinear_interpolation_forward(feats, points)

        ctx.save_for_backward(feats, points)

        return feat_interp

    @staticmethod
    def backward(ctx, dL_dfeat_interp):
        feats, points = ctx.saved_tensors

        dL_dfeats = _C.trilinear_interpolation_backward(dL_dfeat_interp.contiguous(), feats, points)

        return dL_dfeats, None
