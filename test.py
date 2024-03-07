import torch

import diff_trilinear_interpolation
import time

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def trilinear_interpolation_py(feats, points):
    """
    Inputs:
        feats: (N, 8, F)
        points: (N, 3) local coordinates in [-1, 1]

    Outputs:
        feats_interp: (N, F)
    """
    u = (points[:, 0:1]+1)/2
    v = (points[:, 1:2]+1)/2
    w = (points[:, 2:3]+1)/2
    a = (1-v)*(1-w)
    b = (1-v)*w
    c = v*(1-w)
    d = 1-a-b-c

    feats_interp = (1-u)*(a*feats[:, 0] +
                          b*feats[:, 1] +
                          c*feats[:, 2] +
                          d*feats[:, 3]) + \
                       u*(a*feats[:, 4] +
                          b*feats[:, 5] +
                          c*feats[:, 6] +
                          d*feats[:, 7])

    return feats_interp



def main():
    N = 65536; F = 256
    rand = torch.rand(N, 8, F, device='cuda')
    feats = rand.clone().requires_grad_(True)
    feats2 = rand.clone().requires_grad_(True)
    points = torch.rand(N, 3, device='cuda') * 2 - 1

    t = time.time()
    output_cuda = diff_trilinear_interpolation.Trilinear_interpolation_cuda.apply(feats2, points)
    torch.cuda.synchronize()
    print('CUDA forward time:', time.time()-t, 's')

    t = time.time()
    output_py = trilinear_interpolation_py(feats, points)
    torch.cuda.synchronize()
    print('PyTorch forward time:', time.time()-t, 's')

    print('Forward all close', torch.allclose(output_py, output_cuda), '\n')

    t = time.time()
    loss_cuda = output_cuda.sum()
    loss_cuda.backward()
    torch.cuda.synchronize()
    print('CUDA backward time:', time.time()-t, 's')

    t = time.time()
    loss_py = output_py.sum()
    loss_py.backward()
    torch.cuda.synchronize()
    print('PyTorch backward time:', time.time()-t, 's')

    print('Backward all close', torch.allclose(feats.grad, feats2.grad), '\n')

if __name__ == '__main__':
    main()
