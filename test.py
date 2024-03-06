import torch

# must import after torch
import my_cuda_kernel

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

feats = torch.ones(2, device=device)
point = torch.zeros(2, device=device)

out = my_cuda_kernel.trilinear_interpolation_forward(feats, point)

print(out)
