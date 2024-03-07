#ifndef DIFF_INTERPOLATION_FORWARD_H_INCLUDED
#define DIFF_INTERPOLATION_FORWARD_H_INCLUDED

#include <torch/extension.h>

// add micro
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

torch::Tensor trilinear_forward_cuda(const torch::Tensor feats,
                                     const torch::Tensor points);

#endif
