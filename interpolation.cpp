#include <torch/extension.h>

#include "diff_interpolation/interpolation_kernel.h"

torch::Tensor trilinear_interpolation_forward(const torch::Tensor feats,
                                              const torch::Tensor points) {
  // check input tensors
  CHECK_INPUT(feats);
  CHECK_INPUT(points);

  // invoke cuda implementation
  return trilinear_forward_cuda(feats, points);
}

torch::Tensor trilinear_interpolation_backward(
    const torch::Tensor dL_dfeat_interp, const torch::Tensor feats,
    const torch::Tensor points) {
  // check input tensors

  CHECK_INPUT(dL_dfeat_interp);
  CHECK_INPUT(feats);
  CHECK_INPUT(points);

  // invoke cuda implementation
  return trilinear_backward_cuda(dL_dfeat_interp, feats, points);
}
