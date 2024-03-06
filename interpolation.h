#pragma once
#include <torch/extension.h>

torch::Tensor trilinear_interpolation_forward(const torch::Tensor feats,
                                              const torch::Tensor points);
torch::Tensor trilinear_interpolation_backward(
    const torch::Tensor dL_dfeat_interp, const torch::Tensor feats,
    const torch::Tensor points);
