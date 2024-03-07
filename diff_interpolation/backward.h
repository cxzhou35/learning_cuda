#ifndef DIFF_INTERPOLATION_BACKWARD_H_INCLUDED
#define DIFF_INTERPOLATION_BACKWARD_H_INCLUDED

#include <torch/extension.h>

torch::Tensor trilinear_backward_cuda(const torch::Tensor dL_dfeat_interp,
                                      const torch::Tensor feats,
                                      const torch::Tensor points);

#endif
