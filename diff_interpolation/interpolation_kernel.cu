#include <torch/extension.h>

template <typename scalar_t>
__global__ void trilinear_forward_kernel(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits,
                                      size_t>
        feats,
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits,
                                      size_t>
        points,
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>
        feat_interp) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int f = blockIdx.y * blockDim.y + threadIdx.y;

  if (n >= feats.size(0) || f >= feats.size(2))
    return;

  // point -1~1
  const scalar_t u = (points[n][0] + 1) / 2;
  const scalar_t v = (points[n][1] + 1) / 2;
  const scalar_t w = (points[n][2] + 1) / 2;

  const scalar_t a = (1 - v) * (1 - w);
  const scalar_t b = (1 - v) * w;
  const scalar_t c = v * (1 - w);
  const scalar_t d = 1 - a - b - c;
  feat_interp[n][f] = (1 - u) * (a * feats[n][0][f] + b * feats[n][1][f] +
                                 c * feats[n][2][f] + d * feats[n][3][f]) +
                      u * (a * feats[n][4][f] + b * feats[n][5][f] +
                           c * feats[n][6][f] + d * feats[n][7][f]);
}

torch::Tensor trilinear_forward_cuda(const torch::Tensor feats,
                                     const torch::Tensor points) {
  // input: feats (N, 8, F), points (N, 3)
  // feats: features of 8 corner nodes of cubes
  // points: target points in cubes
  // N: num of cubes
  // F: dim of features
  // N and F can be pallarelly processed
  // output: feat_interp (N, F)

  const int N = feats.size(0), F = feats.size(2);

  // create empty tensor for output
  torch::Tensor feat_interp = torch::zeros({N, F}, feats.options());
  //   torch::Tensor feat_interp = torch::empty({N, F}, feats.options());

  // set threads
  const dim3 threads(16, 16, 1);  // 256

  // set blocks
  const dim3 blocks((N + threads.x - 1) / threads.x,
                    (F + threads.y - 1) / threads.y, 1);

  // launching kernel
  // AT_DISPATCH_FLOATING_TYPES_HALF  support half precision operation
  AT_DISPATCH_FLOATING_TYPES(
      feats.type(), "trilinear_forward_cuda", ([&] {
        trilinear_forward_kernel<scalar_t><<<blocks, threads>>>(
            feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits,
                                  size_t>(),
            points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits,
                                   size_t>(),
            feat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits,
                                        size_t>());
      }));

  return feat_interp;
}

torch::Tensor trilinear_backward_cuda(const torch::Tensor dL_dfeat_interp,
                                      const torch::Tensor feats,
                                      const torch::Tensor points) {
  return feats;
}
