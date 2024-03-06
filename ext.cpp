#include <torch/extension.h>

#include "interpolation.h"

// define a pybind11 module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // name_:python_module_name, f: cpp/cuda_function_name
  m.def("trilinear_interpolation_forward", &trilinear_interpolation_forward);
  m.def("trilinear_interpolation_backward", &trilinear_interpolation_backward);
}
