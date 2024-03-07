# learning_cuda

Introduction to cpp/cuda extension for PyTorch and building first cpp bridge.

**PyTorch -> Cpp -> CUDA**

In this tutorial, we will implement a PyTorch CUDA extension called **diff_trilinear_interpolation**. The forward and backward modules are implemented in CUDA and the C++ bridge is used to connect the PyTorch and CUDA.

> [Tutorial Video by YouTuber AI葵](https://www.youtube.com/watch?v=l_Rpk6CRJYI) | [Github Repo](https://github.com/kwea123/pytorch-cppcuda-tutorial)

## References

- CUDA explanation: https://nyu-cds.github.io/python-gpu/02-cuda/
- C++ API: https://pytorch.org/cppdocs/
- Kernel launching: https://pytorch.org/tutorials/advanced/cpp_extension.html

## File Structures

```
.
├── build
├── diff_interpolation                 # cuda implementation
│   ├── backward.cu
│   ├── backward.h
│   ├── forward.cu
│   └── forward.h
├── diff_trilinear_interpolation       # python api
│   ├── __init__.py
├── ext.cpp
├── interpolation.cpp                  # cpp bridge script
├── interpolation.h
├── README.md
├── setup.py                           # pybind build script
└── test.py                            # test script
```

## Test

```python
cd learning_cuda
pip install .

python test.py
```

results:

```
CUDA forward time: 0.002805948257446289 s
PyTorch forward time: 0.004231929779052734 s
Forward all close True

CUDA backward time 0.005219936370849609 s
PyTorch backward time 0.03641915321350098 s
Backward all close True
```
