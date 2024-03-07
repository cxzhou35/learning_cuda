# learning_cuda

Introduction to cpp/cuda extension for PyTorch and building first cpp bridge.

**PyTorch -> Cpp -> CUDA**

[Tutorial Video](https://www.youtube.com/watch?v=l_Rpk6CRJYI) | [Github Repo](https://github.com/kwea123/pytorch-cppcuda-tutorial)

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
