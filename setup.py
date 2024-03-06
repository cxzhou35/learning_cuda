import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))
sources = ["diff_interpolation/interpolation_kernel.cu","interpolation.cpp", "ext.cpp"]

setup(
    name='my_cuda_kernel',
    version='1.0',
    # set external cpp/cuda modules
    ext_modules=[
        CUDAExtension(
            name='my_cuda_kernel',
            sources=sources,
            extra_compile_args={'cxx': ['-O2'],'nvcc': ['-O2']})
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
