import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))
sources = ["diff_interpolation/forward.cu", "diff_interpolation/backward.cu","interpolation.cpp", "ext.cpp"]

setup(
    name='diff_trilinear_interpolation',
    version='1.0',
    # set external cpp/cuda modules
    ext_modules=[
        CUDAExtension(
            name='diff_trilinear_interpolation._C',
            sources=sources,
            extra_compile_args={'cxx': ['-O2'],'nvcc': ['-O2']})
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
