from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='pytorch_minimax',
    ext_modules=[
        CUDAExtension('pytorch_minimax', [
            'minimax_cuda.cpp',
            'minimax_cuda_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
