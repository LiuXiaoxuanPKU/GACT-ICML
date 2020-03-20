from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='quantizers',
      ext_modules=[cpp_extension.CppExtension('quantizers', ['quantizers.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})

