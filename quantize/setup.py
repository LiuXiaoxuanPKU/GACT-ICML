from setuptools import setup, Extension, find_packages
from torch.utils import cpp_extension

print(find_packages())

setup(name='quantize',
      ext_modules=[cpp_extension.CppExtension('C', ['quantizers.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension},
      packages=find_packages()
)

