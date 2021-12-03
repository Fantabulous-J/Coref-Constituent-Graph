from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='coref_cpp',
    ext_modules=[
        CppExtension('coref_cpp', ['coref.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })