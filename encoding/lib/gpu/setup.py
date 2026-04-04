import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Detect if we are on AMD (ROCm) or NVIDIA (CUDA)
is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None

if is_rocm:
    # AMD path
    sources = [
        'operator.cpp',
        'activation_kernel.hip',
        'encoding_kernel.hip',
        'syncbn_kernel.hip',
        'roi_align_kernel.hip',
        'nms_kernel.hip',
        'rectify_hip.hip',
    ]
else:
    # NVIDIA path
    sources = [
        'operator.cpp',
        'activation_kernel.cu',
        'encoding_kernel.cu',
        'syncbn_kernel.cu',
        'roi_align_kernel.cu',
        'nms_kernel.cu',
        'rectify_cuda.cu',
    ]

setup(
    name='enclib_gpu',
    ext_modules=[
        CUDAExtension('enclib_gpu', sources),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
