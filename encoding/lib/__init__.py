import os
import torch
from torch.utils.cpp_extension import load

cwd = os.path.dirname(os.path.realpath(__file__))
cpu_path = os.path.join(cwd, 'cpu')
gpu_path = os.path.join(cwd, 'gpu')

try:
    cpu = load('enclib_cpu', [
            os.path.join(cpu_path, 'operator.cpp'),
            os.path.join(cpu_path, 'encoding_cpu.cpp'),
            os.path.join(cpu_path, 'syncbn_cpu.cpp'),
            os.path.join(cpu_path, 'roi_align_cpu.cpp'),
            os.path.join(cpu_path, 'nms_cpu.cpp'),
            os.path.join(cpu_path, 'rectify_cpu.cpp'),
        ], build_directory=cpu_path, verbose=False)
except Exception as e:
    print(f"Failed to load enclib_cpu: {e}")
    cpu = None

if torch.cuda.is_available():
    # Detect if we're using ROCm (HIP)
    is_hip = hasattr(torch.version, 'hip') and torch.version.hip is not None
    
    # We use the original .cu files; PyTorch JIT will HIPify them automatically on ROCm
    sources = [
        os.path.join(gpu_path, 'operator.cpp'),
        os.path.join(gpu_path, 'activation_kernel.cu'),
        os.path.join(gpu_path, 'encoding_kernel.cu'),
        os.path.join(gpu_path, 'syncbn_kernel.cu'),
        os.path.join(gpu_path, 'roi_align_kernel.cu'),
        os.path.join(gpu_path, 'nms_kernel.cu'),
        os.path.join(gpu_path, 'rectify_cuda.cu'),
    ]
    # lib_ssd is removed because it depends on obsolete THC headers
    
    extra_flags = [] if is_hip else ["--expt-extended-lambda"]
    
    try:
        gpu = load('enclib_gpu', sources, 
            extra_cuda_cflags=extra_flags,
            build_directory=gpu_path, verbose=True)
    except Exception as e:
        print(f"Failed to load enclib_gpu: {e}")
        gpu = None
else:
    gpu = None
