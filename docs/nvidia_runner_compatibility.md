# NVIDIA CUDA Compatibility & Training Guide

This guide describes how to run DANet on NVIDIA CUDA-based systems.

## 0. Prerequisite: Environment Setup

Before running training or testing commands, activate your virtual environment. Ensure you have the CUDA toolkit installed and `torch` with CUDA support.

### Activation:
```bash
source ./venv/bin/activate
```

---

## 1. Quick Pipeline Validation

Use the `--fast-test` flag to verify that your GPU is correctly detected and the CUDA kernels (Ninja build) are compiling correctly.

### Run Fast Test:
```bash
python experiments/segmentation/train.py \
    --dataset citys \
    --model danet \
    --backbone resnet101 \
    --checkname nvidia-fast-test \
    --batch-size 4 \
    --workers 4 \
    --fast-test \
    --epochs 1
```

---

## 2. Full Training Run

To start the actual training for 240 epochs (standard for Cityscapes), run the following:

```bash
python experiments/segmentation/train.py \
    --dataset citys \
    --model danet \
    --backbone resnet101 \
    --checkname danet101-nvidia \
    --batch-size 8 \
    --workers 16
```

*Note: You can usually use a higher batch size on NVIDIA cards with more VRAM.*

---

## 3. Testing / Evaluation

To evaluate a trained checkpoint:

```bash
python experiments/segmentation/test.py \
    --dataset citys \
    --model danet \
    --backbone resnet101 \
    --resume runs/citys/danet/resnet101/danet101-nvidia/checkpoint.pth.tar \
    --eval \
    --base-size 2048 \
    --crop-size 768
```

---

## 4. Resuming Training

If training is interrupted, you can resume it using the `--resume` flag:

```bash
python experiments/segmentation/train.py \
    --dataset citys \
    --model danet \
    --backbone resnet101 \
    --checkname danet101-nvidia \
    --resume runs/citys/danet/resnet101/danet101-nvidia/checkpoint.pth.tar \
    --batch-size 8 \
    --workers 16
```

---

## Troubleshooting Tips

- **Ninja required error**: Run `pip install ninja`. This is needed for JIT compilation of CUDA kernels.
- **Out of Memory (OOM)**: Decrease the `--batch-size` if you encounter memory errors.
- **CUDA Version Mismatch**: Ensure your `torch.version.cuda` matches the output of `nvcc --version`.
