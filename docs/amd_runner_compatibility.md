# AMD ROCm Compatibility & Training Guide

This guide describes how to run DANet on AMD ROCm-based systems (specifically tested on **AMD RX 7600 / gfx1100** architectures).

## 0. Prerequisite: Environment Setup

Before running any training or testing commands in a **new terminal**, you must activate the virtual environment and set the mandatory ROCm environment variables.

### One-liner for Bash:
```bash
source ./venv/bin/activate && \
export HSA_OVERRIDE_GFX_VERSION=11.0.0 && \
export PYTORCH_ROCM_ARCH="gfx1100" && \
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
```

---

## 1. Quick Pipeline Validation (Recommended First Step)

I have added a `--fast-test` flag to the `train.py` script. This runs only 5 batches of training and 5 batches of validation to verify that:
- Your GPU is correctly detected.
- The C++ kernels (Ninja build) are compiling correctly.
- The validation loop (where we fixed the shape bug) won't crash after the first epoch.

### Run Fast Test:
```bash
python experiments/segmentation/train.py \
    --dataset citys \
    --model danet \
    --backbone resnet101 \
    --checkname fast-test-run \
    --batch-size 2 \
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
    --checkname danet101-run \
    --batch-size 2 \
    --workers 4
```

---

## 3. Testing / Evaluation

To evaluate a trained checkpoint (e.g., once you have one in the `runs/` folder):

```bash
python experiments/segmentation/test.py \
    --dataset citys \
    --model danet \
    --backbone resnet101 \
    --resume <path_to_checkpoint>/checkpoint.pth.tar \
    --eval \
    --base-size 2048 \
    --crop-size 768
```

---

## 4. Resuming Training

Since a full epoch on Cityscapes can take ~5 hours, you may need to stop training (using `Ctrl+C`) and resume it later. 

**Wait for the current epoch to finish** to ensure a checkpoint is saved. Checkpoints are automatically saved at the end of every epoch.

### How to Resume:
Add the `--resume` flag and point it to the latest `checkpoint.pth.tar` in your `runs/` directory:

```bash
python experiments/segmentation/train.py \
    --dataset citys \
    --model danet \
    --backbone resnet101 \
    --checkname danet101-run \
    --resume runs/citys/danet/resnet101/danet101-run/checkpoint.pth.tar \
    --batch-size 2 \
    --workers 4
```

---

## Troubleshooting Tips

- **Ninja required error**: If you get a "Ninja is required" error, run `pip install ninja`. This is needed for JIT compilation of C++ kernels.
- **Out of Memory (OOM)**: If training crashes with an OOM error, **decrease** the `--batch-size` (e.g., to 1 or 2) or ensure `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` is set.
- **Architecture mismatch**: If you are using a different AMD card, you may need to update `gfx1100` and `11.0.0` to match your hardware's specific architecture version.
