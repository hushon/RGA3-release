# RGA3 Development Environment Setup

This document summarizes the environment setup that was completed for this workspace and the compatibility fixes that were required to make the demo run successfully.

## Environment

- OS: Ubuntu 22.04.4 LTS
- Python: 3.10.16
- Conda env: `rga3`
- GPU driver: NVIDIA 580.65.06
- CUDA driver version reported by `nvidia-smi`: 13.0
- Runtime used by PyTorch: CUDA 12.4

## Installed Components

### Conda environment

```bash
conda create -n rga3 python=3.10.16 -y
conda activate rga3
```

### PyTorch stack

The README suggests the conda install below:

```bash
conda install pytorch==2.5.1 torchvision==0.20.1 pytorch-cuda=12.4 -c pytorch -c nvidia -y
```

This was installed first, but importing `torch` caused:

```text
ImportError: ... libtorch_cpu.so: undefined symbol: iJIT_NotifyEvent
```

To fix that, PyTorch was reinstalled from the official CUDA 12.4 pip wheels:

```bash
pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
```

Final working versions:

- `torch 2.5.1+cu124`
- `torchvision 0.20.1+cu124`

### Python packages

Attempted installation:

```bash
pip install -r requirements.txt
```

Two issues in `requirements.txt` required adjustment:

1. `skimage==0.0` is not a valid install target for this environment.
   Use `scikit-image` instead.
2. `transformers==4.49.0.dev0` is not available on PyPI.
   Use `transformers==4.49.0` instead.

Working installation command:

```bash
pip install \
  decord==0.6.0 \
  deepspeed==0.16.3 \
  einops==0.8.1 \
  eva_decord==0.6.1 \
  fvcore==0.1.5.post20221221 \
  matplotlib==3.10.1 \
  numpy==1.26.3 \
  openai==1.65.4 \
  opencv_python==4.10.0.84 \
  packaging==24.2 \
  pandas==2.2.3 \
  peft==0.14.0 \
  Pillow==11.1.0 \
  pycocoevalcap==1.2 \
  pycocotools==2.0.8 \
  qwen_vl_utils==0.0.10 \
  Requests==2.32.3 \
  scipy==1.15.2 \
  Shapely==2.0.7 \
  termcolor==2.5.0 \
  tokenizers==0.21.0 \
  tqdm==4.67.1 \
  transformers==4.49.0 \
  scikit-image
```

### Flash Attention

The README notes a working stack around:

- `torch==2.5.1+cu124`
- `flash_attn==2.7.4.post1`

Initially, `flash-attn 2.8.3` was installed and caused this runtime error during generation:

```text
AttributeError: module 'torch.library' has no attribute 'wrap_triton'
```

Fix:

```bash
pip uninstall -y flash-attn
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

Final working version:

- `flash_attn 2.7.4.post1`

### Additional packages from README

```bash
pip install ninja
pip install git+https://github.com/facebookresearch/sam2.git
pip install git+https://github.com/facebookresearch/co-tracker.git
apt update
apt install -y openjdk-11-jdk zip
```

Installed successfully:

- `ninja`
- `SAM2`
- `CoTracker3`
- `openjdk-11-jdk`
- `zip`

## Repository Compatibility Fixes Applied

To make the local demo run successfully in this environment, two repository-side compatibility fixes were applied.

### 1. Processor metadata fix

File changed:

- `checkpoints/UniGR-7B/preprocessor_config.json`

Change made:

```json
"image_processor_type": "Qwen2_5_VLImageProcessor"
```

to:

```json
"image_processor_type": "Qwen2VLImageProcessor"
```

Reason:

`transformers==4.49.0` expects `Qwen2VLImageProcessor` for this checkpoint layout. Without this change, `AutoProcessor.from_pretrained(...)` failed with an unrecognized image processor error.

### 2. Gradio 6 compatibility fix

File changed:

- `app.py`

Change made in `gr.ImageEditor(...)`:

- removed `show_download_button=False`
- removed `show_share_button=False`
- replaced them with `buttons=[]`

Reason:

The installed Gradio version is 6.x, and the old `ImageEditor` keyword arguments are no longer supported there.

## Demo Launch

Recommended launch command:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate rga3
CUDA_VISIBLE_DEVICES=2 python app.py --version checkpoints/UniGR-7B/
```

Notes:

- Do not assume `GPU 0` is free. This model can fail with CUDA OOM if another process already occupies the default GPU.
- Use `nvidia-smi` to choose a GPU with enough free memory and update `CUDA_VISIBLE_DEVICES` accordingly.

## Verification

The following checks completed successfully:

```bash
python - <<'PY'
import importlib
mods = [
    'torch', 'torchvision', 'deepspeed', 'transformers',
    'flash_attn', 'sam2', 'cotracker', 'cv2', 'skimage'
]
for name in mods:
    module = importlib.import_module(name)
    print(name, 'OK', getattr(module, '__version__', 'unknown'))

import torch
print('cuda_available:', torch.cuda.is_available())
print('cuda_version:', torch.version.cuda)
PY
```

Observed working state:

- `torch: 2.5.1+cu124`
- `torchvision: 0.20.1+cu124`
- `deepspeed: 0.16.3`
- `transformers: 4.49.0`
- `flash_attn: 2.7.4.post1`
- `cv2: 4.10.0`
- `skimage: 0.25.2`
- `cuda_available: True`
- `cuda_version: 12.4`

The Gradio demo also launched successfully after the fixes above.

## Remaining Warnings

The following warnings may still appear, but they do not block execution:

1. Slow image processor warning from `transformers`
2. Gradio warning about `theme` having moved from `Blocks(...)` to `launch()` in Gradio 6
3. Flash Attention message during model loading about moving the model to GPU after CPU initialization

These are non-fatal for the current working setup.