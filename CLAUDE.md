# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RGA3 (UniGR) is a multimodal model for object-centric video question answering with visual grounding and referring. It combines:
- **Qwen2.5-VL-7B** as the vision-language backbone (fine-tuned via LoRA)
- **SAM2** (Segment Anything Model 2) as the grounding encoder for mask prediction
- **CoTracker3** (STOM module) for tracking visual prompts across video frames

The trained checkpoint is called `UniGR-7B`.

## Environment Setup

Conda env: `rga3` | Python 3.10.16 | torch 2.5.1+cu124 | flash_attn 2.7.4.post1

```bash
conda create -n rga3 python=3.10.16 -y && conda activate rga3
# Install PyTorch via pip wheels (NOT conda — conda wheels have a known symbol error)
pip install --force-reinstall torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt  # NOTE: replace 'skimage==0.0' with 'scikit-image' and 'transformers==4.49.0.dev0' with 'transformers==4.49.0'
pip install ninja
pip install flash-attn==2.7.4.post1 --no-build-isolation  # Must be exactly this version; 2.8.x breaks generation
pip install git+https://github.com/facebookresearch/sam2.git
pip install git+https://github.com/facebookresearch/co-tracker.git
apt update && apt install -y openjdk-11-jdk zip
```

**Known compatibility fix**: In `checkpoints/UniGR-7B/preprocessor_config.json`, set `"image_processor_type": "Qwen2VLImageProcessor"` (not `Qwen2_5_VLImageProcessor`) for `transformers==4.49.0`.

## Running the Demo

```bash
source /opt/conda/etc/profile.d/conda.sh && conda activate rga3
CUDA_VISIBLE_DEVICES=2 python app.py --version checkpoints/UniGR-7B/
```

Check `nvidia-smi` first — the model requires substantial VRAM. `app.py` patches `flash_attn` imports at startup so it runs without flash attention.

## Training

```bash
bash run_torchrun.sh   # multi-node torchrun; set WORLD_SIZE, GPU_NUM, RANK, MASTER_ADDR, MASTER_PORT
bash merge.sh          # after training: convert DeepSpeed ZeRO → fp32, then merge LoRA weights
```

Key training args are in `run_torchrun.sh`: 8 frames for MLLM, 4 for SAM, LoRA r=128 α=256, bf16, DeepSpeed ZeRO.

## Evaluation

Each benchmark has its own subfolder under `evaluation/`. General pattern:

```bash
bash evaluation/<benchmark>/run_inference_<benchmark>.sh   # Step 1: inference
bash evaluation/<benchmark>/run_eval_<benchmark>.sh        # Step 2: metrics
```

Benchmarks: `mevis_val_u`, `refytvos`, `refdavis`, `revos`, `reason_vos`, `videoinfer`, `videorefer_bench`, `vipbench`, `eval_img`.

For VideoInfer GPT-4 scoring, see `eval_gpt.ipynb` in `evaluation/videoinfer/`.

## Batch Segmentation (NExT-QA preprocessing)

```bash
# Generates mask PNG sequences for NExT-QA visual prompting
bash run_batch_segment.sh
# or directly:
python batch_segment_nextqa_v2.py --version ./checkpoints/UniGR-7B \
  --anno_path ../nextqa/annotations_mc/val.csv \
  --output_dir ./next_qa_segmentation_masks_v2 --num_videos -1
```

## Architecture

### Model (`model/`)
- **`qwen_2_5_vl_sam2.py`**: Core model. `UniGRModel` extends `Qwen2_5_VLForConditionalGeneration`. At inference, `[SEG]` token embeddings are extracted and projected through `text_hidden_fcs` (MLP) into SAM2's embedding space. SAM2 then decodes segmentation masks.
- **`sam2.py`**: Wraps SAM2 with an interface for language-conditioned mask prediction.
- **`qwen_2_5_vl.py`**: Qwen2.5-VL backbone utilities.
- **`STOM.py`**: Spatio-Temporal Object Motion module — wraps CoTracker3 to propagate visual prompts across video frames by tracking from a query frame.

### Utilities (`utils/`)
- **`dataset.py`**: `ImgVidHybridDataset` for training; imports all dataset classes. `collate_fn` builds batches including `images_sam` tensor and mask targets.
- **`visual_prompt_generator.py`**: Generates visual prompts (rectangles, ellipses, arrows, masks, etc.) drawn onto frames; `blend_image_from_mask` and `video_blending_keyframes` are the main entry points used in `app.py`.
- **`visual_prompt_organizer.py`**: Organizes visual prompts across video keyframes.
- **`utils.py`**: Shared helpers including `preprocess` (SAM2 image preprocessing), `DirectResize`, `get_sparse_indices` (uniform frame sampling), `dict_to_cuda`.
- Individual dataset files (`vos_dataset.py`, `mevis_dataset.py`, `revos_dataset.py`, etc.) each implement a `__getitem__` returning `(image_path, images, messages, masks, label, resize, inference)`.

### Inference flow (`app.py`)
1. User draws a visual prompt on a video frame → STOM tracks it across frames → visual prompts overlaid on sampled frames
2. Frames + visual prompts assembled into Qwen2.5-VL chat messages
3. `UniGRModel.generate()` produces text; if `[SEG]` is predicted, its hidden state feeds SAM2 to produce a segmentation mask
4. Masks rendered back onto video and displayed via Gradio

### `utils_gcg/`
CLIP-based utilities (tokenizer, feature extraction, frame padding) used for GCG (Grounded Caption Generation) tasks.

## Coding Guidelines

Derived from [Andrej Karpathy's observations](https://x.com/karpathy/status/2015883857489522876) on LLM coding pitfalls.

### Think Before Coding
- State assumptions explicitly before implementing. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If something is unclear, stop and name what's confusing.

### Simplicity First
- Minimum code that solves the problem. Nothing speculative.
- No features beyond what was asked. No abstractions for single-use code.
- No error handling for impossible scenarios.
- Ask: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### Surgical Changes
- Touch only what you must. Don't "improve" adjacent code, comments, or formatting.
- Match existing style, even if you'd do it differently.
- Remove imports/variables/functions that YOUR changes made unused — but don't remove pre-existing dead code unless asked.
- Every changed line should trace directly to the user's request.

### Goal-Driven Execution
- Transform tasks into verifiable goals: "Fix the bug" → "Write a test that reproduces it, then make it pass."
- For multi-step tasks, state a brief plan with verification steps before starting.
