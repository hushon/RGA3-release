<div align="center">
<h1> VideoQA with visual prompting </h1>

![demo](assets/demo.gif)

</div>

This repository provides an interactive demo for VideoQA with uncertainty-aware visual prompting using NextQA dataset.
It is designed to let a user enter a VideoQA question, generate visual prompted keyframes through a keyframe selection module and a mask generation module, and then run LMM inference to produce both an answer and an uncertainty score.
If the uncertainty suggests that the prediction should be revisited, the user can intervene at the mask generation stage, regenerate the visual prompts with a human-selected mask, and run inference again to obtain the final result.
The demo uses RGA3 for question answering and SAM2 for grounding and mask prediction.

## Environment

We recommend using the Docker image `hushon/rga3` for the demo and environment setup.
After entering the Docker container, activate the conda environment with `conda activate rga3` before running the demo.

```bash
docker run --gpus all -it --rm -v $PWD:/workspace -w /workspace hushon/rga3 bash
conda activate rga3
```

If you prefer to build the environment manually, please refer to the original project: https://github.com/qirui-chen/RGA3-release.

Next, download the [RGA3 checkpoints 🤗](https://huggingface.co/SurplusDeficit/UniGR-7B) and place them in `./checkpoints/UniGR-7B/` under the project root.


## Launch the demo

After downloading checkpoints and preparing the environment, run the demo from the project root with:

```bash
python app_nextqa.py --version ./checkpoints/UniGR-7B
```

This launches the Gradio demo interface for NextQA inference.
The demo was tested on a single NVIDIA RTX 4090 (24GB) GPU.

## Acknowledgements

- Our codes are based on the original [RGA3-release](https://github.com/qirui-chen/RGA3-release) project, [LISA](https://github.com/dvlab-research/LISA/) & [VideoLISA](https://github.com/showlab/VideoLISA/). The copyright for adding language embedding in SAM2 belongs to [Sa2VA](https://github.com/magic-research/Sa2VA). The implementation of generating and processing visual prompts is based on [ViP-LLaVA](https://github.com/WisconsinAIVision/ViP-LLaVA).

- We also thank the open-source projects like [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL), [CoTracker3](https://github.com/facebookresearch/co-tracker) and [SAM2](https://github.com/facebookresearch/sam2).
