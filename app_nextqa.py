import argparse
import os
import sys
import subprocess
import shutil
import tempfile
import types
import importlib.util

# Disable flash attention BEFORE any transformers import
os.environ["TRANSFORMERS_NO_FLASH_ATTN_2"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

fake_module = types.ModuleType('flash_attn')
sys.modules['flash_attn'] = fake_module
sys.modules['flash_attn_2_cuda'] = types.ModuleType('flash_attn_2_cuda')
sys.modules['flash_attn.bert_padding'] = types.ModuleType('bert_padding')

_original_find_spec = importlib.util.find_spec
def _patched_find_spec(name, package=None):
    if 'flash_attn' in name:
        return None
    return _original_find_spec(name, package)
importlib.util.find_spec = _patched_find_spec

import gradio as gr
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from model.qwen_2_5_vl_sam2 import UniGRConfig, UniGRModel
from utils.utils import DirectResize, get_sparse_indices, dict_to_cuda, preprocess
from utils.visual_prompt_generator import blend_image_from_mask


SEG_PROMPTS = [
    ("Mask 1", lambda q: f"Can you segment the key object mentioned in this question? Question: {q}"),
    ("Mask 2", lambda q: f"Can you segment everything mentioned in this prompt? \"{q}\""),
    ("Mask 3", lambda q: f"Segment the main subject (the person or object performing the primary action) in this question: {q}"),
]
SEG_PROMPT_NAMES = [name for name, _ in SEG_PROMPTS]

CHOICE_LETTERS = ["A", "B", "C", "D", "E"]


def parse_args():
    parser = argparse.ArgumentParser(description="NExT-QA VideoQA Demo")
    parser.add_argument("--version", default="/PATH/TO/UniGR-7B", help="Model checkpoint path")
    parser.add_argument("--num_frames_mllm", default=8, type=int,
                        help="Number of frames for MLLM segmentation context")
    parser.add_argument("--max_pixels", default=384 * 28 * 28, type=int)
    parser.add_argument("--image_size", default=1024, type=int)
    return parser.parse_args()


args = parse_args()

print("Loading model...")
processor = AutoProcessor.from_pretrained(args.version)
tokenizer = processor.tokenizer
args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[-1]

config = UniGRConfig.from_pretrained(
    args.version,
    train_mask_decoder=False,
    seg_token_idx=args.seg_token_idx,
    sam_pretrained=None,
)
try:
    model = UniGRModel.from_pretrained(
        args.version,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=False,
    )
except (ImportError, RuntimeError) as e:
    print(f"Flash attention 2 failed: {e}. Falling back to default attention.")
    model = UniGRModel.from_pretrained(
        args.version,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )

model = model.bfloat16().cuda().eval()
transform = DirectResize(args.image_size)
print("Model loaded successfully!")

# Verify A/B/C/D/E are each a single token (required for confidence extraction)
_choice_token_ids = []
for _c in CHOICE_LETTERS:
    _ids = tokenizer(_c, add_special_tokens=False).input_ids
    assert len(_ids) == 1, f"Token '{_c}' is not a single token: {_ids}"
    _choice_token_ids.append(_ids[0])
print(f"Choice token IDs: {dict(zip(CHOICE_LETTERS, _choice_token_ids))}")


def create_video_from_frames(frames, output_path, fps=15):
    try:
        if not frames:
            return None

        first_frame = frames[0]
        if isinstance(first_frame, Image.Image):
            width, height = first_frame.size
            frames_array = [np.array(frame.convert('RGB')) for frame in frames]
        else:
            height, width = first_frame.shape[:2]
            frames_array = frames

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            output_avi = os.path.splitext(output_path)[0] + '.avi'
            out = cv2.VideoWriter(output_avi, fourcc, fps, (width, height))
            output_path = output_avi

        for frame in frames_array:
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            out.write(frame_bgr)

        out.release()

        try:
            ffmpeg_path = shutil.which("ffmpeg")
            if ffmpeg_path is not None:
                base_path, _ = os.path.splitext(output_path)
                h264_target = base_path + "_h264.mp4"
                h264_cmd = [ffmpeg_path, "-y", "-loglevel", "error", "-i", output_path,
                            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", h264_target]
                try:
                    subprocess.run(h264_cmd, check=True)
                    output_path = h264_target
                except Exception:
                    vp9_target = base_path + "_vp9.webm"
                    vp9_cmd = [ffmpeg_path, "-y", "-loglevel", "error", "-i", output_path,
                               "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "30", vp9_target]
                    try:
                        subprocess.run(vp9_cmd, check=True)
                        output_path = vp9_target
                    except Exception as e2:
                        print(f"Warning: ffmpeg VP9 re-encode failed: {e2}")
        except Exception as e:
            print(f"Warning: ffmpeg re-encode failed: {e}")

        return output_path
    except Exception as e:
        print(f"Error creating video: {e}")
        return None


def _apply_masks_to_frames(all_pil, pred_masks):
    """Apply per-frame masks; return (prompted_frames, num_masked)."""
    total_frames = len(all_pil)
    prompted_frames = []
    total_masked = 0
    if len(pred_masks) > 0 and pred_masks[0].shape[0] > 0:
        pred_mask_vid = pred_masks[0].numpy() if not pred_masks[0].is_cuda else pred_masks[0].detach().cpu().numpy()
        for i in range(total_frames):
            if i < pred_mask_vid.shape[0]:
                frame_mask = pred_mask_vid[i] > 0
                if np.sum(frame_mask) >= 100:
                    blended = blend_image_from_mask(
                        all_pil[i].convert("RGB"), frame_mask.astype(np.float32), "red", "mask"
                    )
                    prompted_frames.append(blended)
                    total_masked += 1
                    continue
            prompted_frames.append(all_pil[i])
    else:
        prompted_frames = list(all_pil)
    return prompted_frames, total_masked


def _make_mask_frames(all_pil, pred_masks_cpu):
    """White-on-black mask visualization for each frame."""
    total_frames = len(all_pil)
    mask_frames = []
    if len(pred_masks_cpu) > 0 and pred_masks_cpu[0].shape[0] > 0:
        pred_mask_vid = pred_masks_cpu[0].numpy()
        for i in range(total_frames):
            h, w = np.array(all_pil[i]).shape[:2]
            vis = np.zeros((h, w, 3), dtype=np.uint8)
            if i < pred_mask_vid.shape[0]:
                vis[pred_mask_vid[i] > 0] = 255
            mask_frames.append(Image.fromarray(vis))
    else:
        for pil in all_pil:
            h, w = np.array(pil).shape[:2]
            mask_frames.append(Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8)))
    return mask_frames


def _run_single_segmentation(prompt_fn, question, state_data):
    """
    Run segmentation for one prompt. Reuses preprocessed SAM tensors from state_data.
    Returns (prompted_frames, mask_video_path).
    """
    all_pil = state_data["all_pil"]
    image_sam = state_data["image_sam"]  # CPU tensor
    resize_list = state_data["resize_list"]
    original_size_list = state_data["original_size_list"]
    frames_for_seg = state_data["frames_for_seg"]

    seg_prompt = prompt_fn(question.strip())
    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": frames_for_seg, "max_pixels": args.max_pixels},
            {"type": "text", "text": seg_prompt},
        ]},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Sure, [SEG]."},
        ]},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=text, images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt", **video_kwargs,
    )
    inputs = dict_to_cuda(inputs)

    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs["pixel_values"].bfloat16() if "pixel_values" in inputs else None
    pixel_values_videos = inputs["pixel_values_videos"].bfloat16() if "pixel_values_videos" in inputs else None
    image_grid_thw = inputs.get("image_grid_thw")
    video_grid_thw = inputs.get("video_grid_thw")
    second_per_grid_ts = inputs.get("second_per_grid_ts")

    image_sam_gpu = image_sam.cuda()
    with torch.inference_mode():
        output_ids, pred_masks = model.evaluate(
            input_ids, attention_mask,
            pixel_values, pixel_values_videos,
            image_grid_thw, video_grid_thw, second_per_grid_ts,
            image_sam_gpu, resize_list, original_size_list,
        )

    pred_masks_cpu = [m.detach().cpu() if torch.is_tensor(m) else m for m in pred_masks]
    del pred_masks, image_sam_gpu
    del inputs, input_ids, attention_mask, pixel_values, pixel_values_videos
    del image_grid_thw, video_grid_thw, second_per_grid_ts, output_ids
    torch.cuda.empty_cache()

    prompted_frames, _ = _apply_masks_to_frames(all_pil, pred_masks_cpu)
    mask_frames = _make_mask_frames(all_pil, pred_masks_cpu)
    del pred_masks_cpu

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        mask_video_path = f.name
    mask_video_path = create_video_from_frames(mask_frames, mask_video_path)

    return prompted_frames, mask_video_path


def _run_qa_inference(prompted_frames, question, opt_a, opt_b, opt_c, opt_d, opt_e):
    """
    Run QA inference on prompted_frames.
    Returns (answer, confidence_dict, prompted_video_path).
    confidence_dict: {"A. <opt_a>": prob, ...} normalized over 5 choices.
    """
    opts = [opt_a, opt_b, opt_c, opt_d, opt_e]

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        prompted_video_path = f.name
    prompted_video_path = create_video_from_frames(prompted_frames, prompted_video_path)

    qa_sample_idxs = get_sparse_indices(len(prompted_frames), 4)
    qa_frames = [prompted_frames[i] for i in qa_sample_idxs]

    qa_prompt = (
        "Look at the marked region in the video frames and answer the multiple-choice question.\n"
        f"Question: {question.strip()}\n"
        f"A. {opt_a}\nB. {opt_b}\nC. {opt_c}\nD. {opt_d}\nE. {opt_e}\n"
        "Answer with just the letter (A, B, C, D, or E)."
    )
    qa_messages = [
        {"role": "user", "content": [
            {"type": "video", "video": qa_frames, "max_pixels": args.max_pixels},
            {"type": "text", "text": qa_prompt},
        ]}
    ]

    qa_text = processor.apply_chat_template(qa_messages, tokenize=False, add_generation_prompt=True)
    qa_image_inputs, qa_video_inputs, qa_video_kwargs = process_vision_info(qa_messages, return_video_kwargs=True)
    qa_inputs = processor(
        text=qa_text, images=qa_image_inputs, videos=qa_video_inputs,
        padding=False, return_tensors="pt", **qa_video_kwargs,
    ).to("cuda")

    with torch.inference_mode():
        generated = model.generate(
            **qa_inputs,
            max_new_tokens=16,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True,
        )

    # Extract answer string
    trimmed = [out[len(inp):] for inp, out in zip(qa_inputs.input_ids, generated.sequences)]
    answer = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    # Compute confidence over A/B/C/D/E from first generated token logits
    first_logits = generated.scores[0][0]  # shape: (vocab_size,)
    probs = torch.softmax(first_logits.float(), dim=-1)
    choice_probs = probs[_choice_token_ids]  # shape: (5,)
    choice_probs = choice_probs / choice_probs.sum()
    choice_probs = choice_probs.cpu().tolist()

    # List of lists preserves A→E order (gr.Dataframe doesn't sort like gr.Label does)
    conf_table = [
        [letter, opt, f"{prob:.1%}"]
        for letter, opt, prob in zip(CHOICE_LETTERS, opts, choice_probs)
    ]

    torch.cuda.empty_cache()
    return answer, conf_table, prompted_video_path


def run_stage1(frame_files, question, opt_a, opt_b, opt_c, opt_d, opt_e, progress=gr.Progress()):
    try:
        if not frame_files:
            return None, None, None, "", None, None, "Please upload video frame images.", gr.update()
        if not question or not question.strip():
            return None, None, None, "", None, None, "Please enter a question.", gr.update()
        if not all([opt_a, opt_b, opt_c, opt_d, opt_e]):
            return None, None, None, "", None, None, "Please fill in all five answer options (A–E).", gr.update()

        progress(0.05, desc="Loading frames...")

        paths = sorted(
            [f.name if hasattr(f, 'name') else f for f in frame_files],
            key=lambda p: os.path.basename(p)
        )
        all_pil = [Image.open(p).convert("RGB") for p in paths]
        total_frames = len(all_pil)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            input_video_path = f.name
        input_video_path = create_video_from_frames(all_pil, input_video_path)

        progress(0.15, desc="Preprocessing frames for SAM...")

        image_list_sam = []
        resize_list = []
        original_size_list = []

        for idx in range(total_frames):
            img_np = np.array(all_pil[idx])
            original_size_list.append(img_np.shape[:2])
            img_resized = transform.apply_image(img_np)
            resize_list.append(img_resized.shape[:2])
            tensor = preprocess(
                torch.from_numpy(img_resized).permute(2, 0, 1).contiguous()
            ).unsqueeze(0).cuda().bfloat16()
            image_list_sam.append(tensor)

        # Keep image_sam on CPU in state to free GPU memory between calls
        image_sam = torch.stack(image_list_sam, dim=1).cpu()
        del image_list_sam

        mllm_idxs = get_sparse_indices(total_frames, args.num_frames_mllm)
        frames_for_seg = [all_pil[i] for i in mllm_idxs]

        state_data = {
            "all_pil": all_pil,
            "image_sam": image_sam,
            "resize_list": resize_list,
            "original_size_list": original_size_list,
            "frames_for_seg": frames_for_seg,
            "prompted_frames": [None, None, None],
            "mask1_video_path": None,
            "input_video_path": input_video_path,
        }

        progress(0.30, desc="Segmenting (Mask 1)...")

        prompted_frames_1, mask1_video_path = _run_single_segmentation(
            SEG_PROMPTS[0][1], question, state_data
        )
        state_data["prompted_frames"][0] = prompted_frames_1
        state_data["mask1_video_path"] = mask1_video_path

        progress(0.65, desc="Running QA inference...")

        answer, confidence_dict, prompted_video_path = _run_qa_inference(
            prompted_frames_1, question, opt_a, opt_b, opt_c, opt_d, opt_e
        )

        progress(1.0, desc="Done!")

        return (
            input_video_path,
            mask1_video_path,
            prompted_video_path,
            answer,
            confidence_dict,
            state_data,
            f"Stage 1 complete. Answer: {answer}",
            gr.update(open=True),   # stage1_accordion
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, None, "", None, None, f"Error: {e}", gr.update()


def run_stage2(state_data, question, opt_a, opt_b, opt_c, opt_d, opt_e, progress=gr.Progress()):
    try:
        if state_data is None:
            return None, None, None, None, state_data, "Please run Stage 1 first.", gr.update(), gr.update()

        progress(0.10, desc="Segmenting (Mask 2)...")

        prompted_frames_2, mask2_video_path = _run_single_segmentation(
            SEG_PROMPTS[1][1], question, state_data
        )
        state_data["prompted_frames"][1] = prompted_frames_2

        progress(0.55, desc="Segmenting (Mask 3)...")

        prompted_frames_3, mask3_video_path = _run_single_segmentation(
            SEG_PROMPTS[2][1], question, state_data
        )
        state_data["prompted_frames"][2] = prompted_frames_3

        progress(1.0, desc="Done!")

        input_video_path = state_data.get("input_video_path")
        mask1_video_path = state_data.get("mask1_video_path")
        return (
            input_video_path,        # stage2_input_video
            mask1_video_path,        # mask_video_0
            mask2_video_path,        # mask_video_1
            mask3_video_path,        # mask_video_2
            state_data,
            "Mask 2 and Mask 3 generated. Select a mask and click Re-run QA.",
            gr.update(open=False),   # stage1_accordion: collapse
            gr.update(visible=True), # stage2_view: show
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, None, None, state_data, f"Error: {e}", gr.update(), gr.update()


def run_qa_final(mask_selection, state_data, question, opt_a, opt_b, opt_c, opt_d, opt_e, progress=gr.Progress()):
    try:
        if state_data is None:
            return None, "", None, "Please run Stage 1 first."
        if not all([opt_a, opt_b, opt_c, opt_d, opt_e]):
            return None, "", None, "Please fill in all five answer options (A–E)."

        selected_idx = SEG_PROMPT_NAMES.index(mask_selection) if mask_selection in SEG_PROMPT_NAMES else 0
        prompted_frames = state_data["prompted_frames"][selected_idx]

        if prompted_frames is None:
            return None, "", None, f"{mask_selection} has not been generated yet. Please run Stage 2 first."

        progress(0.20, desc="Running QA inference...")

        answer, confidence_dict, prompted_video_path = _run_qa_inference(
            prompted_frames, question, opt_a, opt_b, opt_c, opt_d, opt_e
        )

        progress(1.0, desc="Done!")

        status = (
            f"Mask used: {mask_selection}\n"
            f"Question: {question}\n"
            f"A. {opt_a}  B. {opt_b}  C. {opt_c}  D. {opt_d}  E. {opt_e}\n"
            f"Predicted answer: {answer}"
        )
        return prompted_video_path, answer, confidence_dict, status

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, "", None, f"Error: {e}"


with gr.Blocks(title="NExT-QA VideoQA Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# NExT-QA VideoQA Demo")
    state = gr.State(value=None)

    with gr.Row():
        # ── Left: inputs (always visible) ─────────────────────────────
        with gr.Column(scale=1, min_width=280):
            frame_input = gr.Files(label="Video Frames (images)", file_types=["image"], height=220)
            question_input = gr.Textbox(label="Question", placeholder="e.g. What is the person doing?")
            with gr.Row():
                opt_a = gr.Textbox(label="A", scale=1)
                opt_b = gr.Textbox(label="B", scale=1)
                opt_c = gr.Textbox(label="C", scale=1)
                opt_d = gr.Textbox(label="D", scale=1)
                opt_e = gr.Textbox(label="E", scale=1)
            stage1_btn = gr.Button("▶  Stage 1: Segment & Answer", variant="primary", size="lg")

        # ── Right: results area ────────────────────────────────────────
        with gr.Column(scale=3):

            # Stage 1 Accordion (closed until stage1 runs, collapses on intervene)
            with gr.Accordion("Stage 1 — Auto Answer", open=False) as stage1_accordion:
                with gr.Row():
                    input_video_output = gr.Video(label="Input Frames", height=220)
                    mask1_video = gr.Video(label="Mask 1", height=220)
                prompted_video_1 = gr.Video(label="Visually Prompted Frames (Mask 1)", height=220)
                with gr.Row():
                    answer_output = gr.Textbox(label="Predicted Answer", scale=1, interactive=False)
                    confidence_output = gr.Dataframe(
                        headers=["", "Option", "Prob."],
                        label="Confidence (A–E)",
                        interactive=False,
                        scale=2,
                        row_count=(5, "fixed"),
                        col_count=(3, "fixed"),
                    )
                stage1_status = gr.Textbox(label="Status", lines=1, interactive=False)
                stage2_btn = gr.Button(
                    "✏  Not satisfied? Intervene — Generate More Masks",
                    variant="secondary",
                    size="lg",
                )

            # Stage 2 view (hidden until intervene)
            with gr.Column(visible=False) as stage2_view:
                gr.Markdown("### Select a Mask")
                with gr.Row():
                    stage2_input_video = gr.Video(label="Input Frames", height=200)
                    mask_video_0 = gr.Video(label="Mask 1", height=200)
                    mask_video_1 = gr.Video(label="Mask 2", height=200)
                    mask_video_2 = gr.Video(label="Mask 3", height=200)
                stage2_status = gr.Textbox(label="Status", lines=1, interactive=False)
                mask_radio = gr.Radio(
                    choices=SEG_PROMPT_NAMES,
                    label="Which mask would you like to use for QA?",
                    value=SEG_PROMPT_NAMES[0],
                )
                qa_final_btn = gr.Button("▶  Re-run QA with Selected Mask", variant="primary", size="lg")

                gr.Markdown("### Final Answer")
                with gr.Row():
                    final_prompted_video = gr.Video(label="Visually Prompted Frames", height=220, scale=1)
                    with gr.Column(scale=1):
                        final_answer_output = gr.Textbox(label="Predicted Answer", interactive=False)
                        final_confidence_output = gr.Dataframe(
                            headers=["", "Option", "Prob."],
                            label="Confidence (A–E)",
                            interactive=False,
                            row_count=(5, "fixed"),
                            col_count=(3, "fixed"),
                        )
                final_status_output = gr.Textbox(label="Details", lines=4, interactive=False)

    _example_frames = sorted(
        [f"./assets/visualprompting_frames/{f}" for f in os.listdir("./assets/visualprompting_frames") if f.endswith(".jpg")],
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[1]),
    )
    gr.Examples(
        examples=[[
            _example_frames,
            "what does the girls do after kicking their right legs in the middle",
            "dancing",
            "tap paper on table",
            "unroll the gold toy",
            "move in circle",
            "run after it",
        ]],
        inputs=[frame_input, question_input, opt_a, opt_b, opt_c, opt_d, opt_e],
        label="Example",
    )

    stage1_btn.click(
        fn=run_stage1,
        inputs=[frame_input, question_input, opt_a, opt_b, opt_c, opt_d, opt_e],
        outputs=[
            input_video_output, mask1_video, prompted_video_1,
            answer_output, confidence_output, state, stage1_status,
            stage1_accordion,
        ],
    )

    stage2_btn.click(
        fn=run_stage2,
        inputs=[state, question_input, opt_a, opt_b, opt_c, opt_d, opt_e],
        outputs=[
            stage2_input_video, mask_video_0, mask_video_1, mask_video_2,
            state, stage2_status,
            stage1_accordion, stage2_view,
        ],
    )

    qa_final_btn.click(
        fn=run_qa_final,
        inputs=[mask_radio, state, question_input, opt_a, opt_b, opt_c, opt_d, opt_e],
        outputs=[final_prompted_video, final_answer_output, final_confidence_output, final_status_output],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
