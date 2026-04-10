import argparse
import os
import sys
import json
from glob import glob
import hashlib
import time
import types
import tempfile
import shutil

# Disable flash attention BEFORE any transformers import
# Create fake module to prevent flash_attn binary from being loaded
os.environ["TRANSFORMERS_NO_FLASH_ATTN_2"] = "1"

# Pre-register fake modules before transformers tries to load them
fake_module = types.ModuleType('flash_attn')
sys.modules['flash_attn'] = fake_module
sys.modules['flash_attn_2_cuda'] = types.ModuleType('flash_attn_2_cuda')
sys.modules['flash_attn.bert_padding'] = types.ModuleType('bert_padding')

# Also disable in transformers utilities before import
import importlib.util

# Mock is_flash_attn_2_available before transformers even loads
_original_find_spec = importlib.util.find_spec
def _patched_find_spec(name, package=None):
    if 'flash_attn' in name:
        return None
    return _original_find_spec(name, package)

importlib.util.find_spec = _patched_find_spec

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import cv2
from tqdm import tqdm

from model.qwen_2_5_vl_sam2 import UniGRConfig, UniGRModel
from utils.utils import DirectResize, get_sparse_indices, dict_to_cuda, preprocess
from model.STOM import STOM
from utils.visual_prompt_generator import blend_image_from_mask, video_blending_keyframes, color_pool, words_shape
from pycocotools import mask as coco_mask



def parse_args():
    parser = argparse.ArgumentParser(description="UniGR Video Chat")
    parser.add_argument("--version", default="/PATH/TO/UniGR-7B", help="Model path")
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument("--num_frames", default=16, type=int)
    parser.add_argument("--num_frames_mllm", default=4, type=int)
    parser.add_argument("--sam_max_frames", default=24, type=int,
                        help="Maximum number of frames used for SAM-based segmentation in visual prompting")
    parser.add_argument("--stom_max_frames", default=48, type=int,
                        help="Maximum number of frames used for STOM/static visual-prompt video creation")
    parser.add_argument("--qa_max_frames", default=16, type=int,
                        help="Maximum number of frames used for QA inference after visual prompting")
    parser.add_argument("--max_pixels", default=384*28*28, type=int)
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--precision", default="bf16", type=str)
    return parser.parse_args()


args = parse_args()
os.makedirs(args.vis_save_path, exist_ok=True)


print("Loading model...")
processor = AutoProcessor.from_pretrained(args.version)
tokenizer = processor.tokenizer
args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[-1]

model_args = {
    "train_mask_decoder": False,
    "seg_token_idx": args.seg_token_idx,
    "sam_pretrained": None,
}

config = UniGRConfig.from_pretrained(args.version, **model_args)
try:
    model = UniGRModel.from_pretrained(
        args.version,
        config=config,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=False,
    )
except (ImportError, RuntimeError) as e:
    print(f"Flash attention 2 failed: {e}")
    print("Falling back to default attention implementation...")
    model = UniGRModel.from_pretrained(
        args.version,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )

model = model.bfloat16().cuda().eval()
transform = DirectResize(args.image_size)
print("Model loaded successfully!")

# Initialize STOM for temporal propagation
print("Loading STOM for temporal propagation...")
try:
    propagator = STOM(device="cuda:0")
    use_stom = True
except Exception as e:
    print(f"Warning: STOM initialization failed: {e}. Visual prompting will use static blending.")
    propagator = None
    use_stom = False


class VideoState:
    def __init__(self):
        self.frames = []
        self.current_frame_idx = 0
        self.original_frames = []
        self.original_video_path = None
        self.processed_video_path = None
    
    def reset(self):
        self.frames = []
        self.current_frame_idx = 0
        self.original_frames = []
        self.original_video_path = None
        self.processed_video_path = None

video_state = VideoState()


def process_video_frames(video_path, max_frames=50):

    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return None
            

        step = max(1, total_frames // max_frames)
        frame_indices = list(range(0, total_frames, step))[:max_frames]
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
        
        cap.release()
        return frames
    except Exception as e:
        print(f"Error processing video: {e}")
        return None

def question_to_segmentation_prompt(question: str) -> str:
    """
    Convert NExT-QA style question to segmentation instruction.
    E.g., "Where is the cat sitting on?" -> "Can you segment the key object mentioned in this question? Question: Where is the cat sitting on?"
    """
    question = question.strip()
    prompt = f"Can you segment the key object mentioned in this question? Question: {question}"
    return prompt


def process_video_with_visual_prompting(video_file, question_text, progress=gr.Progress()):
    """
    End-to-end pipeline: Question -> Segmentation Prompt -> Single-frame Segmentation -> 
    STOM Propagation -> Visual Prompt Creation -> QA Inference
    """
    try:
        if not video_file:
            return None, None, None, "Please upload a video file."
        
        if not question_text or not question_text.strip():
            return None, None, None, "Please enter a question."
        
        progress(0, desc="Loading video...")
        
        # Load and process video
        cap = cv2.VideoCapture(video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            return None, None, None, "Failed to read video frames."
        
        # Get sparse frames for MLLM
        sparse_idxs = get_sparse_indices(total_frames, args.num_frames_mllm)
        
        progress(0.05, desc="Loading frames...")
        
        frames_list = []
        for frm_idx in sparse_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frm_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(frame_rgb)
                frames_list.append(image_pil)
        
        # Select middle frame for segmentation
        key_frame_idx = total_frames // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, key_frame_idx)
        ret, key_frame_bgr = cap.read()
        if not ret:
            return None, None, None, f"Failed to read frame at index {key_frame_idx}"
        
        key_frame_rgb = cv2.cvtColor(key_frame_bgr, cv2.COLOR_BGR2RGB)
        key_frame_pil = Image.fromarray(key_frame_rgb)
        
        progress(0.15, desc="Preparing frames for segmentation...")
        
        # Load a bounded number of frames for SAM2 processing to avoid OOM
        sam_num_frames = min(total_frames, args.sam_max_frames)
        sam_frame_indices = get_sparse_indices(total_frames, sam_num_frames)
        if key_frame_idx not in sam_frame_indices:
            sam_frame_indices.append(key_frame_idx)
            sam_frame_indices = sorted(set(sam_frame_indices))

        # Find nearest sampled frame to the true key frame index
        key_frame_pos_in_sam = min(
            range(len(sam_frame_indices)),
            key=lambda i: abs(sam_frame_indices[i] - key_frame_idx)
        )

        image_list_sam, image_list_np = [], []
        original_size_list = []
        resize_list = []
        
        for frm_idx in sam_frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frm_idx)
            ret, frame = cap.read()
            if not ret:
                break
            
            image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_size_list.append(image_np.shape[:2])
            
            image = transform.apply_image(image_np)
            resize_list.append(image.shape[:2])
            
            image_tensor = preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).cuda()
            image_tensor = image_tensor.bfloat16()
            
            image_list_sam.append(image_tensor)
            image_list_np.append(image_np)
        
        cap.release()
        
        progress(0.25, desc="Rewriting question as segmentation prompt...")
        
        # Step 1: Convert question to segmentation prompt
        seg_prompt = question_to_segmentation_prompt(question_text)
        
        progress(0.35, desc="Preparing model inputs...")
        
        # Step 2: Prepare messages for segmentation
        messages = [
            {"role": "user", "content": [
                {"type": "video", "video": frames_list, "max_pixels": args.max_pixels},
                {"type": "text", "text": seg_prompt}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Sure, [SEG]."}
            ]}
        ]
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        
        inputs = dict_to_cuda(inputs)
        
        progress(0.45, desc="Running segmentation...")
        
        # Step 3: Run segmentation
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask'] if 'attention_mask' in inputs else None
        pixel_values = inputs['pixel_values'].bfloat16() if 'pixel_values' in inputs else None
        pixel_values_videos = inputs['pixel_values_videos'].bfloat16() if 'pixel_values_videos' in inputs else None
        image_grid_thw = inputs['image_grid_thw'] if 'image_grid_thw' in inputs else None
        video_grid_thw = inputs['video_grid_thw'] if 'video_grid_thw' in inputs else None
        second_per_grid_ts = inputs['second_per_grid_ts'] if 'second_per_grid_ts' in inputs else None
        
        image_sam = torch.stack(image_list_sam, dim=1)

        with torch.inference_mode():
            output_ids, pred_masks = model.evaluate(
                input_ids,
                attention_mask,
                pixel_values,
                pixel_values_videos,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                image_sam,
                resize_list,
                original_size_list,
            )
        
        progress(0.60, desc="Processing segmentation results...")
        
        # Step 4: Extract mask from segmentation results
        if len(pred_masks) == 0 or pred_masks[0].shape[0] == 0:
            return None, None, None, "No segmentation mask found. Try a different question."
        
        pred_mask_vid = pred_masks[0].detach().cpu().numpy()
        
        # Get mask near the key frame from sampled SAM frames
        if key_frame_pos_in_sam < pred_mask_vid.shape[0]:
            key_mask = pred_mask_vid[key_frame_pos_in_sam] > 0
        else:
            key_mask = pred_mask_vid[0] > 0
        
        mask_area = np.sum(key_mask)
        if mask_area < 100:
            return None, None, None, f"Segmentation mask too small ({int(mask_area)} pixels). Try a different question."
        
        progress(0.70, desc="Creating visual prompt...")
        
        # Step 5: Create visual prompt overlay on key frame
        color = "red"
        # Use true mask overlay (not bbox outline) for key-frame visualization.
        shape = "mask"
        blended_frame = blend_image_from_mask(key_frame_pil.convert('RGB'), key_mask.astype(np.float32), color, shape)

        # Free large segmentation tensors before STOM/QA to reduce peak VRAM
        del image_sam, image_list_sam, image_list_np
        del input_ids, attention_mask, pixel_values, pixel_values_videos
        del image_grid_thw, video_grid_thw, second_per_grid_ts, output_ids, pred_masks
        if 'inputs' in locals():
            del inputs
        torch.cuda.empty_cache()
        
        # Step 6: Create frames with visual prompt
        blended_frames = None
        propagated_frames = None
        if use_stom and propagator is not None:
            progress(0.75, desc="Propagating visual prompt across frames...")
            try:
                cap = cv2.VideoCapture(video_file)
                prop_num_frames = min(total_frames, args.stom_max_frames)
                prop_frame_indices = get_sparse_indices(total_frames, prop_num_frames)
                if key_frame_idx not in prop_frame_indices:
                    prop_frame_indices.append(key_frame_idx)
                    prop_frame_indices = sorted(set(prop_frame_indices))
                key_frame_pos_in_prop = min(
                    range(len(prop_frame_indices)),
                    key=lambda i: abs(prop_frame_indices[i] - key_frame_idx)
                )
                is_key_frame = [i == key_frame_pos_in_prop for i in range(len(prop_frame_indices))]
                
                all_frames = []
                all_masks = []
                for i, frame_idx in enumerate(prop_frame_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        all_frames.append(Image.fromarray(frame_rgb))
                        
                        if i == key_frame_pos_in_prop:
                            all_masks.append(key_mask.astype(np.float32))
                        else:
                            all_masks.append(np.zeros_like(key_mask, dtype=np.float32))
                
                cap.release()
                
                blended_frames, vip_img = video_blending_keyframes(
                    all_frames, all_masks, is_key_frame, color, shape, return_vip_img=True
                )
                
                if vip_img is not None and np.any(np.array(vip_img)[:, :, 3] > 0):
                    propagated_frames = propagator.propagate_in_video(
                        all_frames, vip_img, is_key_frame.index(True), shape=shape
                    )
                else:
                    propagated_frames = blended_frames
                
            except Exception as e:
                print(f"STOM propagation failed: {e}. Using static blending.")
                cap = cv2.VideoCapture(video_file)
                prop_num_frames = min(total_frames, args.stom_max_frames)
                prop_frame_indices = get_sparse_indices(total_frames, prop_num_frames)
                if key_frame_idx not in prop_frame_indices:
                    prop_frame_indices.append(key_frame_idx)
                    prop_frame_indices = sorted(set(prop_frame_indices))
                key_frame_pos_in_prop = min(
                    range(len(prop_frame_indices)),
                    key=lambda i: abs(prop_frame_indices[i] - key_frame_idx)
                )
                propagated_frames = []
                for i, frame_idx in enumerate(prop_frame_indices):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_pil = Image.fromarray(frame_rgb)
                        if i == key_frame_pos_in_prop:
                            propagated_frames.append(blended_frame)
                        else:
                            propagated_frames.append(frame_pil)
                cap.release()
        else:
            cap = cv2.VideoCapture(video_file)
            prop_num_frames = min(total_frames, args.stom_max_frames)
            prop_frame_indices = get_sparse_indices(total_frames, prop_num_frames)
            if key_frame_idx not in prop_frame_indices:
                prop_frame_indices.append(key_frame_idx)
                prop_frame_indices = sorted(set(prop_frame_indices))
            key_frame_pos_in_prop = min(
                range(len(prop_frame_indices)),
                key=lambda i: abs(prop_frame_indices[i] - key_frame_idx)
            )
            propagated_frames = []
            for i, frame_idx in enumerate(prop_frame_indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_pil = Image.fromarray(frame_rgb)
                    if i == key_frame_pos_in_prop:
                        propagated_frames.append(blended_frame)
                    else:
                        propagated_frames.append(frame_pil)
            cap.release()

        # Decide which frame sequence to use for each output video
        static_vip_frames = blended_frames if blended_frames is not None else propagated_frames
        
        progress(0.80, desc="Creating output video...")
        
        # Static visual prompt video (key-frame overlay only)
        vip_static_basename = f"vip_static_{int(time.time() * 1000)}_{hashlib.md5((video_file + question_text).encode('utf-8')).hexdigest()[:8]}.mp4"
        vip_static_path = os.path.join(args.vis_save_path, vip_static_basename)
        vip_static_path = create_video_from_frames(static_vip_frames, vip_static_path)
        if not vip_static_path:
            return None, None, blended_frame, None, "Failed to create static visual prompt video file."
        vip_static_path = os.path.abspath(vip_static_path)

        # STOM propagation video (if available, otherwise fall back to static)
        vip_stom_path = None
        if use_stom and propagator is not None and propagated_frames is not None:
            vip_stom_basename = f"vip_stom_{int(time.time() * 1000)}_{hashlib.md5((video_file + question_text).encode('utf-8')).hexdigest()[:8]}.mp4"
            vip_stom_path = os.path.join(args.vis_save_path, vip_stom_basename)
            vip_stom_path = create_video_from_frames(propagated_frames, vip_stom_path)
            if not vip_stom_path:
                vip_stom_path = vip_static_path
            else:
                vip_stom_path = os.path.abspath(vip_stom_path)
        else:
            vip_stom_path = vip_static_path
        
        torch.cuda.empty_cache()
        
        progress(0.90, desc="Running QA inference...")
        
        # Step 7: Run QA inference with visual prompt
        qa_prompt = f"Look at the marked region and then answer the question. {question_text}"

        # Bound QA frames to avoid large video tensors on GPU
        qa_num_frames = min(len(propagated_frames), args.qa_max_frames)
        qa_frame_indices = get_sparse_indices(len(propagated_frames), qa_num_frames)
        qa_frames = [propagated_frames[i] for i in qa_frame_indices]
        
        qa_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": qa_frames, "max_pixels": args.max_pixels},
                    {"type": "text", "text": qa_prompt},
                ],
            }
        ]
        
        qa_text = processor.apply_chat_template(qa_messages, tokenize=False, add_generation_prompt=True)
        qa_image_inputs, qa_video_inputs, qa_video_kwargs = process_vision_info(qa_messages, return_video_kwargs=True)
        qa_inputs = processor(
            text=qa_text,
            images=qa_image_inputs,
            videos=qa_video_inputs,
            padding=False,
            return_tensors="pt",
            **qa_video_kwargs,
        )
        qa_inputs = qa_inputs.to("cuda")
        
        with torch.inference_mode():
            generated_ids = model.generate(
                **qa_inputs,
                max_new_tokens=128,
                do_sample=False,
                num_beams=1,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(qa_inputs.input_ids, generated_ids)
            ]
            qa_output = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        
        torch.cuda.empty_cache()
        
        progress(1.0, desc="Complete!")
        
        status_msg = f"""✅ Visual Prompting Complete!

📝 Original Question: {question_text}
🔄 Segmentation Prompt: {seg_prompt}
🎯 Key Frame Index: {key_frame_idx}/{total_frames}
📊 Mask Area: {int(mask_area)} pixels
🎬 Propagation: {'STOM (with temporal coherence)' if use_stom and propagator else 'Static (key frame only)'}

🤖 Answer: {qa_output}"""
        
        return vip_static_path, vip_stom_path, blended_frame, qa_output, status_msg
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, None, f"Error during visual prompting: {str(e)}"

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

        # Step 1: write video using OpenCV (mp4v or fallback) so we always get a file
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not out.isOpened():
            # Fallback to MJPG in case mp4v is not available
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

        # Step 2: if ffmpeg is available, re-encode to a browser-friendly format
        try:
            import shutil
            import subprocess

            ffmpeg_path = shutil.which("ffmpeg")
            if ffmpeg_path is not None:
                base_path, _ = os.path.splitext(output_path)

                # 2-1) Try H.264 (libx264) mp4 first
                h264_target = base_path + "_h264.mp4"
                h264_cmd = [
                    ffmpeg_path,
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    output_path,
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    h264_target,
                ]

                try:
                    subprocess.run(h264_cmd, check=True)
                    output_path = h264_target
                except Exception:
                    # 2-2) If libx264 is unavailable, fall back to VP9 WebM (libvpx-vp9)
                    vp9_target = base_path + "_vp9.webm"
                    vp9_cmd = [
                        ffmpeg_path,
                        "-y",
                        "-loglevel",
                        "error",
                        "-i",
                        output_path,
                        "-c:v",
                        "libvpx-vp9",
                        "-b:v",
                        "0",
                        "-crf",
                        "30",
                        vp9_target,
                    ]

                    try:
                        subprocess.run(vp9_cmd, check=True)
                        output_path = vp9_target
                    except Exception as e2:
                        print(f"Warning: ffmpeg VP9 re-encode failed or unavailable: {e2}")
        except Exception as e:
            print(f"Warning: ffmpeg re-encode failed or unavailable: {e}")

        return output_path
    except Exception as e:
        print(f"Error creating video: {e}")
        return None


def load_video(video_file):

    if video_file is None:
        return (
            gr.update(value=None),
            "Please upload a video file.", 
            gr.update(maximum=0, value=0, visible=False), 
            gr.update(value=None),
            gr.update(value=None)
        )
    
    try:
        frames = process_video_frames(video_file, args.num_frames)
        if frames is None or len(frames) == 0:
            return (
                gr.update(value=None),
                "Failed to process video.", 
                gr.update(maximum=0, value=0, visible=False), 
                gr.update(value=None),
                gr.update(value=None)
            )
        

        video_state.original_frames = frames.copy()
        video_state.frames = frames.copy()
        video_state.current_frame_idx = 0
        video_state.original_video_path = video_file
        video_state.processed_video_path = None
        
        first_frame = frames[0]
        if first_frame.mode != 'RGB':
            first_frame = first_frame.convert('RGB')
        
        editor_value = {
            "background": first_frame,
            "layers": [],
            "composite": None
        }
        
        return (
            editor_value,
            f"Video loaded! {len(frames)} frames available. You can now select frames and draw on them.",
            gr.update(maximum=len(frames)-1, value=0, visible=True),
            gr.update(value=video_file),
            gr.update(value=None)
        )
    
    except Exception as e:
        return (
            gr.update(value=None),
            f"Error loading video: {str(e)}", 
            gr.update(maximum=0, value=0, visible=False), 
            gr.update(value=None),
            gr.update(value=None)
        )

def update_frame(frame_idx):

    if not video_state.original_frames or frame_idx >= len(video_state.original_frames):
        return gr.update(value=None), "No frames available"
    
    video_state.current_frame_idx = int(frame_idx)
    current_frame = video_state.original_frames[video_state.current_frame_idx]
    
    if current_frame.mode != 'RGB':
        current_frame = current_frame.convert('RGB')
    
    status_info = f"Frame {video_state.current_frame_idx + 1}/{len(video_state.original_frames)} selected"
    
    editor_value = {
        "background": current_frame,
        "layers": [],
        "composite": None
    }
    
    return editor_value, status_info

def extract_drawing_from_editor(editor_data):

    try:
        if editor_data is None:
            return None, "No drawing data"
        
        if isinstance(editor_data, dict):
            if 'composite' in editor_data and editor_data['composite'] is not None:
                return editor_data['composite'], None
            elif 'background' in editor_data and editor_data['background'] is not None:
                return editor_data['background'], None
            else:
                return None, "No valid image found in editor data"
        elif isinstance(editor_data, Image.Image):
            return editor_data, None
        
        return None, f"Unsupported data type: {type(editor_data)}"
        
    except Exception as e:
        return None, f"Error extracting drawing: {str(e)}"

def process_video_qa(video_file, question_text, drawing_data=None):

    try:
        if not question_text or not question_text.strip():
            return gr.update(value=None), "Please enter a question."
        
        if video_file is None:
            return gr.update(value=None), "Please upload a video file."
        

        if not video_state.original_frames:
            frames = process_video_frames(video_file, args.num_frames)
            if frames is None or len(frames) == 0:
                return gr.update(value=None), "Failed to process video file."
            video_state.original_frames = frames.copy()
            video_state.frames = frames.copy()
        
        frames_to_use = video_state.original_frames.copy()
        response_prefix = ""
        processed_video_path = None
        

        has_drawing = False
        if drawing_data is not None:
            drawn_image, error_msg = extract_drawing_from_editor(drawing_data)
            if drawn_image is not None:

                current_frame = video_state.original_frames[video_state.current_frame_idx]
                if drawn_image.size != current_frame.size:
                    drawn_image = drawn_image.resize(current_frame.size, Image.Resampling.LANCZOS)
                

                original_array = np.array(current_frame.convert("RGB"))
                drawn_array = np.array(drawn_image.convert("RGB"))
                
                if not np.array_equal(original_array, drawn_array):
                    has_drawing = True
                    frames_to_use[video_state.current_frame_idx] = drawn_image.convert("RGB")
                    response_prefix = f"[Analysis with drawing on frame {video_state.current_frame_idx + 1}]\n"
                    

                    processed_video_output = os.path.join(args.vis_save_path, "processed_video.mp4")
                    processed_video_path = create_video_from_frames(frames_to_use, processed_video_output)
                    video_state.processed_video_path = processed_video_path
        

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames_to_use},
                    {"type": "text", "text": question_text},
                ],
            }
        ]
        

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to("cuda")
        

        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=128)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            qa_output = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        
        full_response = f"{response_prefix}Q: {question_text}\nA: {qa_output}"
        torch.cuda.empty_cache()
        
        return (
            gr.update(value=processed_video_path) if processed_video_path else gr.update(value=None),
            full_response
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return gr.update(value=None), f"Error during inference: {str(e)}"


def process_video_segmentation(video_file, query_text, progress=gr.Progress()):

    try:
        if not video_file:
            return None, None, "Please upload a video file."
        
        if not query_text or not query_text.strip():
            return None, None, "Please enter a segmentation query."
        
        progress(0, desc="Loading video...")
        

        cap = cv2.VideoCapture(video_file)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            return None, None, "Failed to read video frames."
        

        sparse_idxs = get_sparse_indices(total_frames, args.num_frames_mllm)
        
        progress(0.1, desc="Processing frames...")
        

        frames_list = []
        for frm_idx in sparse_idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frm_idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_pil = Image.fromarray(frame_rgb)
                frames_list.append(image_pil)
        

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        image_list_sam, image_list_np = [], []
        
        for frm_idx in range(total_frames):
            if frm_idx not in sparse_idxs:
                continue
            ret, frame = cap.read()
            if not ret:
                break
                
            image_np = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_size_list = [image_np.shape[:2]]
            

            image = transform.apply_image(image_np)
            resize_list = [image.shape[:2]]
            
            image_tensor = preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).cuda()
            image_tensor = image_tensor.bfloat16()
            
            image_list_sam.append(image_tensor)
            image_list_np.append(image_np)
        
        cap.release()
        
        progress(0.3, desc="Preparing model inputs...")
        

        ref_query = query_text.strip()
        if ref_query[-1] == '?':
            prompt_template = "{sent} Please output the segmentation mask."
            prompt = prompt_template.format(sent=ref_query)
        else:
            if ref_query and ref_query[0].islower() and ref_query.endswith('.'):
                ref_query = ref_query[:-1]
            prompt_template = "Please segment the {class_name} in this image."
            prompt = prompt_template.format(class_name=ref_query.lower())
        

        messages = [
            {"role": "user", "content": [
                {"type": "video", "video": frames_list, "max_pixels": args.max_pixels},
                {"type": "text", "text": prompt}
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": "Sure, [SEG]."}  # teacher forcing
            ]}
        ]
        
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        
        inputs = dict_to_cuda(inputs)
        
        progress(0.5, desc="Running segmentation...")
        

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask'] if 'attention_mask' in inputs else None
        pixel_values = inputs['pixel_values'].bfloat16() if 'pixel_values' in inputs else None
        pixel_values_videos = inputs['pixel_values_videos'].bfloat16() if 'pixel_values_videos' in inputs else None
        image_grid_thw = inputs['image_grid_thw'] if 'image_grid_thw' in inputs else None
        video_grid_thw = inputs['video_grid_thw'] if 'video_grid_thw' in inputs else None
        second_per_grid_ts = inputs['second_per_grid_ts'] if 'second_per_grid_ts' in inputs else None
        
        image_sam = torch.stack(image_list_sam, dim=1)

        output_ids, pred_masks = model.evaluate(
            input_ids,
            attention_mask,
            pixel_values,
            pixel_values_videos,
            image_grid_thw,
            video_grid_thw,
            second_per_grid_ts,
            image_sam,
            resize_list,
            original_size_list,
        )
        
        progress(0.8, desc="Processing results...")
        
        segmented_frames = []
        mask_frames = []
        
        if len(pred_masks) > 0 and pred_masks[0].shape[0] > 0:
            pred_mask_vid = pred_masks[0]
            color = np.array([255, 0, 0])  
            
            for frame_idx in range(min(total_frames, pred_mask_vid.shape[0])):
                pred_mask = pred_mask_vid.detach().cpu().numpy()[frame_idx]
                pred_mask = pred_mask > 0
                
                mask_vis = np.zeros_like(image_list_np[frame_idx])
                mask_vis[pred_mask] = color
                mask_frames.append(Image.fromarray(mask_vis.astype(np.uint8)))
                
                overlay_img = image_list_np[frame_idx].copy()
                overlay_img[pred_mask] = (
                    image_list_np[frame_idx] * 0.6 + 
                    pred_mask[:, :, None].astype(np.uint8) * color * 0.4
                )[pred_mask]
                
                segmented_frames.append(Image.fromarray(overlay_img.astype(np.uint8)))
        
        progress(0.9, desc="Creating output videos...")
        
        segmented_video_path = None
        mask_video_path = None
        
        if segmented_frames:
            segmented_video_path = os.path.join(args.vis_save_path, "segmented_video.mp4")
            segmented_video_path = create_video_from_frames(segmented_frames, segmented_video_path)
            
            mask_video_path = os.path.join(args.vis_save_path, "mask_video.mp4")
            mask_video_path = create_video_from_frames(mask_frames, mask_video_path)
        
        torch.cuda.empty_cache()
        
        progress(1.0, desc="Complete!")
        
        if segmented_video_path:
            return (
                segmented_video_path,
                mask_video_path,
                f"Segmentation completed! Processed {len(segmented_frames)} frames.\nQuery: {query_text}"
            )
        else:
            return None, None, "No segmentation results found. Please try a different query."
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, f"Error during segmentation: {str(e)}"

qa_examples = [
    ["./assets/cat.mp4", "Look at the marked region and then answer the question. What is it?"]
]

seg_examples = [
    ["./assets/cat.mp4", "Can you segment the place where the cat stands in this video."]
]

vip_examples = [
    ["./assets/cat.mp4", "Where is the cat?"]
]

title = "🎥 Object-centric Video Question Answering with Visual Grounding and Referring"
description = """
<div style="text-align: center; margin-bottom: 10px;">
    <p><strong>Two modes available:</strong></p>
    <p>1. Referring Video QA: Ask questions about video content with optional visual prompts</p>
    <p>2. Video Object Segmentation: Segment objects in video based on text descriptions</p>
</div>
"""


with gr.Blocks(title=title, theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {title}")
    gr.HTML(description)
    
    with gr.Tabs():

        with gr.TabItem("🗣️ Referring Video QA"):
            with gr.Row():

                with gr.Column(scale=2):
                    with gr.Group():
                        gr.Markdown("### 📤 Required Inputs")
                        video_input = gr.File(
                            label="Upload Video File",
                            file_types=[".mp4", ".avi", ".mov"],
                            type="filepath",
                        )
                        question_input = gr.Textbox(
                            lines=3,
                            placeholder="Enter your question about the video...",
                            label="Question",
                        )
                    
                    with gr.Group():
                        gr.Markdown("### 🎨 Optional: Interactive Drawing")
                        load_video_btn = gr.Button(
                            "📹 Load Video Frames", 
                            variant="secondary", 
                            size="sm"
                        )
                        
                        frame_slider = gr.Slider(
                            minimum=0, 
                            maximum=0, 
                            step=1, 
                            value=0,
                            label="Select Frame",
                            visible=False
                        )
                        
                        video_status = gr.Textbox(
                            label="Status",
                            value="Upload a video and click 'Load Video Frames' to enable drawing",
                            interactive=False,
                            lines=2
                        )
                    
                    submit_btn = gr.Button("🚀 Analyze Video", variant="primary", size="lg")
                

                with gr.Column(scale=3):
                    with gr.Group():
                        gr.Markdown("### ✏️ Drawing Canvas")
                        drawing_canvas = gr.ImageEditor(
                            label="Draw on the selected frame (optional)",
                            type='pil',
                            value=None,
                            brush=gr.Brush(
                                default_size=5, 
                                colors=["#FF0000", "#0000FF", "#00FF00", "#FFFF00", "#FF00FF", "#FFA500"]
                            ),
                            interactive=True,
                            height=300,
                            layers=False
                        )

                    with gr.Group():
                        gr.Markdown("### 📺 Video Preview")
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("**Original**")
                                original_video_preview = gr.Video(
                                    label="Original Video",
                                    interactive=False,
                                    height=200
                                )
                            
                            with gr.Column():
                                gr.Markdown("**With Drawing**")
                                processed_video_preview = gr.Video(
                                    label="Processed Video",
                                    interactive=False,
                                    height=200
                                )
                        
                        gr.Markdown("### 🤖 AI Response")
                        output_text = gr.Textbox(
                            lines=8, 
                            label="Response", 
                            interactive=False,
                            placeholder="AI response will appear here..."
                        )
            

            with gr.Group():
                gr.Markdown("### 💡 Examples")
                gr.Examples(
                    examples=qa_examples,
                    inputs=[video_input, question_input],
                )
            

            load_video_btn.click(
                fn=load_video,
                inputs=[video_input],
                outputs=[drawing_canvas, video_status, frame_slider, original_video_preview, processed_video_preview]
            )
            
            frame_slider.change(
                fn=update_frame,
                inputs=[frame_slider],
                outputs=[drawing_canvas, video_status]
            )
            
            submit_btn.click(
                fn=process_video_qa,
                inputs=[video_input, question_input, drawing_canvas],
                outputs=[processed_video_preview, output_text]
            )
        

        with gr.TabItem("🎯 Video Segmentation"):
            with gr.Row():

                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### 📤 Inputs")
                        seg_video_input = gr.File(
                            label="Upload Video File",
                            file_types=[".mp4", ".avi", ".mov"],
                            type="filepath",
                        )
                        seg_query_input = gr.Textbox(
                            lines=3,
                            placeholder="Enter object description to segment (e.g., 'red apple', 'person walking', 'car on the road')...",
                            label="Query",
                        )
                        seg_submit_btn = gr.Button("🎯 Start Segmentation", variant="primary", size="lg")
                

                with gr.Column(scale=2):
                    with gr.Group():
                        gr.Markdown("### 🎬 Results")
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("**Segmented Video**")
                                seg_result_video = gr.Video(
                                    label="Segmentation Overlay",
                                    interactive=False,
                                    height=250
                                )
                            
                            with gr.Column():
                                gr.Markdown("**Mask Video**")
                                mask_result_video = gr.Video(
                                    label="Segmentation Mask",
                                    interactive=False,
                                    height=250
                                )
                        
                        seg_status_text = gr.Textbox(
                            lines=4,
                            label="Status",
                            interactive=False,
                            placeholder="Upload a video and enter a query to start segmentation..."
                        )
            

            with gr.Group():
                gr.Markdown("### 💡 Examples")
                gr.Examples(
                    examples=seg_examples,
                    inputs=[seg_video_input, seg_query_input],
                )
            

            seg_submit_btn.click(
                fn=process_video_segmentation,
                inputs=[seg_video_input, seg_query_input],
                outputs=[seg_result_video, mask_result_video, seg_status_text]
            )

        with gr.TabItem("✨ Visual Prompting for QA"):
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Group():
                        gr.Markdown("### 📤 Inputs")
                        vip_video_input = gr.File(
                            label="Upload Video File",
                            file_types=[".mp4", ".avi", ".mov"],
                            type="filepath",
                        )
                        vip_question_input = gr.Textbox(
                            lines=3,
                            placeholder="Enter a question (e.g., 'Where is the cat?', 'What is the person doing?')...",
                            label="Question",
                        )
                        vip_submit_btn = gr.Button("🚀 Generate & Answer", variant="primary", size="lg")
                
                with gr.Column(scale=2):
                    with gr.Group():
                        gr.Markdown("### 🎬 Visual Prompt Videos")
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("**Static Visual Prompt (key-frame overlay)**")
                                vip_result_video = gr.Video(
                                    label="Static Visual Prompt Video",
                                    interactive=False,
                                    height=260
                                )
                            with gr.Column():
                                gr.Markdown("**STOM Propagation Video**")
                                vip_stom_video = gr.Video(
                                    label="STOM Propagated Video",
                                    interactive=False,
                                    height=260
                                )
                        
                        gr.Markdown("### 🖼️ Key Frame with Overlay")
                        vip_key_frame = gr.Image(
                            label="Key Frame Segmentation Overlay",
                            interactive=False,
                            height=250
                        )
                    
                    with gr.Group():
                        gr.Markdown("### 💬 QA Result")
                        vip_answer_output = gr.Textbox(
                            lines=2,
                            label="Answer",
                            interactive=False,
                            placeholder="AI answer will appear here..."
                        )
                        
                        vip_status_text = gr.Textbox(
                            lines=6,
                            label="Process Details",
                            interactive=False,
                            placeholder="Status will appear here..."
                        )

            with gr.Group():
                gr.Markdown("### 💡 Examples")
                gr.Examples(
                    examples=vip_examples,
                    inputs=[vip_video_input, vip_question_input],
                )
            
            vip_submit_btn.click(
                fn=process_video_with_visual_prompting,
                inputs=[vip_video_input, vip_question_input],
                outputs=[vip_result_video, vip_stom_video, vip_key_frame, vip_answer_output, vip_status_text]
            )


if __name__ == "__main__":
    demo.queue(max_size=10)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
