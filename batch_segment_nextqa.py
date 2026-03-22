#!/usr/bin/env python3
"""
Batch segmentation script for NExT-QA validation set
Segments key objects in videos and saves mask PNG sequences for visual prompting
"""

import argparse
import os
import sys
import json
import types
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import cv2
from PIL import Image

# Disable flash attention BEFORE any transformers import
os.environ["TRANSFORMERS_NO_FLASH_ATTN_2"] = "1"

fake_module = types.ModuleType('flash_attn')
sys.modules['flash_attn'] = fake_module
sys.modules['flash_attn_2_cuda'] = types.ModuleType('flash_attn_2_cuda')
sys.modules['flash_attn.bert_padding'] = types.ModuleType('bert_padding')

import importlib.util
_original_find_spec = importlib.util.find_spec
def _patched_find_spec(name, package=None):
    if 'flash_attn' in name:
        return None
    return _original_find_spec(name, package)
importlib.util.find_spec = _patched_find_spec

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from model.qwen_2_5_vl_sam2 import UniGRConfig, UniGRModel
from utils.utils import DirectResize, get_sparse_indices, dict_to_cuda, preprocess
from model.STOM import STOM

# Import NEXTQADataset
from nextqa import NEXTQADataset


def get_uniform_frames(total_frames, num_frames=32):
    """
    Extract num_frames uniformly from total_frames of video.
    Returns actual video frame indices (not frame indices within a list).
    
    Args:
        total_frames: total number of frames in video
        num_frames: number of frames to sample (default 32)
    
    Returns:
        list of actual video frame indices (integers)
    
    Example:
        If total_frames=1000, num_frames=32: returns [0, 31, 62, 93, ...]
    """
    if total_frames <= num_frames:
        return list(range(total_frames))
    
    # Uniform sampling: spread frames evenly across the video
    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    return indices


def get_frames(lst, M):
    """
    Uniformly sample M frames from a list of frames.
    Follows the sampling pattern from GCG/utils/utils.py for consistency.
    
    Args:
        lst: list of frame elements (typically 32 frames from dataset)
        M: number of frames to sample (1, 4, 8, 16, etc)
    
    Returns:
        list of sampled frame elements in order
    """
    frame_num = len(lst)
    
    if frame_num == 32:    
        if M == 16:
            result = [lst[i] for i in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30]]
        elif M == 8:
            result = [lst[i] for i in [2, 6, 10, 14, 18, 22, 26, 30]]
        elif M == 4:
            result = [lst[i] for i in [4, 12, 20, 28]]
        elif M == 1:
            result = [lst[16]]
        else:
            result = lst
    elif frame_num == 16:
        if M == 8:
            result = [lst[i] for i in [0, 2, 4, 6, 8, 10, 12, 15]]
        elif M == 4:
            result = [lst[i] for i in [2, 6, 10, 14]]
        elif M == 1:
            result = [lst[7]]
        else:
            result = lst
    else:
        # For other sizes, use uniform spacing
        if M >= frame_num:
            result = lst
        else:
            indices = list(range(0, frame_num, max(1, frame_num // M)))[:M]
            result = [lst[i] for i in indices]
    
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Batch Segment NExT-QA Validation Set")
    parser.add_argument("--version", default="./checkpoints/UniGR-7B", help="Model path")
    parser.add_argument("--anno_path", default="../nextqa/annotations_mc/val.csv", type=str, help="Annotation CSV path")
    parser.add_argument("--mapper_path", default="../nextqa/map_vid_vidorID.json", type=str, help="Video mapper JSON")
    parser.add_argument("--video_path", default="../nextqa/videos", type=str, help="Video directory")
    parser.add_argument("--frame_path", default="../nextqa/frames_32", type=str, help="Frame directory")
    parser.add_argument("--feature_path", default="../nextqa/vision_features/feats_wo_norm_32.h5", type=str, help="Feature path")
    parser.add_argument("--output_dir", default="./next_qa_segmentation_masks", type=str, help="Output directory for masks")
    parser.add_argument("--num_frames_mllm", default=4, type=int, help="Number of frames for MLLM context")
    parser.add_argument("--sam_max_frames", default=24, type=int, help="Max frames for SAM processing")
    parser.add_argument("--stom_max_frames", default=24, type=int, help="Max frames for STOM propagation (reduced for memory efficiency)")
    parser.add_argument("--image_size", default=1024, type=int, help="Image size for model")
    parser.add_argument("--max_pixels", default=384*28*28, type=int, help="Max pixels for vision input")
    parser.add_argument("--enable_stom", default=True, action="store_true", help="Enable STOM propagation")
    parser.add_argument("--stom_batch_size", default=8, type=int, help="Batch size for STOM processing")
    parser.add_argument("--mask_threshold", default=0.5, type=float, help="Threshold for mask binarization (0-1, default: 0.5)")
    parser.add_argument("--precision", default="bf16", type=str, help="Model precision")
    parser.add_argument("--num_workers", default=0, type=int, help="Number of workers")
    parser.add_argument("--batch_size", default=1, type=int, help="Batch size (always 1 for now)")
    parser.add_argument("--num_videos", default=-1, type=int, help="Number of videos to process (-1 for all)")
    parser.add_argument("--debug", default=False, action="store_true", help="Debug mode")
    return parser.parse_args()


def load_csv(path):
    """Load CSV annotations"""
    import pandas as pd
    file_list = []
    data = pd.read_csv(path)
    columns = data.columns.tolist()
    for index, row in data.iterrows():
        file_list.append({})
        for column in columns:
            file_list[index][column] = row[column]
    return file_list


def load_json(path):
    """Load JSON file"""
    with open(path) as f:
        return json.load(f)


def save_json(data, path):
    """Save JSON file"""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def question_to_segmentation_prompt(question: str) -> str:
    """Convert NExT-QA question to segmentation prompt"""
    question = question.strip()
    # prompt = f"Can you segment the key object mentioned in this question? Question: {question}"
    prompt = f"Segment the main subject (the person or object performing the primary action) in this question: {question}"
    return prompt


def segment_video(
    frame_dir_path,
    question,
    model,
    processor,
    tokenizer,
    transform,
    args,
    propagator=None,
    use_stom=True
):
    """
    Segment key object in a single video using pre-extracted JPG frames
    Returns: masks_dict {frame_idx: binary_mask_array}, key_frame_idx, total_frames or None on error
    """
    try:
        # Load JPG frames from directory
        frame_files = sorted(os.listdir(frame_dir_path))
        if not frame_files:
            return None, None, None, f"No frames found in {frame_dir_path}"
        
        total_frames = len(frame_files)
        
        # Load first frame to get dimensions
        first_frame_path = os.path.join(frame_dir_path, frame_files[0])
        first_frame = Image.open(first_frame_path)
        frame_width, frame_height = first_frame.size
        
        # Get sparse frames for MLLM context
        sparse_idxs = get_sparse_indices(total_frames, args.num_frames_mllm)
        frames_list = []
        
        for frm_idx in sparse_idxs:
            if frm_idx < len(frame_files):
                frame_path = os.path.join(frame_dir_path, frame_files[frm_idx])
                try:
                    frame_img = Image.open(frame_path)
                    frames_list.append(frame_img)
                except Exception as e:
                    print(f"Warning: Failed to load frame {frm_idx}: {e}")
        
        # Extract 32 frames uniformly from the video
        frame_indices_32 = get_uniform_frames(total_frames, 32)
        
        # Select first frame from 32 frames for segmentation
        key_frame_idx = frame_indices_32[0]  # first frame (0th out of 32)
        if key_frame_idx >= len(frame_files):
            return None, None, None, f"Key frame index {key_frame_idx} out of range"
        
        key_frame_path = os.path.join(frame_dir_path, frame_files[key_frame_idx])
        try:
            key_frame_pil = Image.open(key_frame_path)
        except Exception as e:
            return None, None, None, f"Failed to read key frame at {key_frame_idx}: {e}"
        
        # Prepare SAM frames - ONLY KEY FRAME for efficiency
        # SAM processes only the first frame (0th), STOM handles temporal propagation
        sam_frame_indices = [key_frame_idx]
        key_frame_pos_in_sam = 0
        
        image_list_sam = []
        original_size_list = []
        resize_list = []
        
        # Process only the key frame
        image_np = np.array(key_frame_pil.convert('RGB'))
        original_size_list.append(image_np.shape[:2])
        
        image = transform.apply_image(image_np)
        resize_list.append(image.shape[:2])
        
        image_tensor = preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous()).unsqueeze(0).cuda()
        image_tensor = image_tensor.bfloat16()
        image_list_sam.append(image_tensor)
        
        # Create segmentation prompt and prepare model inputs
        seg_prompt = question_to_segmentation_prompt(question)
        
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
        
        # Run segmentation
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask', None)
        pixel_values = inputs.get('pixel_values', None)
        if pixel_values is not None:
            pixel_values = pixel_values.bfloat16()
        pixel_values_videos = inputs.get('pixel_values_videos', None)
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.bfloat16()
        image_grid_thw = inputs.get('image_grid_thw', None)
        video_grid_thw = inputs.get('video_grid_thw', None)
        second_per_grid_ts = inputs.get('second_per_grid_ts', None)
        
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
        
        # Extract key frame mask
        if len(pred_masks) == 0 or pred_masks[0].shape[0] == 0:
            return None, None, None, "No segmentation mask found"
        
        pred_mask_vid = pred_masks[0].detach().cpu().numpy()
        
        if key_frame_pos_in_sam < pred_mask_vid.shape[0]:
            pred_mask = pred_mask_vid[key_frame_pos_in_sam]
        else:
            pred_mask = pred_mask_vid[0]
        
        # Apply threshold for binarization (default: 0.5 instead of 0.0)
        key_mask = pred_mask > args.mask_threshold
        
        # Calculate mask statistics
        mask_area = np.sum(key_mask)
        total_pixels = pred_mask.shape[0] * pred_mask.shape[1]
        mask_ratio = mask_area / total_pixels
        
        if args.debug:
            print(f"  Mask stats - Area: {int(mask_area)}, Ratio: {mask_ratio:.4f}, Threshold: {args.mask_threshold}")
            print(f"  Pred mask range: [{pred_mask.min():.4f}, {pred_mask.max():.4f}]")
        
        if mask_area < 100:
            return None, None, None, f"Mask too small ({int(mask_area)} pixels, ratio: {mask_ratio:.4f})"
        
        # Clean up large tensors
        del image_sam, image_list_sam, input_ids, attention_mask, pixel_values, pixel_values_videos
        del image_grid_thw, video_grid_thw, second_per_grid_ts, output_ids, pred_masks, inputs
        torch.cuda.empty_cache()
        
        # Propagate mask to all frames using STOM
        masks_dict = {}
        all_frame_masks = []
        
        if use_stom and propagator is not None and args.enable_stom:
            try:
                # Propagate mask to all 32 frames uniformly sampled from video
                # frame_indices_32 already computed above
                prop_frame_indices = frame_indices_32  # Use all 32 frames
                
                # Key frame is at index 0 in the 32-frame list
                key_frame_pos_in_prop = 0
                is_key_frame = [i == key_frame_pos_in_prop for i in range(len(prop_frame_indices))]
                
                # Load frames in batches to reduce memory
                all_frames = []
                all_masks = []
                for i, frame_idx in enumerate(prop_frame_indices):
                    if frame_idx < len(frame_files):
                        frame_path = os.path.join(frame_dir_path, frame_files[frame_idx])
                        try:
                            frame_pil = Image.open(frame_path).convert('RGB')
                            frame_np = np.array(frame_pil)
                            
                            # Resize frame if too large to reduce memory
                            frame_h, frame_w = frame_np.shape[:2]
                            if frame_h * frame_w > 1920 * 1080:  # Larger than 1080p
                                scale = (1920 * 1080) / (frame_h * frame_w)
                                new_h, new_w = int(frame_h * (scale ** 0.5)), int(frame_w * (scale ** 0.5))
                                frame_np = cv2.resize(frame_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
                            
                            all_frames.append(Image.fromarray(frame_np))
                            
                            if i == key_frame_pos_in_prop:
                                all_masks.append(key_mask.astype(np.float32))
                            else:
                                all_masks.append(np.zeros_like(key_mask, dtype=np.float32))
                        except Exception as e:
                            print(f"Warning: Failed to load frame {frame_idx} for STOM: {e}")
                
                # Create VIP image for propagation
                from utils.visual_prompt_generator import video_blending_keyframes
                blended_frames, vip_img = video_blending_keyframes(
                    all_frames, all_masks, is_key_frame, "red", "mask", return_vip_img=True
                )
                
                if vip_img is not None and np.any(np.array(vip_img)[:, :, 3] > 0):
                    try:
                        propagated_frames = propagator.propagate_in_video(
                            all_frames, vip_img, is_key_frame.index(True), shape="mask"
                        )
                    except RuntimeError as stom_err:
                        if 'out of memory' in str(stom_err).lower():
                            print(f"  STOM OOM: reducing frames and retrying...")
                            # Fall back to static blending on OOM
                            propagated_frames = blended_frames
                        else:
                            raise
                else:
                    propagated_frames = blended_frames
                
                # Extract masks from propagated frames
                for i, frame_idx in enumerate(prop_frame_indices):
                    if isinstance(propagated_frames[i], Image.Image):
                        frame_array = np.array(propagated_frames[i])
                    else:
                        frame_array = propagated_frames[i]
                    
                    # Extract mask (red channel > 0)
                    if len(frame_array.shape) == 3:
                        mask = (frame_array[:, :, 0] > 128).astype(np.uint8) * 255
                    else:
                        mask = frame_array.astype(np.uint8) * 255
                    
                    masks_dict[frame_idx] = mask
                    all_frame_masks.append(mask)
                # Clean up STOM-related large objects
                del all_frames, all_masks, propagated_frames, blended_frames, vip_img
                torch.cuda.empty_cache()
                print(f"  STOM propagation: propagated to {len(prop_frame_indices)} frames")
                
            except Exception as e:
                print(f"  STOM propagation failed: {e}. Using zero-padding fallback.")
                # Fallback: only key frame has mask, others are zero
                for frame_idx in range(total_frames):
                    if frame_idx == key_frame_idx:
                        masks_dict[frame_idx] = (key_mask.astype(np.uint8) * 255)
                    else:
                        masks_dict[frame_idx] = np.zeros((frame_height, frame_width), dtype=np.uint8)
                    all_frame_masks.append(masks_dict[frame_idx])
        else:
            # No STOM: only key frame has mask (fallback mode)
            # For memory efficiency, only save propagated frame indices
            prop_num_frames = min(total_frames, args.stom_max_frames)
            prop_frame_indices = get_sparse_indices(total_frames, prop_num_frames)
            if key_frame_idx not in prop_frame_indices:
                prop_frame_indices.append(key_frame_idx)
                prop_frame_indices = sorted(set(prop_frame_indices))
            
            for frame_idx in prop_frame_indices:
                if frame_idx == key_frame_idx:
                    masks_dict[frame_idx] = (key_mask.astype(np.uint8) * 255)
                else:
                    masks_dict[frame_idx] = np.zeros((frame_height, frame_width), dtype=np.uint8)
        
        return masks_dict, key_frame_idx, total_frames, None
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, None, str(e)


def batch_process(args):
    """Main batch processing loop"""
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("NExT-QA Batch Segmentation")
    print("=" * 80)
    
    # Load NExT-QA dataset with 32-frame processing
    print(f"\nLoading NExT-QA dataset from {args.anno_path}...")
    dataset = NEXTQADataset(
        anno_path=args.anno_path,
        mapper_path=args.mapper_path,
        video_path=args.video_path,
        frame_path=args.frame_path,
        feature_path=args.feature_path,
        frame_count=32
    )
    
    if args.num_videos > 0:
        # Slice dataset attributes for limited processing
        dataset.video_ids = dataset.video_ids[:args.num_videos]
        dataset.videos = dataset.videos[:args.num_videos]
        dataset.questions = dataset.questions[:args.num_videos]
        dataset.qids = dataset.qids[:args.num_videos]
    
    print(f"Found {len(dataset)} videos to process")
    
    # Initialize model
    print(f"\nLoading model from {args.version}...")
    processor = AutoProcessor.from_pretrained(args.version)
    tokenizer = processor.tokenizer
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[-1]
    
    model_args = {
        "train_mask_decoder": False,
        "seg_token_idx": seg_token_idx,
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
    
    # Initialize STOM
    print("\nInitializing STOM for temporal propagation...")
    try:
        propagator = STOM(device="cuda:0")
        use_stom = True
        print("STOM initialized successfully!")
    except Exception as e:
        print(f"Warning: STOM initialization failed: {e}. Will use static blending.")
        propagator = None
        use_stom = False
    
    # Process videos
    print("\n" + "=" * 80)
    print("Processing videos...")
    print("=" * 80 + "\n")
    
    success_count = 0
    failed_count = 0
    error_log = []
    
    for idx in tqdm(range(len(dataset)), desc="Processing"):
        video_id = str(dataset.video_ids[idx])
        qid = str(dataset.qids[idx])
        question = str(dataset.questions[idx])
        frame_dir_path = str(dataset.frames[idx])
        
        # Verify frame directory exists
        if not os.path.exists(frame_dir_path):
            error_log.append({
                'video_id': video_id,
                'error': f'Frame directory not found: {frame_dir_path}'
            })
            failed_count += 1
            continue
        
        # Create output directory for this video
        video_output_dir = os.path.join(args.output_dir, video_id)
        os.makedirs(video_output_dir, exist_ok=True)
        
        # Check if already processed
        metadata_path = os.path.join(video_output_dir, f"{video_id}_metadata.json")
        if os.path.exists(metadata_path):
            success_count += 1
            continue
        
        # Segment video
        masks_dict, key_frame_idx, total_frames, error_msg = segment_video(
            frame_dir_path,
            question,
            model,
            processor,
            tokenizer,
            transform,
            args,
            propagator=propagator,
            use_stom=use_stom
        )
        
        if error_msg is not None:
            error_log.append({
                'video_id': video_id,
                'qid': qid,
                'error': error_msg
            })
            failed_count += 1
            continue
        
        # Save mask PNG sequences
        try:
            for frame_idx, mask in masks_dict.items():
                mask_filename = f"{video_id}_frame_{frame_idx:06d}_mask.png"
                mask_path = os.path.join(video_output_dir, mask_filename)
                cv2.imwrite(mask_path, mask)
            
            # Save metadata
            metadata = {
                'video_id': video_id,
                'qid': qid,
                'question': question,
                'total_frames': total_frames,
                'key_frame_idx': key_frame_idx,
                'num_masks_saved': len(masks_dict),
                'propagation_method': 'STOM' if use_stom else 'static',
            }
            save_json(metadata, metadata_path)
            
            success_count += 1
            
        except Exception as e:
            error_log.append({
                'video_id': video_id,
                'error': f'Failed to save masks: {str(e)}'
            })
            failed_count += 1
        
        finally:
            # Clean up memory after each video
            if 'masks_dict' in locals():
                del masks_dict
            torch.cuda.empty_cache()
    
    # Print summary
    print("\n" + "=" * 80)
    print("Processing Complete!")
    print("=" * 80)
    print(f"Successful: {success_count}/{len(dataset)}")
    print(f"Failed: {failed_count}/{len(dataset)}")
    print(f"Output directory: {args.output_dir}")
    
    if error_log:
        error_log_path = os.path.join(args.output_dir, "error_log.json")
        save_json(error_log, error_log_path)
        print(f"Error log saved to: {error_log_path}")
    
    return success_count, failed_count


if __name__ == "__main__":
    args = parse_args()
    print(args)
    
    success_count, failed_count = batch_process(args)
    sys.exit(0 if failed_count == 0 else 1)
