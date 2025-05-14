#!/usr/bin/env python
# encoding: utf-8

import os
import sys
import argparse
import torch
import numpy as np
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.quantization import prepare, convert
from tqdm import tqdm
import openvino as ov
import nncf
from nncf.parameters import ModelType
from nncf.quantization import quantize

from tapnet.utils import sample_grid_points, preprocess_frame
from tapnet.tapir_inference import TapirInference
from dataset.tapir_dataset import TapVidDataset

class TapirQuantize(torch.nn.Module):
    """Wrapper for TAPIR to make it quantization-friendly"""
    
    def __init__(self, tapir_inference):
        super().__init__()
        self.model = tapir_inference
        
    def forward(self, frame, query_feats, hires_query_feats, causal_context):
        tracks, visibles, new_causal_context, feature_grid, hires_feats_grid = self.model.predictor(
            frame, query_feats, hires_query_feats, causal_context
        )
        return tracks, visibles, new_causal_context, feature_grid, hires_feats_grid

def create_calibration_inputs(model, dataset, device, num_points, num_iters, num_mixer_blocks, 
                             input_size, max_samples=100, max_frames_per_video=20):
    """
    Create calibration inputs from the dataset
    
    Args:
        model: TapirInference model
        dataset: Dataset to use for calibration
        device: Device to use for tensors
        num_points: Number of tracking points
        num_iters: Number of iterations
        num_mixer_blocks: Number of mixer blocks
        input_size: Input resolution (height, width)
        max_samples: Maximum number of samples to use
        max_frames_per_video: Maximum frames to use from each video
        
    Returns:
        List of calibration input dictionaries
    """
    calibration_inputs = []
    sample_count = 0
    
    for video_idx in tqdm(range(min(len(dataset), max_samples)), desc="Preparing calibration data"):
        video_data = dataset[video_idx]
        video_frames = video_data['rgbs']
        video_id = video_data['vid']
        
        num_frames = video_frames.shape[0]
        num_frames = min(num_frames, max_frames_per_video)
                
        first_frame = video_frames[0].permute(2, 0, 1).unsqueeze(0).to(device)
       
        if len(first_frame.shape) != 4 or first_frame.shape[1] != 3:
            print(f"Warning: Frame has invalid shape: {first_frame.shape}")
            continue
            
        if 'trajs' in video_data and video_data['trajs'].shape[0] >= num_points:
            height, width = input_size
            trajs = video_data['trajs'][:num_points, 0].numpy()
            query_points = np.zeros((num_points, 2), dtype=np.float32)
            query_points[:, 0] = trajs[:, 0] * width
            query_points[:, 1] = trajs[:, 1] * height
        else:
            height, width = input_size
            query_points = sample_grid_points(height, width, num_points)
        
        model.set_points(first_frame, query_points, False)
        
        query_feats = model.query_feats.clone()
        hires_query_feats = model.hires_query_feats.clone()
        causal_context = torch.zeros(
            (num_iters, num_mixer_blocks, num_points, 2, 512 + 2048),
            dtype=torch.float32,
            device=device
        )
        
        for frame_idx in range(num_frames):
            frame = video_frames[frame_idx].permute(2, 0, 1).unsqueeze(0).to(device)
            
            calibration_inputs.append({
                'input_frame': frame.cpu().numpy(),
                'query_feats': query_feats.cpu().numpy(),
                'hires_query_feats': hires_query_feats.cpu().numpy(),
                'causal_state': causal_context.cpu().numpy()
            })
            
            sample_count += 1
            
            with torch.no_grad():
                _, _, causal_context, _, _ = model.predictor(
                    frame, query_feats, hires_query_feats, causal_context
                )
            
            if sample_count >= max_samples:
                print(f"Reached maximum number of samples ({max_samples})")
                break
    
    print(f"Prepared {len(calibration_inputs)} calibration inputs")
    return calibration_inputs

def quantize_model_openvino(model_path, dataset_path, output_path, input_size=(480, 480), 
                           num_calibration_samples=100, max_frames_per_video=20):
    """
    Quantize OpenVINO model (.onnx or .xml) to INT8 using NNCF
    
    Args:
        model_path: Path to .onnx or .xml model
        dataset_path: Path to calibration dataset
        output_path: Directory to save quantized model (.xml and .bin)
        input_size: Input resolution (height, width)
        num_calibration_samples: Number of frames to use for calibration
        max_frames_per_video: Maximum frames to use from each video
    """
    num_points = 100
    num_iters = 4
    num_mixer_blocks = 12
    device = torch.device('xpu')
    
    model_fp32 = TapirInference(model_path, input_size, num_iters, device, "FP32")
    
    print(f"Loading dataset from {dataset_path}")
    dataset = TapVidDataset(dataset_path, resize=input_size)
    
    print(f"Preparing calibration data using up to {num_calibration_samples} frames")
    calibration_inputs = create_calibration_inputs(
        model_fp32,
        dataset,
        device,
        num_points,
        num_iters,
        num_mixer_blocks,
        input_size,
        max_samples=num_calibration_samples,
        max_frames_per_video=max_frames_per_video
    )
    
    if not calibration_inputs:
        print("No valid calibration inputs were prepared. Aborting quantization.")
        return None
    
    core = ov.Core()
    try:
        ov_model = core.read_model(model_path)
    except Exception as e:
        print(f"Failed to load OpenVINO model: {e}")
        return None
    
    def transform_fn(data_item):
        """
        Transform function for NNCF quantization. Returns input data as a dictionary.
        """
        return {
            'input_frame': data_item['input_frame'],
            'query_feats': data_item['query_feats'],
            'hires_query_feats': data_item['hires_query_feats'],
            'causal_state': data_item['causal_state']
        }
    
    quantization_dataset = nncf.Dataset(calibration_inputs, transform_fn)
    
    print(f"Calibration dataset size: {len(calibration_inputs)}")
    print("Starting NNCF quantization")
    try:
        quantized_model = quantize(
            ov_model,
            quantization_dataset,
            model_type=ModelType.TRANSFORMER,
            subset_size=len(calibration_inputs)
        )
    except Exception as e:
        print(f"NNCF quantization failed: {e}")
        return None
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"Saving quantized model to {output_path}")
    ov.serialize(quantized_model, output_path)
    print("Model saved successfully")
    
    return quantized_model

def quantize_model(model_path, dataset_path, output_path, input_size=(480, 480), 
                   num_calibration_samples=100, max_frames_per_video=20):
    """
    Quantize TAPIR model using Intel IPEX or OpenVINO NNCF
    
    Args:
        model_path: Path to FP32 model (.pt, .onnx, or .xml)
        dataset_path: Path to calibration dataset
        output_path: Path to save quantized model
        input_size: Input resolution (height, width)
        num_calibration_samples: Number of frames to use for calibration
        max_frames_per_video: Maximum frames to use from each video
    """

    _, ext = os.path.splitext(model_path)
    ext = ext.lower()
    
    if ext in ('.onnx', '.xml'):
        return quantize_model_openvino(
            model_path,
            dataset_path,
            output_path,
            input_size,
            num_calibration_samples,
            max_frames_per_video
        )
    
    num_points = 100
    num_iters = 4
    num_mixer_blocks = 12
    device = torch.device('cpu')
    
    model_fp32 = TapirInference(model_path, input_size, num_iters, device, "FP32")
    
    print(f"Loading dataset from {dataset_path}")
    dataset = TapVidDataset(dataset_path, resize=input_size)
    
    print(f"Preparing calibration data using up to {num_calibration_samples} frames")
    calibration_inputs = create_calibration_inputs(
        model_fp32,
        dataset,
        device,
        num_points,
        num_iters,
        num_mixer_blocks,
        input_size,
        max_samples=num_calibration_samples,
        max_frames_per_video=max_frames_per_video
    )
    
    if not calibration_inputs:
        print("No valid calibration inputs were prepared. Aborting quantization.")
        return None
    
    example_frame, example_query_feats, example_hires_query_feats, example_causal_context = calibration_inputs[0]
    
    tapir_quantize = TapirQuantize(model_fp32).to(device)
    tapir_quantize.eval()
    
    qconfig_mapping = ipex.quantization.default_static_qconfig_mapping
    
    print("Preparing model for static quantization")
    prepared_model = prepare(
        tapir_quantize,
        qconfig_mapping,
        example_inputs=(
            example_frame,
            example_query_feats,
            example_hires_query_feats,
            example_causal_context
        ),
        inplace=False
    )
    
    print("Starting calibration process")
    prepared_model.eval()
    
    with torch.no_grad():
        for cal_frame, cal_query_feats, cal_hires_query_feats, cal_causal_context in tqdm(
            calibration_inputs, desc="Calibrating model"
        ):
            prepared_model(cal_frame, cal_query_feats, cal_hires_query_feats, cal_causal_context)
    
    print("Calibration completed successfully")
    
    print("Converting model to static INT8")
    quantized_model = convert(prepared_model)
    
    with torch.no_grad():
        traced_model = torch.jit.trace(
            quantized_model,
            (example_frame, example_query_feats, example_hires_query_feats, example_causal_context),
            strict=False,
            check_trace=False
        )
        traced_model = torch.jit.freeze(traced_model)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"Saving quantized model to {output_path}")
    traced_model.save(output_path)
    print("Model saved successfully")
    
    print("Model quantization completed")
    return quantized_model

def main():
    parser = argparse.ArgumentParser(description='Quantize TAPIR model to INT8 using TapVid dataset')
    
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Path to input model (.pt, .onnx, or .xml)')
    
    parser.add_argument('-d', '--dataset', type=str, default='/workspace/dataset/tapvid_davis/',
                        help='Path to the TapVid dataset directory')
    
    parser.add_argument('--resize', type=int, nargs=2, default=[480, 480],
                        help='Resolution to resize frames to (height width)')
    
    parser.add_argument('--num_samples', type=int, default=300,
                        help='Number of frames to use for calibration')
    
    parser.add_argument('--max_frames_per_video', type=int, default=20,
                        help='Maximum frames to use from each video')
    
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output path for quantized model (directory for OpenVINO, file for PyTorch)')
    
    args = parser.parse_args()
    
    _, ext = os.path.splitext(args.model)
    ext = ext.lower()
    
    if args.output is None:
        if ext in ('.onnx', '.xml'):
            output_path = os.path.splitext(args.model)[0] + '_int8.xml'
        else:
            base, ext = os.path.splitext(args.model)
            output_path = f"{base}_int8{ext}"
    else:
        output_path = args.output
    
    quantize_model(
        args.model,
        args.dataset,
        output_path,
        input_size=tuple(args.resize),
        num_calibration_samples=args.num_samples,
        max_frames_per_video=args.max_frames_per_video
    )

if __name__ == '__main__':
    main()