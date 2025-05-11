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

from tapnet.utils import sample_grid_points, preprocess_frame
from tapnet.tapir_inference import TapirInference
from dataset.tapir_dataset import TapVidDataset

class TapirQuantize(torch.nn.Module):
    """Wrapper for TAPIR to make it quantization-friendly"""
    
    def __init__(self, tapir_inference):
        super().__init__()
        self.model = tapir_inference
        
    def forward(self, frame, query_feats, hires_query_feats, causal_context):
        """
        Forward pass for quantization
        
        Args:
            frame: Input frame tensor [1, 3, H, W]
            query_feats: Query features [1, N, 256]
            hires_query_feats: High-resolution query features [1, N, 128]
            causal_context: Causal context tensor
            
        Returns:
            Tuple of model outputs
        """
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
        List of calibration input tuples
    """
    calibration_inputs = []
    sample_count = 0
    
    # Process each video sequence
    for video_idx in tqdm(range(min(len(dataset), max_samples)), desc="Preparing calibration data"):
        # Get video data
        video_data = dataset[video_idx]
        video_frames = video_data['rgbs']  # S,H,W,C format
        video_id = video_data['vid']
        
        # Get number of frames in the video
        num_frames = video_frames.shape[0]
        num_frames = min(num_frames, max_frames_per_video)
                
        # Get the first frame for initializing points
        first_frame = video_frames[0].permute(2, 0, 1).unsqueeze(0).to(device)  # Convert to NCHW

       
        # Ensure frame has 3 channels - crucial for OpenCV
        if len(first_frame.shape) != 4 or first_frame.shape[1] != 3:
            print(f"Warning: Frame has invalid shape for OpenCV: {first_frame.shape}")
            continue
            
        # Use trajectory points from the dataset if available
        if 'trajs' in video_data and video_data['trajs'].shape[0] >= num_points:
            # Get coordinates from first frame
            height, width = input_size
            trajs = video_data['trajs'][:num_points, 0].numpy()  # Get first frame positions (normalized)
            
            # Convert normalized [0,1] to pixel coordinates
            query_points = np.zeros((num_points, 2), dtype=np.float32)
            query_points[:, 0] = trajs[:, 0] * width  # x coordinate
            query_points[:, 1] = trajs[:, 1] * height  # y coordinate
            
        else:
            # Sample grid points
            height, width = input_size
            query_points = sample_grid_points(height, width, num_points)
        
        # Initialize tracking with the first frame
        # This is the critical part where the OpenCV error occurs
        model.set_points(first_frame, query_points, False)
        
        # Get initial features and context
        query_feats = model.query_feats.clone()
        hires_query_feats = model.hires_query_feats.clone()
        causal_context = torch.zeros(
            (num_iters, num_mixer_blocks, num_points, 2, 512 + 2048),
            dtype=torch.float32,
            device=device
        )
        
        # Process frames sequentially to maintain temporal context
        for frame_idx in range(num_frames):
            # Convert frame to NCHW format
            frame = video_frames[frame_idx].permute(2, 0, 1).unsqueeze(0).to(device)
            
            # Add frame to calibration inputs
            calibration_inputs.append(
                (frame.clone(), query_feats.clone(), hires_query_feats.clone(), causal_context.clone())
            )
            
            sample_count += 1
            
            # Update tracking state by running inference
            with torch.no_grad():
                # Use the predictor to update causal context
                _, _, causal_context, _, _ = model.predictor(
                    frame, query_feats, hires_query_feats, causal_context
                )
            
        # Check if we've reached the maximum number of samples
        if sample_count >= max_samples:
            print(f"Reached maximum number of samples ({max_samples})")
            break
    
    print(f"Prepared {len(calibration_inputs)} calibration inputs")
    return calibration_inputs


def quantize_model(model_path, dataset_path, output_path, input_size=(480, 480), 
                   num_calibration_samples=100, max_frames_per_video=20):
    """
    Quantize TAPIR model using Intel IPEX
    
    Args:
        model_path: Path to FP32 model
        dataset_path: Path to calibration dataset
        output_path: Path to save quantized model
        input_size: Input resolution (height, width)
        num_calibration_samples: Number of frames to use for calibration
        max_frames_per_video: Maximum frames to use from each video
    """
    # Model parameters
    num_points = 100  # Number of points to track
    num_iters = 4     # Number of PIP iterations 
    num_mixer_blocks = 12  # Number of mixer blocks
    
    # Always use CPU for quantization
    device = torch.device('cpu')
    
    model_fp32 = TapirInference(model_path, input_size, num_iters, device, "FP32")
    
    # Load dataset for calibration
    print(f"Loading dataset from {dataset_path}")
    dataset = TapVidDataset(dataset_path, resize=input_size)
    
    # Prepare calibration inputs
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
    
    # Get example input for model preparation
    example_frame, example_query_feats, example_hires_query_feats, example_causal_context = calibration_inputs[0]
    
    # Wrap model for quantization
    tapir_quantize = TapirQuantize(model_fp32).to(device)
    tapir_quantize.eval()
    
    # Configure static quantization
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
    
    # Calibrate model with real data
    print("Starting calibration process")
    prepared_model.eval()
    
    with torch.no_grad():
        for batch_idx, (cal_frame, cal_query_feats, cal_hires_query_feats, cal_causal_context) in enumerate(
            tqdm(calibration_inputs, desc="Calibrating model")
        ):
            # Run calibration
            prepared_model(cal_frame, cal_query_feats, cal_hires_query_feats, cal_causal_context)
    
    print("Calibration completed successfully")
    
    # Convert model to INT8
    print("Converting model to static INT8")
    quantized_model = convert(prepared_model)
    
    # Save quantized model
    with torch.no_grad():
        traced_model = torch.jit.trace(
            quantized_model,
            (example_frame, example_query_feats, example_hires_query_feats, example_causal_context),
            strict=False,
            check_trace=False
        )
        
        # Freeze the model to optimize it
        traced_model = torch.jit.freeze(traced_model)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    print(f"Saving quantized model to {output_path}")
    traced_model.save(output_path)
    print("Model saved successfully")
    
    print(f"Model quantization completed")
    return quantized_model


def main():
    parser = argparse.ArgumentParser(description='Quantize TAPIR model to INT8 using TapVid dataset')
    
    parser.add_argument('-m', '--model', type=str, required=True,
                        help='Path to input PyTorch model')
    
    parser.add_argument('-d', '--dataset', type=str, default='/workspace/dataset/tapvid_davis/',
                        help='Path to the TapVid dataset directory')
    
    parser.add_argument('--resize', type=int, nargs=2, default=[256, 256],
                        help='Resolution to resize frames to (height width)')
    
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of frames to use for calibration')
    
    parser.add_argument('--max_frames_per_video', type=int, default=20,
                        help='Maximum frames to use from each video')
    
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output path for quantized model (default: input_model_static_int8.pt)')
    
    args = parser.parse_args()
    
    # Set default output path if not specified
    if args.output is None:
        base, ext = os.path.splitext(args.model)
        output_path = f"{base}_static_int8{ext}"
    else:
        output_path = args.output
    
    # Run quantization
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