#!/usr/bin/env python
# encoding: utf-8

import os
import argparse
import torch
import numpy as np
import cv2
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.quantization import prepare, convert
from tapnet.tapir_model import TAPIR

class TapirQuantize(torch.nn.Module):
    """Wrapper for TAPIR to make it quantization-friendly"""
    
    def __init__(self, tapir_model):
        super().__init__()
        self.model = tapir_model
        
    def forward(self, frame, query_feats, hires_query_feats, causal_context):
        # Extract feature grids - only uses frame
        feature_grid, hires_feats_grid = self.model.get_feature_grids(frame)
        
        # Estimate trajectories - uses all inputs
        tracks, occlusions, expected_dist, new_causal_context = self.model.estimate_trajectories(
            feature_grid=feature_grid,
            hires_feats_grid=hires_feats_grid,
            query_feats=query_feats,
            hires_query_feats=hires_query_feats,
            causal_context=causal_context
        )
        
        visibles = (1 - torch.sigmoid(occlusions)) * (1 - torch.sigmoid(expected_dist)) > 0.5
        
        return tracks, visibles, new_causal_context, feature_grid, hires_feats_grid

def quantize_model(model_path: str, output_path: str):
    input_size = 480
    num_points = 100
    num_pips_iter = 4  # Use 1 for faster inference, and 4 for better results
    num_mixer_blocks = 12  # Default value from TAPIR model

    # Always use CPU for quantization
    device = torch.device('cpu')
    
    # Initialize model
    model_fp32 = TAPIR(
        num_pips_iter=num_pips_iter,
        initial_resolution=(input_size, input_size),
        device=device,
        use_casual_conv=True
    )
    model_fp32.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
    model_fp32.eval()
    
    # Ensure model is on CPU for quantization
    model_fp32 = model_fp32.to(device)
    
    # Wrap the model to ensure consistent inputs/outputs for quantization
    wrapped_model = TapirQuantize(model_fp32)

    # Prepare example inputs for tracing with correct shapes
    frame = torch.rand(1, 3, input_size, input_size, device=device)  # Batch, channels, height, width
    query_feats = torch.zeros((1, num_points, 256), dtype=torch.float32, device=device)  # Batch, num_queries, channels
    hires_query_feats = torch.zeros((1, num_points, 128), dtype=torch.float32, device=device)  # Batch, num_queries, channels
    
    # Create causal_context with the correct shape
    # Shape should be (num_pips_iter, num_mixer_blocks, num_points, 2, 512 + 2048)
    causal_context = torch.zeros(
        (num_pips_iter, num_mixer_blocks, num_points, 2, 512 + 2048), 
        dtype=torch.float32, device=device
    )

    # Print shapes for debugging
    print(f"Frame shape: {frame.shape}")
    print(f"Query features shape: {query_feats.shape}")
    print(f"High-res query features shape: {hires_query_feats.shape}")
    print(f"Causal context shape: {causal_context.shape}")
    print(f"All tensors on device: {device}")

    # Use QConfigMapping for dynamic quantization
    qconfig_mapping = ipex.quantization.default_dynamic_qconfig_mapping

    # Prepare model for quantization
    print("Preparing model for quantization...")
    prepared_model = prepare(wrapped_model, qconfig_mapping, 
                             example_inputs=(frame, query_feats, hires_query_feats, causal_context), 
                             inplace=False)

    # Convert to quantized model
    print("Converting model to INT8...")
    quantized_model = convert(prepared_model)
    
    # Trace the quantized model
    with torch.no_grad():
        try:
            print("Starting model tracing...")
            # First try with small batch for debugging
            output = quantized_model(frame, query_feats, hires_query_feats, causal_context)
            print(f"Model ran successfully with output shapes: {[out.shape for out in output]}")
            
            # Then trace the model
            traced_model = torch.jit.trace(
                quantized_model, 
                (frame, query_feats, hires_query_feats, causal_context),
                check_trace=False
            )
            traced_model = torch.jit.freeze(traced_model)
            print("Model tracing completed successfully")
        except Exception as e:
            print(f"Error during tracing: {type(e).__name__}: {e}")
            print("Attempting to script the model instead...")
            try:
                traced_model = torch.jit.script(quantized_model)
                print("Model scripting completed successfully")
            except Exception as e2:
                print(f"Error during scripting: {type(e2).__name__}: {e2}")
                raise RuntimeError("Could not trace or script the model")

    # Save the quantized model
    print(f'Saving quantized model to {output_path}')
    traced_model.save(output_path)
    
    # Add additional metadata to the saved model for device handling
    print("Model quantized successfully on CPU. When using on GPU, all inputs must be on the same device.")
    return traced_model

def main():
    parser = argparse.ArgumentParser(description='Quantize TAPIR model to INT8 using Intel IPEX')
    parser.add_argument('-m', '--model', type=str, required=True, help='Path to input PyTorch model')
    args = parser.parse_args()

    # Prepare output path
    base, ext = os.path.splitext(args.model)
    output_path = f"{base}_int8{ext}"
    
    # Quantize the model
    quantize_model(args.model, output_path)

if __name__ == '__main__':
    main()