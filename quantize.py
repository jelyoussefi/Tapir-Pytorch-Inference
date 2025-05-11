#!/usr/bin/env python
# encoding: utf-8

import os
import argparse
import torch
import numpy as np
import cv2
import intel_extension_for_pytorch as ipex
from intel_extension_for_pytorch.quantization import prepare, convert
import tapnet.utils as utils
from tapnet.tapir_inference import TapirInference

class TapirQuantize(torch.nn.Module):
    """Wrapper for TAPIR to make it quantization-friendly"""
    
    def __init__(self, tapir_inference):
        super().__init__()
        self.model = tapir_inference
        
    def forward(self, frame, query_feats, hires_query_feats, causal_context):
        # Use the predictor from TapirInference for inference
        tracks, visibles, new_causal_context, feature_grid, hires_feats_grid = self.model.predictor(
            frame, query_feats, hires_query_feats, causal_context
        )
        return tracks, visibles, new_causal_context, feature_grid, hires_feats_grid

def quantize_model(model_path: str, output_path: str):
    input_size = 480
    num_points = 100
    num_iters = 4 
    num_mixer_blocks = 12  # Default value from TapirInference

    # Always use CPU for quantization
    device = torch.device('cpu')
    frame = torch.rand(1, 3, input_size, input_size, device=device) 
    query_feats = torch.rand(1, num_points, 256, device=device)
    hires_query_feats = torch.rand(1, num_points, 128, device=device)
    causal_context = torch.rand(num_iters, num_mixer_blocks, num_points, 2, 512 + 2048, device=device)

    model_fp32 = TapirInference(model_path, (input_size, input_size), num_iters, device, "FP32")
    query_points = utils.sample_grid_points(input_size, input_size, num_points)
    model_fp32.set_points(frame, query_points)

    # Wrap the model for quantization
    tapir_quantize = TapirQuantize(model_fp32).to(device)

    # Use QConfigMapping for dynamic quantization
    qconfig_mapping = ipex.quantization.default_dynamic_qconfig_mapping

    # Prepare model for quantization
    print("Preparing model for quantization...")
    prepared_model = prepare(tapir_quantize, qconfig_mapping, example_inputs=(frame, query_feats, hires_query_feats, causal_context), inplace=False)

    # Convert to quantized model
    print("Converting model to INT8...")
    quantized_model = convert(prepared_model)

    # Trace the quantized model
    with torch.no_grad():
        try:
            print("Starting model tracing...")
            # Trace with all required inputs
            traced_model = torch.jit.trace(
                quantized_model,
                (frame, query_feats, hires_query_feats, causal_context),
                strict=False,
                check_trace=False
            )
            traced_model = torch.jit.freeze(traced_model)
            print("Model tracing completed successfully")
        except Exception as e:
            raise RuntimeError(f"Could not trace or script the model: {e}")

    # Save the quantized model
    print(f'Saving quantized model to {output_path}')
    traced_model.save(output_path)
    
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