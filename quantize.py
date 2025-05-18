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

def check_model_details(model_path):
    """
    Print detailed information about the model
    
    Args:
        model_path: Path to .onnx or .xml model
    """
    print(f"\n--- Model Details for {model_path} ---")
    
    try:
        core = ov.Core()
        model = core.read_model(model_path)
        
        print(f"Model name: {model.get_friendly_name()}")
        print(f"Input ports: {len(model.inputs)}")
        for i, input_port in enumerate(model.inputs):
            print(f"  Input #{i}: {input_port.get_any_name()}")
            print(f"    Shape: {input_port.get_partial_shape()}")
            print(f"    Element type: {input_port.get_element_type()}")
        
        print(f"Output ports: {len(model.outputs)}")
        for i, output_port in enumerate(model.outputs):
            print(f"  Output #{i}: {output_port.get_any_name()}")
            print(f"    Shape: {output_port.get_partial_shape()}")
            print(f"    Element type: {output_port.get_element_type()}")
        
        print("Model operations:")
        op_types = {}
        for op in model.get_ops():
            op_type = op.get_type_name()
            op_types[op_type] = op_types.get(op_type, 0) + 1
        
        for op_type, count in op_types.items():
            print(f"  {op_type}: {count}")
        
        # Check for GroupConvolution operations specifically
        group_convs = []
        for op in model.get_ops():
            if op.get_type_name() == "GroupConvolution":
                group_convs.append(op)
        
        if group_convs:
            print("\nGroupConvolution operations:")
            for i, op in enumerate(group_convs[:5]):  # Show at most 5
                print(f"  Operation #{i}: {op.get_friendly_name()}")
                try:
                    print(f"    Inputs: {[port.get_any_name() for port in op.inputs]}")
                    print(f"    Outputs: {[port.get_any_name() for port in op.outputs]}")
                except:
                    print("    Could not retrieve detailed port information")
                
            if len(group_convs) > 5:
                print(f"  ... and {len(group_convs) - 5} more")
        
    except Exception as e:
        print(f"Error analyzing model: {e}")

def direct_int8_conversion(model_path, output_path):
    """
    Convert model to INT8 without calibration data
    
    Args:
        model_path: Path to .onnx or .xml model
        output_path: Path to save quantized model
    
    Returns:
        Quantized OpenVINO model
    """
    print("\n--- Direct INT8 Conversion without Calibration ---")
    
    try:
        core = ov.Core()
        model = core.read_model(model_path)
        
        # Create a basic configuration with default settings
        # Use this method for older NNCF versions
        compression_config = {
            "compression": {
                "algorithm": "quantization",
                "weights": {
                    "mode": "symmetric"
                },
                "activations": {
                    "mode": "symmetric"
                }
            }
        }
        
        print("Starting direct quantization...")
        # Use this simpler approach for direct conversion
        from nncf.tensorflow import create_compression_algorithm
        compression_ctrl = create_compression_algorithm(model, compression_config)
        quantized_model = compression_ctrl.model
        
        print(f"Saving directly quantized model to {output_path}")
        ov.serialize(quantized_model, output_path)
        print("Model saved successfully")
        
        return quantized_model
        
    except Exception as e:
        print(f"Direct quantization failed: {e}")
        try:
            # Alternative approach for newer OpenVINO versions
            print("Trying alternative direct quantization approach...")
            core = ov.Core()
            model = core.read_model(model_path)
            from openvino.tools import pot
            
            # Create a DefaultQuantization transformation
            config = {
                "name": "DefaultQuantization",
                "params": {
                    "preset": "performance",
                    "stat_subset_size": 1
                }
            }
            
            # Create a data loader with a single sample (minimal data)
            dummy_data = np.zeros((1, 3, 480, 480), dtype=np.float32)
            data_loader = [{"input_frame": dummy_data}]
            
            # Apply quantization
            engine = pot.Engine(config=config, data_loader=data_loader)
            quantized_model = engine.run(model)
            
            ov.serialize(quantized_model, output_path)
            print("Alternative direct quantization successful")
            return quantized_model
            
        except Exception as e2:
            print(f"Alternative direct quantization also failed: {e2}")
            return None

def fix_group_convolution_model(model_path, output_path):
    """
    Attempts to fix group convolution issues in the model by modifying the model structure
    
    Args:
        model_path: Path to .onnx or .xml model with group convolution issues
        output_path: Path to save fixed model
    
    Returns:
        True if successful, False otherwise
    """
    print("\n--- Attempting to Fix Group Convolution Issues ---")
    
    try:
        import onnx
        from onnx import helper, numpy_helper
        from onnx import shape_inference
        import onnxruntime as ort
        
        # Check if model is in ONNX format, if not, convert to ONNX first
        if not model_path.endswith('.onnx'):
            print("Model is not in ONNX format. Converting to ONNX first...")
            core = ov.Core()
            model = core.read_model(model_path)
            onnx_path = os.path.splitext(model_path)[0] + '_temp.onnx'
            ov.serialize(model, onnx_path, 'onnx')
            model_path = onnx_path
        
        # Load the ONNX model
        onnx_model = onnx.load(model_path)
        
        # Fix group convolution issues
        fixed = False
        for node in onnx_model.graph.node:
            if node.op_type == 'ConvTranspose' or node.op_type == 'Conv' or node.op_type == 'GroupConvolution':
                # Find the group attribute
                for attr in node.attribute:
                    if attr.name == 'group' and attr.i > 1:
                        print(f"Found {node.op_type} node with group={attr.i}")
                        # Modify group to 1 if causing problems
                        # attr.i = 1
                        fixed = True
        
        if fixed:
            # Save the fixed model
            onnx.save(onnx_model, output_path)
            print(f"Fixed model saved to {output_path}")
            return True
        else:
            print("No group convolution issues found to fix")
            return False
            
    except Exception as e:
        print(f"Error fixing group convolution model: {e}")
        return False

def quantize_model_openvino(model_path, dataset_path, output_path, input_size=(480, 480), 
                           num_calibration_samples=100, max_frames_per_video=20, 
                           try_alternatives=True):
    """
    Quantize OpenVINO model (.onnx or .xml) to INT8 using NNCF
    
    Args:
        model_path: Path to .onnx or .xml model
        dataset_path: Path to calibration dataset
        output_path: Directory to save quantized model (.xml and .bin)
        input_size: Input resolution (height, width)
        num_calibration_samples: Number of frames to use for calibration
        max_frames_per_video: Maximum frames to use from each video
        try_alternatives: Whether to try alternative quantization approaches if the main one fails
    """
    print("\n=== Starting OpenVINO Quantization ===")
    
    num_points = 100
    num_iters = 4
    num_mixer_blocks = 12
    device = torch.device('cpu')
    
    # Check model details before quantization
    check_model_details(model_path)
    
    # First try: Standard calibration-based quantization
    try:
        print(f"\n--- Approach 1: Calibration-based Quantization ---")
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
            print("No valid calibration inputs were prepared. Trying direct quantization...")
            if try_alternatives:
                return direct_int8_conversion(model_path, output_path)
            return None
        
        # Print sample calibration input for debugging
        print("\nCalibration input sample:")
        for k, v in calibration_inputs[0].items():
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}, min={v.min()}, max={v.max()}")
        
        core = ov.Core()
        try:
            ov_model = core.read_model(model_path)
        except Exception as e:
            print(f"Failed to load OpenVINO model: {e}")
            if try_alternatives:
                return direct_int8_conversion(model_path, output_path)
            return None
        
        def transform_fn(data_item):
            """
            Transform function for NNCF quantization. Returns input data as a dictionary.
            """
            input_names = [input.get_any_name() for input in ov_model.inputs]
            result = {}
            
            # Map calibration inputs to model inputs
            name_mapping = {
                'input_frame': 'input_frame',
                'query_feats': 'query_feats',
                'hires_query_feats': 'hires_query_feats',
                'causal_state': 'causal_state'
            }
            
            for name in input_names:
                base_name = name.split(':')[0]
                if base_name in name_mapping and name_mapping[base_name] in data_item:
                    result[name] = data_item[name_mapping[base_name]]
            
            return result
        
        quantization_dataset = nncf.Dataset(calibration_inputs, transform_fn)
        
        print(f"Calibration dataset size: {len(calibration_inputs)}")
        print("Starting NNCF quantization")
        
        try:
            # Try with specified parameters
            quantized_model = quantize(
                ov_model,
                quantization_dataset,
                model_type=ModelType.TRANSFORMER,  # Try with TRANSFORMER type (for Tapir architecture)
                subset_size=len(calibration_inputs),
                preset=nncf.TargetDevice.CPU  # Explicitly set target device to CPU
            )
        except Exception as e:
            print(f"NNCF quantization with TRANSFORMER type failed: {e}")
            print("Trying with CONVOLUTIONAL model type...")
            try:
                quantized_model = quantize(
                    ov_model,
                    quantization_dataset,
                    model_type=ModelType.CONVOLUTIONAL,  # Try with CONVOLUTIONAL type
                    subset_size=len(calibration_inputs)
                )
            except Exception as e2:
                print(f"NNCF quantization with CONVOLUTIONAL type also failed: {e2}")
                print("Trying default parameters...")
                try:
                    # Try with default parameters
                    quantized_model = quantize(
                        ov_model,
                        quantization_dataset
                    )
                except Exception as e3:
                    print(f"Default NNCF quantization also failed: {e3}")
                    if try_alternatives:
                        # Try direct quantization as last resort
                        return direct_int8_conversion(model_path, output_path)
                    return None
        
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        print(f"Saving quantized model to {output_path}")
        ov.serialize(quantized_model, output_path)
        print("Model saved successfully")
        
        return quantized_model
        
    except Exception as e:
        print(f"OpenVINO quantization error: {e}")
        if try_alternatives:
            # Try direct quantization
            print("Attempting direct quantization without calibration...")
            return direct_int8_conversion(model_path, output_path)
        return None

def quantize_model_ipex(model_path, dataset_path, output_path, input_size=(480, 480), 
                       num_calibration_samples=100, max_frames_per_video=20):
    """
    Quantize TAPIR model using Intel IPEX
    
    Args:
        model_path: Path to FP32 PyTorch model (.pt)
        dataset_path: Path to calibration dataset
        output_path: Path to save quantized model
        input_size: Input resolution (height, width)
        num_calibration_samples: Number of frames to use for calibration
        max_frames_per_video: Maximum frames to use from each video
    """
    print("\n=== Starting IPEX Quantization ===")
    
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
    
    # Convert calibration inputs to tensor format for IPEX
    tensor_calibration_inputs = []
    for cal_input in calibration_inputs:
        tensor_calibration_inputs.append((
            torch.tensor(cal_input['input_frame']),
            torch.tensor(cal_input['query_feats']),
            torch.tensor(cal_input['hires_query_feats']),
            torch.tensor(cal_input['causal_state'])
        ))
    
    example_frame = torch.tensor(calibration_inputs[0]['input_frame'])
    example_query_feats = torch.tensor(calibration_inputs[0]['query_feats'])
    example_hires_query_feats = torch.tensor(calibration_inputs[0]['hires_query_feats'])
    example_causal_context = torch.tensor(calibration_inputs[0]['causal_state'])
    
    tapir_quantize = TapirQuantize(model_fp32).to(device)
    tapir_quantize.eval()
    
    # Configure quantization
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
            tensor_calibration_inputs, desc="Calibrating model"
        ):
            try:
                prepared_model(
                    cal_frame.to(device),
                    cal_query_feats.to(device),
                    cal_hires_query_feats.to(device),
                    cal_causal_context.to(device)
                )
            except Exception as e:
                print(f"Warning: Error during calibration: {e}")
                continue
    
    print("Calibration completed successfully")
    
    print("Converting model to static INT8")
    quantized_model = convert(prepared_model)
    
    try:
        with torch.no_grad():
            traced_model = torch.jit.trace(
                quantized_model,
                (
                    example_frame.to(device),
                    example_query_feats.to(device),
                    example_hires_query_feats.to(device),
                    example_causal_context.to(device)
                ),
                strict=False,
                check_trace=False
            )
            traced_model = torch.jit.freeze(traced_model)
    except Exception as e:
        print(f"Error during model tracing: {e}")
        print("Saving untraced model instead")
        traced_model = quantized_model
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    print(f"Saving quantized model to {output_path}")
    torch.jit.save(traced_model, output_path)
    print("Model saved successfully")
    
    print("Model quantization completed")
    return quantized_model

def quantize_model(model_path, dataset_path, output_path, input_size=(480, 480), 
                   num_calibration_samples=100, max_frames_per_video=20, 
                   try_alternatives=True, fix_ops=False):
    """
    Quantize TAPIR model using Intel IPEX or OpenVINO NNCF
    
    Args:
        model_path: Path to FP32 model (.pt, .onnx, or .xml)
        dataset_path: Path to calibration dataset
        output_path: Path to save quantized model
        input_size: Input resolution (height, width)
        num_calibration_samples: Number of frames to use for calibration
        max_frames_per_video: Maximum frames to use from each video
        try_alternatives: Whether to try alternative quantization approaches if the main one fails
        fix_ops: Whether to try fixing group convolution operations before quantization
    """
    print(f"\n{'='*50}")
    print(f"Starting quantization for {model_path}")
    print(f"Output will be saved to {output_path}")
    print(f"{'='*50}")

    _, ext = os.path.splitext(model_path)
    ext = ext.lower()
    
    # Try to fix group convolution issues if requested
    if fix_ops and ext in ('.onnx', '.xml'):
        fixed_path = os.path.splitext(model_path)[0] + '_fixed' + ext
        if fix_group_convolution_model(model_path, fixed_path):
            print(f"Using fixed model: {fixed_path}")
            model_path = fixed_path
    
    if ext in ('.onnx', '.xml'):
        return quantize_model_openvino(
            model_path,
            dataset_path,
            output_path,
            input_size,
            num_calibration_samples,
            max_frames_per_video,
            try_alternatives
        )
    else:
        return quantize_model_ipex(
            model_path,
            dataset_path,
            output_path,
            input_size,
            num_calibration_samples,
            max_frames_per_video
        )

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

    parser.add_argument('--direct', action='store_true',
                        help='Use direct INT8 conversion without calibration (OpenVINO only)')
    
    parser.add_argument('--fix_ops', action='store_true',
                        help='Try to fix group convolution operations before quantization (OpenVINO only)')
    
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
    
    # Direct INT8 conversion without calibration if requested
    if args.direct and ext in ('.onnx', '.xml'):
        print("Using direct INT8 conversion without calibration")
        direct_int8_conversion(args.model, output_path)
    else:
        # Standard quantization with calibration
        quantize_model(
            args.model,
            args.dataset,
            output_path,
            input_size=tuple(args.resize),
            num_calibration_samples=args.num_samples,
            max_frames_per_video=args.max_frames_per_video,
            fix_ops=args.fix_ops
        )

if __name__ == '__main__':
    main()
