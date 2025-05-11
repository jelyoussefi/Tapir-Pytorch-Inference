# Suppress TensorFlow, IPEX, and ONNX warnings before any imports
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # Suppress TensorFlow logs
os.environ['TF_LOGGING'] = '0'            # Additional TensorFlow suppression
os.environ['TF_CUDNN_LOGGING_LEVEL'] = '3' # Suppress cuDNN logs
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Prevent GPU memory issues
os.environ['PYTORCH_IPEX_VERBOSE'] = '0'   # Suppress IPEX verbose output
os.environ['TORCH_CPP_LOG_LEVEL'] = 'ERROR'  # Suppress PyTorch C++ logs
os.environ['QT_LOGGING_RULES'] = 'qt5ct.debug=false'  # Suppress Qt/Wayland warnings
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)  # Stronger TensorFlow suppression
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('intel_extension_for_pytorch').setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)  # Suppress IPEX info logs
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=FutureWarning, module='intel_extension_for_pytorch')
warnings.filterwarnings('ignore', category=UserWarning, module='torch.onnx')
warnings.filterwarnings('ignore', category=Warning, module='torch.jit')

import numpy as np
import torch
import intel_extension_for_pytorch as ipex
from torch import nn
from tapnet.tapir_model import TAPIR
from tapnet.utils import get_query_features, postprocess_occlusions, preprocess_frame
from openvino.runtime import Core

def build_model(model_path: str, input_resolution: tuple[int, int], num_pips_iter: int, use_casual_conv: bool,
                device: torch.device, precision: str = "FP32"):
    if model_path is not None:
        if precision == 'INT8':
            print(f"Loading INT8 model from {model_path} to device {device}")
            # For INT8 models, load the TorchScript model
            try:
                # First create a standard TAPIR model for utility functions
                model = TAPIR(use_casual_conv=use_casual_conv, num_pips_iter=num_pips_iter,
                            initial_resolution=input_resolution, device=device)
                
                # Load the quantized model to the target device 
                # Note: For XPU (Intel GPU), INT8 models should be loaded to CPU first,
                # then moved to XPU if needed
                quantized_model = torch.jit.load(model_path, map_location='cpu')
                
                # Set flags
                model.is_quantized = True
                model.quantized_model = quantized_model
                model.device = device
                
                print(f"INT8 model loaded successfully, device={device}")
                return model
                
            except Exception as e:
                print(f"Error loading INT8 model: {e}")
                raise
        else:
            # Standard FP32 model loading
            model = TAPIR(use_casual_conv=use_casual_conv, num_pips_iter=num_pips_iter,
                          initial_resolution=input_resolution, device=device)
            model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
            model.is_quantized = False
            model = model.to(device)

        # Set model to evaluation mode
        model = model.eval()
        
        # Optimize with IPEX if on XPU and using FP32
        if str(device).startswith('xpu') and precision == "FP32":
            # Optimize model with IPEX
            model = ipex.optimize(model, dtype=torch.float32)

        return model
    return None

class TapirPredictor(nn.Module):
    def __init__(self, model: TAPIR):
        super().__init__()
        self.model = model
        # Store the device from the model for consistency
        self.device = getattr(model, 'device', torch.device('cpu'))

    @torch.no_grad()
    def forward(self, frame, query_feats, hires_query_feats, causal_context):
        # Check if we're using a quantized model
        if hasattr(self.model, 'is_quantized') and self.model.is_quantized:
            # For quantized model, we need to call the traced model directly
            if hasattr(self.model, 'quantized_model'):
                # Device management for different device types
                device_str = str(self.device)
                
                # If using XPU (Intel GPU), ensure model is on CPU for INT8 inference
                if device_str.startswith('xpu'):
                    # For Intel XPU, first make sure model is on CPU for INT8 inference
                    # then move outputs to XPU if needed
                    cpu_model = self.model.quantized_model.to('cpu')
                    
                    # Make sure all inputs are on CPU
                    frame_cpu = frame.cpu()
                    query_feats_cpu = query_feats.cpu()
                    hires_query_feats_cpu = hires_query_feats.cpu()
                    causal_context_cpu = causal_context.cpu()
                    
                    # Run the model on CPU
                    outputs = cpu_model(frame_cpu, query_feats_cpu, hires_query_feats_cpu, causal_context_cpu)
                    
                    # Move outputs back to XPU if needed
                    tracks, visibles, new_causal_context, feature_grid, hires_feats_grid = outputs
                    
                    # Move outputs to the target device (XPU)
                    tracks = tracks.to(self.device)
                    visibles = visibles.to(self.device)
                    new_causal_context = new_causal_context.to(self.device)
                    feature_grid = feature_grid.to(self.device)
                    hires_feats_grid = hires_feats_grid.to(self.device)
                    
                    return tracks, visibles, new_causal_context, feature_grid, hires_feats_grid
                else:
                    # For CPU or CUDA, ensure everything is on the correct device
                    # No need to get model device - just use self.device
                    # This fixes the CPU StopIteration error
                    model_device = self.device
                    
                    # Make sure all inputs are on the right device
                    frame = frame.to(model_device)
                    query_feats = query_feats.to(model_device)
                    hires_query_feats = hires_query_feats.to(model_device)
                    causal_context = causal_context.to(model_device)
                    
                    # Run the model
                    return self.model.quantized_model(frame, query_feats, hires_query_feats, causal_context)
            else:
                raise RuntimeError("Quantized model not properly initialized")
        else:
            # Original code path for non-quantized model
            feature_grid, hires_feats_grid = self.model.get_feature_grids(frame)
    
            tracks, occlusions, expected_dist, causal_context = self.model.estimate_trajectories(
                feature_grid=feature_grid,
                hires_feats_grid=hires_feats_grid,
                query_feats=query_feats,
                hires_query_feats=hires_query_feats,
                causal_context=causal_context
            )
            visibles = postprocess_occlusions(occlusions, expected_dist)
            return tracks, visibles, causal_context, feature_grid, hires_feats_grid

class TapirPointEncoder(nn.Module):
    def __init__(self, model: TAPIR):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, query_points, feature_grid, hires_feats_grid):
        return get_query_features(query_points, feature_grid, hires_feats_grid)

class TapirInference(nn.Module):
    def __init__(self, model_path: str, input_resolution: tuple[int, int], num_pips_iter: int, device: torch.device, precision: str = "FP32"):
        super().__init__()
        self.model = build_model(model_path, input_resolution, num_pips_iter, True, device, precision)
        self.predictor = TapirPredictor(self.model).to(device)
        self.encoder = TapirPointEncoder(self.model).to(device)
        self.device = device
        self.input_resolution = input_resolution
        self.num_pips_iter = num_pips_iter
        self.num_points = 256
        self.num_mixer_blocks = getattr(self.model, 'num_mixer_blocks', 12)  # Default to 12 if not available
        self.precision = precision

        causal_state_shape = (num_pips_iter, self.num_mixer_blocks, self.num_points, 2, 512 + 2048)
        self.causal_state = torch.zeros(causal_state_shape, dtype=torch.float32, device=self.device)
        self.query_feats = torch.zeros((1, self.num_points, 256), dtype=torch.float32, device=self.device)
        self.hires_query_feats = torch.zeros((1, self.num_points, 128), dtype=torch.float32, device=self.device)

    def set_points(self, frame: np.ndarray, query_points: np.ndarray, debug: bool = False):
        query_points = query_points.astype(np.float32)
        query_points[..., 0] = query_points[..., 0] / self.input_resolution[1]
        query_points[..., 1] = query_points[..., 1] / self.input_resolution[0]

        query_points = torch.tensor(query_points).to(self.device)

        num_points = query_points.shape[0]
        causal_state_shape = (self.num_pips_iter, self.num_mixer_blocks, num_points, 2, 512 + 2048)
        self.causal_state = torch.zeros(causal_state_shape, dtype=torch.float32, device=self.device)
        query_feats = torch.zeros((1, num_points, 256), dtype=torch.float32, device=self.device)
        hires_query_feats = torch.zeros((1, num_points, 128), dtype=torch.float32, device=self.device)

        input_frame = preprocess_frame(frame, resize=self.input_resolution, device=self.device)
        
        # Get feature grids
        try:
            print(f"Running initial inference to get feature grids...")
            _, _, _, feature_grid, hires_feats_grid = self.predictor(input_frame, query_feats, hires_query_feats, self.causal_state)
            print(f"Feature grids extracted successfully.")
            
            print(f"Computing query features...")
            self.query_feats, self.hires_query_feats = self.encoder(query_points[None], feature_grid, hires_feats_grid)
            print(f"Query features computed successfully.")
            
        except Exception as e:
            print(f"Error during initialization: {type(e).__name__}: {e}")
            raise

    @torch.no_grad()
    def forward(self, frame: np.ndarray):
        height, width = frame.shape[:2]

        input_frame = preprocess_frame(frame, resize=self.input_resolution, device=self.device)
        tracks, visibles, self.causal_state, _, _ = self.predictor(input_frame, self.query_feats,
                                                                   self.hires_query_feats, self.causal_state)

        visibles = visibles.cpu().numpy().squeeze()
        tracks = tracks.cpu().numpy().squeeze()

        tracks[:, 0] = tracks[:, 0] * width / self.input_resolution[1]
        tracks[:, 1] = tracks[:, 1] * height / self.input_resolution[0]

        return tracks, visibles