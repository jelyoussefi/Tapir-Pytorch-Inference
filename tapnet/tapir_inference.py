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
                device: torch.device):
    if model_path is not None:
        model = TAPIR(use_casual_conv=use_casual_conv, num_pips_iter=num_pips_iter,
                      initial_resolution=input_resolution, device=device)
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
        model = model.to(device)
        model = model.eval()
        if device == "xpu":
            # Optimize model with IPEX
            model = ipex.optimize(model, dtype=torch.float32)
        return model
    return None

class TapirPredictor(nn.Module):
    def __init__(self, model: TAPIR):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, frame, query_feats, hires_query_feats, causal_context):
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
    def __init__(self, model_path: str, input_resolution: tuple[int, int], num_pips_iter: int, device: torch.device):
        super().__init__()
        self.model = build_model(model_path, input_resolution, num_pips_iter, True, device)
        self.predictor = TapirPredictor(self.model).to(device)
        self.encoder = TapirPointEncoder(self.model).to(device)
        self.device = device
        self.input_resolution = input_resolution
        self.num_pips_iter = num_pips_iter
        self.num_points = 256
        self.num_mixer_blocks = self.model.num_mixer_blocks

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
        _, _, _, feature_grid, hires_feats_grid = self.predictor(input_frame, query_feats, hires_query_feats, self.causal_state)

        self.query_feats, self.hires_query_feats = self.encoder(query_points[None], feature_grid, hires_feats_grid)

    @torch.no_grad()
    def forward(self, frame: np.ndarray, debug: bool = False):
        height, width = frame.shape[:2]

        input_frame = preprocess_frame(frame, resize=self.input_resolution, device=self.device)
        tracks, visibles, self.causal_state, _, _ = self.predictor(input_frame, self.query_feats,
                                                                   self.hires_query_feats, self.causal_state)

        visibles = visibles.cpu().numpy().squeeze()
        tracks = tracks.cpu().numpy().squeeze()

        tracks[:, 0] = tracks[:, 0] * width / self.input_resolution[1]
        tracks[:, 1] = tracks[:, 1] * height / self.input_resolution[0]

        return tracks, visibles

class TapirInferenceOpenVINO(nn.Module):
    def __init__(self, predictor_path: str, encoder_path: str, input_resolution: tuple[int, int], num_pips_iter: int, device: torch.device):
        super().__init__()
        self.predictor_path = predictor_path
        self.encoder_path = encoder_path
        self.input_resolution = input_resolution
        self.num_pips_iter = num_pips_iter
        self.device = device
        self.num_points = 256  # Default, will be updated in set_points
        self.num_mixer_blocks = 12  # From TAPIR default in tapir_model.py

        # Initialize OpenVINO models
        self.ov_core = Core()
        ov_encoder_model = self.ov_core.read_model(encoder_path)
        ov_predictor_model = self.ov_core.read_model(predictor_path)
        self.ov_encoder_compiled = self.ov_core.compile_model(ov_encoder_model, 'GPU' if 'xpu' in device.type else 'CPU')
        self.ov_predictor_compiled = self.ov_core.compile_model(ov_predictor_model, 'GPU' if 'xpu' in device.type else 'CPU')

        self.ov_encoder_request = self.ov_encoder_compiled.create_infer_request()
        self.ov_predictor_request = self.ov_predictor_compiled.create_infer_request()

        # Initialize state tensors
        causal_state_shape = (self.num_pips_iter, self.num_mixer_blocks, self.num_points, 2, 512 + 2048)
        self.causal_state = torch.zeros(causal_state_shape, dtype=torch.float32, device=self.device)
        self.query_feats = torch.zeros((1, self.num_points, 256), dtype=torch.float32, device=self.device)
        self.hires_query_feats = torch.zeros((1, self.num_points, 128), dtype=torch.float32, device=self.device)

    def set_points(self, frame: np.ndarray, query_points: np.ndarray, debug: bool = False):
        # Normalize query points
        query_points = query_points.astype(np.float32)
        query_points[..., 0] = query_points[..., 0] / self.input_resolution[1]
        query_points[..., 1] = query_points[..., 1] / self.input_resolution[0]

        query_points_tensor = torch.tensor(query_points, dtype=torch.float32, device=self.device)
        num_points = query_points.shape[0]

        # Update causal state and feature tensors for new number of points
        causal_state_shape = (self.num_pips_iter, self.num_mixer_blocks, num_points, 2, 512 + 2048)
        self.causal_state = torch.zeros(causal_state_shape, dtype=torch.float32, device=self.device)
        query_feats = torch.zeros((1, num_points, 256), dtype=torch.float32, device=self.device)
        hires_query_feats = torch.zeros((1, num_points, 128), dtype=torch.float32, device=self.device)

        # Preprocess frame on CPU
        cpu_device = torch.device('cpu')
        input_frame = preprocess_frame(frame, resize=self.input_resolution, device=cpu_device)

        # Initialize temporary PyTorch model on CPU
        model = TAPIR(
            use_casual_conv=True,
            num_pips_iter=self.num_pips_iter,
            initial_resolution=self.input_resolution,
            device=cpu_device
        ).to(cpu_device)
        model.eval()
        predictor = TapirPredictor(model).to(cpu_device)

        # Compute feature grids
        _, _, _, feature_grid, hires_feats_grid = predictor(
            input_frame, query_feats.to(cpu_device), hires_query_feats.to(cpu_device), self.causal_state.to(cpu_device)
        )

        # Move feature grids to XPU
        feature_grid = feature_grid.to(self.device)
        hires_feats_grid = hires_feats_grid.to(self.device)

        # Debug logging
        if debug:
            logging.debug(f"Feature grid shape: {feature_grid.shape}, min: {feature_grid.min()}, max: {feature_grid.max()}")
            logging.debug(f"Hires feats grid shape: {hires_feats_grid.shape}, min: {hires_feats_grid.min()}, max: {hires_feats_grid.max()}")

        # Run encoder inference with OpenVINO
        self.ov_encoder_request.infer({
            'query_points': query_points_tensor[None].cpu().numpy(),
            'feature_grid': feature_grid.cpu().numpy(),
            'hires_feats_grid': hires_feats_grid.cpu().numpy()
        })
        self.query_feats = torch.from_numpy(self.ov_encoder_request.get_output_tensor(0).data).to(self.device)
        self.hires_query_feats = torch.from_numpy(self.ov_encoder_request.get_output_tensor(1).data).to(self.device)

        # Debug logging
        if debug:
            logging.debug(f"Query feats shape: {self.query_feats.shape}, min: {self.query_feats.min()}, max: {self.query_feats.max()}")
            logging.debug(f"Hires query feats shape: {self.hires_query_feats.shape}, min: {self.hires_query_feats.min()}, max: {self.hires_query_feats.max()}")

        # Free PyTorch model memory
        del model
        del predictor
        if 'xpu' in self.device.type:
            torch.xpu.empty_cache()
        elif 'cuda' in self.device.type:
            torch.cuda.empty_cache()

    @torch.no_grad()
    def forward(self, frame: np.ndarray, debug: bool = False):
        height, width = frame.shape[:2]

        # Preprocess frame
        input_frame = preprocess_frame(frame, resize=self.input_resolution, device=self.device)

        # Run predictor inference with OpenVINO
        self.ov_predictor_request.infer({
            'input_frame': input_frame.cpu().numpy(),
            'query_feats': self.query_feats.cpu().numpy(),
            'hires_query_feats': self.hires_query_feats.cpu().numpy(),
            'causal_state': self.causal_state.cpu().numpy()
        })

        tracks_out = self.ov_predictor_request.get_output_tensor(0).data
        visibles = self.ov_predictor_request.get_output_tensor(1).data
        self.causal_state = torch.from_numpy(self.ov_predictor_request.get_output_tensor(2).data).to(self.device)

        tracks_out = tracks_out.squeeze()
        visibles = visibles.squeeze()

        # Debug logging
        if debug:
            logging.debug(f"Tracks out shape: {tracks_out.shape}, min: {tracks_out.min()}, max: {tracks_out.max()}")
            logging.debug(f"Visibles shape: {visibles.shape}, sum: {visibles.sum()}")

        # Postprocess outputs
        tracks_out = tracks_out * np.array([width / self.input_resolution[1], height / self.input_resolution[0]])
        visibles = postprocess_occlusions(torch.from_numpy(visibles), torch.ones_like(torch.from_numpy(visibles))).numpy()

        # Validate outputs
        if tracks_out.shape != (self.query_feats.shape[1], 2) or visibles.shape != (self.query_feats.shape[1],):
            raise ValueError(f"Invalid output shapes: tracks_out {tracks_out.shape}, visibles {visibles.shape}")

        return tracks_out, visibles