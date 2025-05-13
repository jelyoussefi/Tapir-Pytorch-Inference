# Suppress TensorFlow, IPEX, and ONNX warnings before any imports
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_LOGGING'] = '0'
os.environ['TF_CUDNN_LOGGING_LEVEL'] = '3'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['PYTORCH_IPEX_VERBOSE'] = '0'
os.environ['TORCH_CPP_LOG_LEVEL'] = 'ERROR'
os.environ['QT_LOGGING_RULES'] = 'qt5ct.debug=false'
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('intel_extension_for_pytorch').setLevel(logging.ERROR)
logging.getLogger().setLevel(logging.ERROR)
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
from tapnet.tapir_openvino import TAPIR_OpenVINO
from tapnet.utils import get_query_features, postprocess_occlusions, preprocess_frame
from openvino.runtime import Core

def build_model(model_path: str, input_resolution: tuple[int, int], num_pips_iter: int, use_causal_conv: bool,
                device: torch.device, precision: str = "FP32"):
    if model_path is not None:
        model_ext = os.path.splitext(model_path)[1].lower()
        
        if model_ext in ['.onnx', '.xml']:
            print(f"Using OpenVINO for  model")
            model = TAPIR_OpenVINO(model_path=model_path, use_causal_conv=use_causal_conv, num_pips_iter=num_pips_iter,
                                   initial_resolution=input_resolution, device=device)
        else:
            if precision == 'INT8':
                print(f"Loading INT8 model from {model_path} to device {device}")
                model = torch.jit.load(model_path, map_location='cpu').to(device)
            else:
                print(f"Loading FP32 model from {model_path} to device {device}")
                model = TAPIR(use_causal_conv=use_causal_conv, num_pips_iter=num_pips_iter,
                              initial_resolution=input_resolution, device=device)
                model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))

            model.to(device)
            model = model.eval()

            if str(device).startswith('xpu'):
                model = ipex.optimize(model, dtype=torch.float32)

        return model
    return None

class TapirPredictor(nn.Module):
    def __init__(self, model: TAPIR):
        super().__init__()
        self.model = model
        self.device = getattr(model, 'device', torch.device('cpu'))

    @torch.no_grad()
    def forward(self, frame, query_feats, hires_query_feats, causal_context):
        frame = frame.to(self.device)
        query_feats = query_feats.to(self.device)
        hires_query_feats = hires_query_feats.to(self.device)
        causal_context = causal_context.to(self.device)

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
        self.device = device
        self.input_resolution = input_resolution
        self.num_pips_iter = num_pips_iter
        self.num_points = 100
        self.num_mixer_blocks = getattr(self.model, 'num_mixer_blocks', 12)
        self.precision = precision
        self.is_quantized = precision == "INT8"

        if not self.is_quantized:
            self.predictor = TapirPredictor(self.model).to(device)
            self.encoder = TapirPointEncoder(self.model).to(device)
        else:
            self.predictor = self.model
            self.encoder = None

        causal_state_shape = (num_pips_iter, self.num_mixer_blocks, self.num_points, 2, 512 + 2048)
        self.causal_state = torch.zeros(causal_state_shape, dtype=torch.float32, device=self.device)
        self.query_feats = torch.zeros((1, self.num_points, 256), dtype=torch.float32, device=self.device)
        self.hires_query_feats = torch.zeros((1, self.num_points, 128), dtype=torch.float32, device=self.device)

    def set_points(self, frame: np.ndarray, query_points: np.ndarray, preprocess: bool = True):
        query_points = query_points.astype(np.float32)
        query_points[..., 0] = query_points[..., 0] / self.input_resolution[1]
        query_points[..., 1] = query_points[..., 1] / self.input_resolution[0]

        query_points = torch.tensor(query_points).to(self.device)

        num_points = query_points.shape[0]
        causal_state_shape = (self.model.num_pips_iter, self.model.num_mixer_blocks, num_points, 2, 512 + 2048)
        self.causal_state = torch.zeros(causal_state_shape, dtype=torch.float32, device=self.device)
        query_feats = torch.zeros((1, num_points, 256), dtype=torch.float32, device=self.device)
        hires_query_feats = torch.zeros((1, num_points, 128), dtype=torch.float32, device=self.device)

        if preprocess:
            input_frame = preprocess_frame(frame, resize=self.input_resolution, device=self.device)
        else:
            input_frame = frame

        input_frame = preprocess_frame(frame, resize=self.input_resolution, device=self.device)
        _, _, _, feature_grid, hires_feats_grid = self.predictor(input_frame, query_feats, hires_query_feats, self.causal_state)

        self.query_feats, self.hires_query_feats = self.encoder(query_points[None], feature_grid, hires_feats_grid)

    
    @torch.no_grad()
    def forward(self, frame: np.ndarray):
        tensor = False
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
            tensor = True

        height, width = frame.shape[:2]

        input_frame = preprocess_frame(frame, resize=self.input_resolution, device=self.device)

        tracks, visibles, self.causal_state, _, _ = self.predictor( input_frame, self.query_feats, self.hires_query_feats, self.causal_state)
           

        visibles = visibles.cpu().numpy().squeeze()
        tracks = tracks.cpu().numpy().squeeze()
       
        tracks[:, 0] = tracks[:, 0] * width / self.input_resolution[1]
        tracks[:, 1] = tracks[:, 1] * height / self.input_resolution[0]

        if not tensor:
            return tracks, visibles
        else:
            return torch.from_numpy(tracks), torch.from_numpy(visibles)

