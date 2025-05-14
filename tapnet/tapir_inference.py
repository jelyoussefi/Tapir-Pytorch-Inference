# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""TAPIR inference module for point tracking.

This module provides classes to initialize and run TAPIR models,
supporting both PyTorch and OpenVINO backends for video point tracking.
"""

import os
import numpy as np
import torch
import intel_extension_for_pytorch as ipex
from torch import nn
from tapnet.tapir_model import TAPIR
from tapnet.tapir_openvino import TAPIR_OpenVINO
from tapnet.utils import get_query_features, postprocess_occlusions, preprocess_frame

# Suppress warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_LOGGING"] = "0"
os.environ["TF_CUDNN_LOGGING_LEVEL"] = "3"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["PYTORCH_IPEX_VERBOSE"] = "0"
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
os.environ["QT_LOGGING_RULES"] = "qt5ct.debug=false"


def build_model(
    model_path: str,
    input_resolution: tuple[int, int],
    num_pips_iter: int,
    use_causal_conv: bool,
    device: torch.device,
    precision: str = "FP32",
):
    """Build and initialize TAPIR model.

    Args:
        model_path: Path to model (.pt, .onnx, or .xml).
        input_resolution: Input resolution (height, width).
        num_pips_iter: Number of PIPS iterations.
        use_causal_conv: Use causal convolutions.
        device: Torch device (e.g., CPU, GPU).
        precision: Model precision ("FP32" or "INT8").

    Returns:
        Initialized TAPIR model (PyTorch or OpenVINO).
    """
    if not model_path:
        return None

    model_ext = os.path.splitext(model_path)[1].lower()

    if model_ext in [".onnx", ".xml"]:
        model = TAPIR_OpenVINO(
            model_path=model_path,
            use_causal_conv=use_causal_conv,
            num_pips_iter=num_pips_iter,
            initial_resolution=input_resolution,
            device=device,
        )
    else:
        if precision == "INT8":
            model = torch.jit.load(model_path, map_location="cpu").to(device)
        else:
            model = TAPIR(
                use_causal_conv=use_causal_conv,
                num_pips_iter=num_pips_iter,
                initial_resolution=input_resolution,
                device=device,
            )
            model.load_state_dict(
                torch.load(model_path, weights_only=True, map_location=device)
            )

        model.to(device)
        model.eval()

        if str(device).startswith("xpu"):
            model = ipex.optimize(model, dtype=torch.float32)

    return model


class TapirPredictor(nn.Module):
    """Predictor for TAPIR point tracking.

    Processes frames and query features to estimate point trajectories.
    """

    def __init__(self, model: TAPIR):
        super().__init__()
        self.model = model
        self.device = getattr(model, "device", torch.device("cpu"))

    @torch.no_grad()
    def forward(self, frame, query_feats, hires_query_feats, causal_context):
        """Run inference on a single frame.

        Args:
            frame: Input frame tensor (1, C, H, W).
            query_feats: Query features (1, N, 256).
            hires_query_feats: High-resolution query features (1, N, 128).
            causal_context: Causal state tensor.

        Returns:
            Tuple of tracks, visibles, updated causal context, feature grid,
            and high-resolution feature grid.
        """
        frame = frame.to(self.device)
        query_feats = query_feats.to(self.device)
        hires_query_feats = hires_query_feats.to(self.device)
        causal_context = causal_context.to(self.device)

        if isinstance(self.model, TAPIR_OpenVINO):
            feature_grid, hires_feats_grid = self.model.get_feature_grids(
                frame, query_feats, hires_query_feats, causal_context
            )
        else:
            feature_grid, hires_feats_grid = self.model.get_feature_grids(frame)

        tracks, occlusions, expected_dist, causal_context = self.model.estimate_trajectories(
            feature_grid=feature_grid,
            hires_feats_grid=hires_feats_grid,
            query_feats=query_feats,
            hires_query_feats=hires_query_feats,
            causal_context=causal_context,
        )

        visibles = (
            occlusions
            if isinstance(self.model, TAPIR_OpenVINO)
            else postprocess_occlusions(occlusions, expected_dist)
        )

        return tracks, visibles, causal_context, feature_grid, hires_feats_grid


class TapirPointEncoder(nn.Module):
    """Encoder for extracting query features from points.

    Converts query points to features using feature grids.
    """

    def __init__(self, model: TAPIR):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, query_points, feature_grid, hires_feats_grid):
        """Extract features for query points.

        Args:
            query_points: Query points tensor (1, N, 2).
            feature_grid: Feature grid (1, H, W, 256).
            hires_feats_grid: High-resolution feature grid (1, H, W, 128).

        Returns:
            Tuple of query features and high-resolution query features.
        """
        return get_query_features(query_points, feature_grid, hires_feats_grid)


class TapirInference(nn.Module):
    """Main class for TAPIR inference.

    Initializes the model and handles point tracking across video frames.
    """

    def __init__(
        self,
        model_path: str,
        input_resolution: tuple[int, int],
        num_pips_iter: int,
        device: torch.device,
        precision: str = "FP32",
    ):
        super().__init__()
        self.model = build_model(
            model_path, input_resolution, num_pips_iter, True, device, precision
        )
        self.device = device
        self.input_resolution = input_resolution
        self.num_pips_iter = num_pips_iter
        self.num_points = 100
        self.num_mixer_blocks = getattr(self.model, "num_mixer_blocks", 12)
        self.precision = precision
        self.is_quantized = precision == "INT8"

        if not self.is_quantized:
            self.predictor = TapirPredictor(self.model).to(device)
            self.encoder = TapirPointEncoder(self.model).to(device)
        else:
            self.predictor = self.model
            self.encoder = None

        causal_state_shape = (
            num_pips_iter,
            self.num_mixer_blocks,
            self.num_points,
            2,
            512 + 2048,
        )
        self.causal_state = torch.zeros(
            causal_state_shape, dtype=torch.float32, device=self.device
        )
        self.query_feats = torch.zeros(
            (1, self.num_points, 256), dtype=torch.float32, device=self.device
        )
        self.hires_query_feats = torch.zeros(
            (1, self.num_points, 128), dtype=torch.float32, device=self.device
        )

    def set_points(self, frame: np.ndarray, query_points: np.ndarray, preprocess: bool = True):
        """Initialize query points for tracking.

        Args:
            frame: Input frame (H, W, 3).
            query_points: Query points (N, 2).
            preprocess: Whether to preprocess the frame.
        """
        query_points = query_points.astype(np.float32)
        query_points[..., 0] = query_points[..., 0] / self.input_resolution[1]
        query_points[..., 1] = query_points[..., 1] / self.input_resolution[0]

        query_points = torch.tensor(query_points).to(self.device)

        num_points = query_points.shape[0]
        causal_state_shape = (
            self.model.num_pips_iter,
            self.model.num_mixer_blocks,
            num_points,
            2,
            512 + 2048,
        )
        self.causal_state = torch.zeros(
            causal_state_shape, dtype=torch.float32, device=self.device
        )
        query_feats = torch.zeros((1, num_points, 256), dtype=torch.float32, device=self.device)
        hires_query_feats = torch.zeros(
            (1, num_points, 128), dtype=torch.float32, device=self.device
        )

        input_frame = preprocess_frame(frame, resize=self.input_resolution, device=self.device)
        if isinstance(self.model, TAPIR_OpenVINO):
            feature_grid, hires_feats_grid = self.model.get_feature_grids(
                input_frame, query_feats, hires_query_feats, self.causal_state
            )
            self.query_feats, self.hires_query_feats = get_query_features(
                query_points[None].to(self.device), feature_grid, hires_feats_grid
            )
            self.model.query_feats = self.query_feats
            self.model.hires_query_feats = self.hires_query_feats
            self.model.causal_state = self.causal_state
        else:
            _, _, _, feature_grid, hires_feats_grid = self.predictor(
                input_frame, query_feats, hires_query_feats, self.causal_state
            )
            self.query_feats, self.hires_query_feats = self.encoder(
                query_points[None], feature_grid, hires_feats_grid
            )

    @torch.no_grad()
    def forward(self, frame: np.ndarray):
        """Process a frame to track points.

        Args:
            frame: Input frame (H, W, 3) or tensor.

        Returns:
            Tuple of tracks (N, 2) and visibles (N,) as numpy arrays or tensors.
        """
        tensor = isinstance(frame, torch.Tensor)
        if tensor:
            frame = frame.cpu().numpy()

        height, width = frame.shape[:2]
        input_frame = preprocess_frame(frame, resize=self.input_resolution, device=self.device)

        tracks, visibles, self.causal_state, _, _ = self.predictor(
            input_frame, self.query_feats, self.hires_query_feats, self.causal_state
        )

        visibles = visibles.cpu().numpy().squeeze()
        tracks = tracks.cpu().numpy().squeeze()

        # Scale tracks to original frame dimensions
        if np.max(tracks, axis=0).any() > 1.0 or np.min(tracks, axis=0).any() < 0.0:
            tracks = (tracks - np.min(tracks, axis=0)) / (
                np.max(tracks, axis=0) - np.min(tracks, axis=0) + 1e-6
            )
        tracks[:, 0] = tracks[:, 0] * width / self.input_resolution[1]
        tracks[:, 1] = tracks[:, 1] * height / self.input_resolution[0]

        return (
            (tracks, visibles)
            if not tensor
            else (torch.from_numpy(tracks), torch.from_numpy(visibles))
        )