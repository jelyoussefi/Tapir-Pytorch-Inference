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

"""OpenVINO implementation of TAPIR model for inference."""

from typing import Optional, Tuple

import numpy as np
import torch
from openvino import Core
from torch import nn, Tensor

from tapnet import utils
from tapnet.tapir_model import TAPIR


class TAPIR_OpenVINO(TAPIR):
    """TAPIR model for OpenVINO inference.

    This class extends the TAPIR model to support inference using OpenVINO,
    handling model loading, feature extraction, and trajectory estimation.
    """

    def __init__(
        self,
        model_path: str,
        bilinear_interp_with_depthwise_conv: bool = False,
        num_pips_iter: int = 4,
        pyramid_level: int = 1,
        num_mixer_blocks: int = 12,
        patch_size: int = 7,
        softmax_temperature: float = 20.0,
        parallelize_query_extraction: bool = False,
        initial_resolution: Tuple[int, int] = (256, 256),
        feature_extractor_chunk_size: int = 10,
        extra_convs: bool = True,
        use_causal_conv: bool = False,
        device: Optional[torch.device] = None,
    ):
        """Initialize TAPIR model for OpenVINO inference.

        Args:
            model_path: Path to the OpenVINO model (.onnx or .xml).
            bilinear_interp_with_depthwise_conv: Use bilinear interpolation with depthwise convolution.
            num_pips_iter: Number of PIPS iterations.
            pyramid_level: Feature pyramid level.
            num_mixer_blocks: Number of mixer blocks.
            patch_size: Patch size for feature extraction.
            softmax_temperature: Temperature for softmax in attention.
            parallelize_query_extraction: Parallelize query feature extraction.
            initial_resolution: Input resolution (height, width).
            feature_extractor_chunk_size: Chunk size for feature extraction.
            extra_convs: Use extra convolutional layers.
            use_causal_conv: Use causal convolutions.
            device: Torch device (e.g., CPU, GPU).
        """
        super().__init__(
            model_path=model_path,
            bilinear_interp_with_depthwise_conv=bilinear_interp_with_depthwise_conv,
            num_pips_iter=num_pips_iter,
            pyramid_level=pyramid_level,
            num_mixer_blocks=num_mixer_blocks,
            patch_size=patch_size,
            softmax_temperature=softmax_temperature,
            parallelize_query_extraction=parallelize_query_extraction,
            initial_resolution=initial_resolution,
            feature_extractor_chunk_size=feature_extractor_chunk_size,
            extra_convs=extra_convs,
            use_causal_conv=use_causal_conv,
            device=device,
        )

        self.device = device or torch.device("cpu")
        self.num_points = 100
        self.ov_core = Core()

        # Initialize storage for inference outputs
        self.feature_grid = None
        self.hires_feats_grid = None
        self.tracks = None
        self.occlusions = None
        self.new_causal_context = None

        # Initialize query features and causal state
        self.query_feats = torch.zeros(
            (1, self.num_points, 256), dtype=torch.float32, device=self.device
        )
        self.hires_query_feats = torch.zeros(
            (1, self.num_points, 128), dtype=torch.float32, device=self.device
        )
        causal_state_shape = (
            num_pips_iter,
            num_mixer_blocks,
            self.num_points,
            2,
            512 + 2048,
        )
        self.causal_state = torch.zeros(
            causal_state_shape, dtype=torch.float32, device=self.device
        )

        # Determine OpenVINO device
        ov_device = "GPU" if str(self.device).startswith(("xpu", "cuda")) else "CPU"

        # Load and compile OpenVINO model
        try:
            ov_model = self.ov_core.read_model(model_path)
            self.ov_compiled_model = self.ov_core.compile_model(ov_model, ov_device)
            self.ov_infer_request = self.ov_compiled_model.create_infer_request()
        except Exception as e:
            raise RuntimeError(f"Failed to load OpenVINO model: {e}")

        # Store input/output names
        self.input_names = [input.any_name for input in self.ov_compiled_model.inputs]
        self.output_names = [output.any_name for output in self.ov_compiled_model.outputs]

    def get_feature_grids(
        self,
        frame: torch.Tensor,
        query_feats: torch.Tensor,
        hires_query_feats: torch.Tensor,
        causal_state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract and store feature grids and trajectory outputs.

        Args:
            frame: Input frame tensor (1, C, H, W).
            query_feats: Query features (1, N, 256).
            hires_query_feats: High-resolution query features (1, N, 128).
            causal_state: Causal state (num_pips_iter, num_mixer_blocks, N, 2, 2560).

        Returns:
            Tuple of feature_grid (1, H, W, 256) and hires_feats_grid (1, H, W, 128).

        Raises:
            RuntimeError: If feature grids are not retrieved.
        """
        # Prepare inputs for inference
        inputs = {
            "input_frame": frame.cpu().numpy(),
            "query_feats": query_feats.cpu().numpy(),
            "hires_query_feats": hires_query_feats.cpu().numpy(),
            "causal_state": causal_state.cpu().numpy(),
        }

        # Map inputs to model input names
        input_dict = {
            name: inputs.get(name, inputs.get(name.split(":")[0], np.zeros([1])))
            for name in self.input_names
        }

        # Run inference
        try:
            self.ov_infer_request.infer(input_dict)
        except Exception as e:
            raise RuntimeError(f"OpenVINO inference failed in get_feature_grids: {e}")

        # Reset output storage
        self.feature_grid = None
        self.hires_feats_grid = None
        self.tracks = None
        self.occlusions = None
        self.new_causal_context = None

        # Store outputs
        for i, name in enumerate(self.output_names):
            output = self.ov_infer_request.get_output_tensor(i).data
            output_tensor = torch.from_numpy(output).to(self.device)
            if "tracks" in name:
                self.tracks = output_tensor.squeeze(0)
            elif "visibles" in name:
                self.occlusions = output_tensor.squeeze()
            elif "feature_grid" in name:
                self.feature_grid = output_tensor
            elif "hires_feats_grid" in name:
                self.hires_feats_grid = output_tensor
            elif "causal_state" in name:
                self.new_causal_context = output_tensor

        if self.feature_grid is None or self.hires_feats_grid is None:
            raise RuntimeError("Failed to retrieve feature grids from model outputs")

        return self.feature_grid, self.hires_feats_grid

    def estimate_trajectories(
        self,
        feature_grid: torch.Tensor,
        hires_feats_grid: torch.Tensor,
        query_feats: torch.Tensor,
        hires_query_feats: torch.Tensor,
        causal_context: torch.Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Process stored trajectory outputs from get_feature_grids.

        Args:
            feature_grid: Feature grid (1, H, W, 256).
            hires_feats_grid: High-resolution feature grid (1, H, W, 128).
            query_feats: Query features (1, N, 256).
            hires_query_feats: High-resolution query features (1, N, 128).
            causal_context: Causal state (num_pips_iter, num_mixer_blocks, N, 2, 2560).

        Returns:
            Tuple of tracks (N, 2), occlusions (N,), expected_dist (N,), and new causal context.

        Raises:
            RuntimeError: If trajectory outputs are not initialized or tracks shape is invalid.
        """
        if (
            self.tracks is None
            or self.occlusions is None
            or self.new_causal_context is None
        ):
            raise RuntimeError(
                "Trajectory outputs not initialized. Call get_feature_grids first."
            )

        tracks = self.tracks.squeeze()
        if tracks.ndim == 1:
            tracks = tracks.reshape(-1, 2)
        if tracks.shape != (self.num_points, 2):
            raise RuntimeError(f"Unexpected tracks shape: {tracks.shape}")

        occlusions = self.occlusions.squeeze()
        if occlusions.dtype != torch.bool:
            occlusions = torch.sigmoid(occlusions) > 0.5

        expected_dist = torch.zeros_like(occlusions, dtype=torch.float32)

        return tracks, occlusions, expected_dist, self.new_causal_context