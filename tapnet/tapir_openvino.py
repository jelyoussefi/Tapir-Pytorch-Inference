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
import torch.nn.functional as F
import torch
import numpy as np
import cv2
import openvino as ov
import openvino.properties.hint as hints

class TapirInferenceOpenVINO():
    def __init__(self, model_path: str, input_resolution: tuple[int, int], num_pips_iter: int, device):
        super().__init__()
        self.input_resolution = input_resolution
        self.num_pips_iter = num_pips_iter
        self.device = device
        self.num_mixer_blocks = 12
        self.num_points = 256

        causal_state_shape = (num_pips_iter, self.num_mixer_blocks, self.num_points, 2, 512 + 2048)
        self.causal_state = np.zeros(causal_state_shape, dtype=np.float32)
        self.query_feats = np.zeros((1, self.num_points, 256), dtype=np.float32)
        self.hires_query_feats = np.zeros((1, self.num_points, 128), dtype=np.float32)

        self.ov_core = ov.Core()
        
        config = {hints.performance_mode: hints.PerformanceMode.THROUGHPUT,
                  hints.num_requests: "4"}

        self.model = self.ov_core.compile_model(model_path, device, config)

    def preprocess_frame(self, frame, resize=(256, 256)):
        input = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input = cv2.resize(input, resize)
        input = input[np.newaxis, :, :, :].astype(np.float32)
        input = input / 255 * 2 - 1
        input = input.transpose(0, 3, 1, 2)

        return input

    def map_coordinates_2d(self,feats, coordinates):
        x = torch.from_numpy(feats.transpose(0, 3, 1, 2))

        y = coordinates[:, :, None, :]
        y = 2 * (y / x.shape[2:]) - 1
        y = torch.from_numpy(np.flip(y, axis=-1).astype(np.float32))

        out = F.grid_sample(
            x, y, mode='bilinear', align_corners=False, padding_mode='border'
        )
        out = out.squeeze(dim=-1)
        out = out.permute(0, 2, 1)

        return out.cpu().numpy()


    def get_query_features(self, query_points, feature_grid, hires_feats_grid):
        position_in_grid = query_points * feature_grid.shape[1:3]
        position_in_grid_hires = query_points * hires_feats_grid.shape[1:3]

        query_feats = self.map_coordinates_2d(
            feature_grid, position_in_grid
        )
        hires_query_feats = self.map_coordinates_2d(
            hires_feats_grid, position_in_grid_hires
        )
        return query_feats, hires_query_feats

    def set_points(self, frame: np.ndarray, query_points: np.ndarray):
        """Initialize query points for tracking.

        Args:
            frame: Input frame (H, W, 3).
            query_points: Query points (N, 2).
            preprocess: Whether to preprocess the frame.
        """
        query_points = query_points.astype(np.float32)
        query_points[..., 0] = query_points[..., 0] / self.input_resolution[1]
        query_points[..., 1] = query_points[..., 1] / self.input_resolution[0]

        num_points = query_points.shape[0]
        causal_state_shape = ( self.num_pips_iter, self.num_mixer_blocks, num_points, 2, 512 + 2048)
        self.causal_state = np.zeros( causal_state_shape, dtype=np.float32)
        query_feats = np.zeros((1, num_points, 256), dtype=np.float32)
        hires_query_feats = np.zeros( (1, num_points, 128), dtype=np.float32)

        input_frame = self.preprocess_frame(frame, resize=self.input_resolution)

        _, _, _, feature_grid, hires_feats_grid = self.infer(input_frame, query_feats, hires_query_feats, self.causal_state)
        self.query_feats, self.hires_query_feats = self.get_query_features(query_points[None],feature_grid, hires_feats_grid)
           

    def infer(self,  frame, query_feats, hires_query_feats, causal_state):
        
        inputs = {
            "input_frame": frame,
            "query_feats": query_feats,
            "hires_query_feats": hires_query_feats,
            "causal_state": causal_state
        }

        output = self.model(inputs)

        tracks = output['tracks']
        visibles = output['visibles']
        feature_grid = output['feature_grid']
        hires_feats_grid = output['hires_feats_grid']
        causal_state = output['causal_state']
        
        return tracks, visibles, causal_state, feature_grid, hires_feats_grid

    
    def __call__(self, frame: np.ndarray):
        """Process a frame to track points.

        Args:
            frame: Input frame (H, W, 3) or tensor.

        Returns:
            Tuple of tracks (N, 2) and visibles (N,) as numpy arrays or tensors.
        """
        
        height, width = frame.shape[:2]
        input_frame = self.preprocess_frame(frame, resize=self.input_resolution)

        tracks, visibles, self.causal_state, _, _ = self.infer(
            input_frame, self.query_feats, self.hires_query_feats, self.causal_state
        )

        visibles = visibles.squeeze()
        tracks = tracks.squeeze()

        tracks[:, 0] = tracks[:, 0] * width / self.input_resolution[1]
        tracks[:, 1] = tracks[:, 1] * height / self.input_resolution[0]

        return (tracks, visibles)