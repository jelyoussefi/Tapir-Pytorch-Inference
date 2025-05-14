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

"""Optimized PyTorch neural network definitions for Intel GPUs with OpenVINO and IPEX."""

from typing import Sequence, Union

import torch
from torch import nn
import torch.nn.functional as F
import intel_extension_for_pytorch as ipex

def fold_instance_norm(conv: nn.Conv2d, norm: nn.InstanceNorm2d):
    """Folds InstanceNorm2d into Conv2d by adjusting weights and biases, if possible."""
    # Check if folding is possible (requires running statistics)
    if not norm.track_running_stats or norm.running_var is None or norm.running_mean is None:
        return conv, norm  # Return original conv and norm if folding is not possible

    # Extract InstanceNorm parameters
    gamma = norm.weight.data if norm.affine else torch.ones_like(norm.running_mean)
    beta = norm.bias.data if norm.affine else torch.zeros_like(norm.running_mean)
    mean = norm.running_mean
    var = norm.running_var
    eps = norm.eps

    # Compute scaling factor for folding
    scale = gamma / torch.sqrt(var + eps)

    # Adjust Conv2d weights
    weight = conv.weight.data
    new_weight = weight * scale.view(-1, 1, 1, 1)
    conv.weight.data = new_weight

    # Note: No bias adjustment, as Conv2d layers have bias=False
    return conv, None  # Return folded conv and None to indicate norm is folded

class ExtraConvBlock(nn.Module):
    """Additional convolution block."""
    def __init__(
            self,
            channel_dim,
            channel_multiplier,
    ):
        super().__init__()
        self.channel_dim = channel_dim
        self.channel_multiplier = channel_multiplier

        self.layer_norm = nn.LayerNorm(
            normalized_shape=channel_dim, elementwise_affine=True, bias=True
        )
        self.conv = nn.Conv2d(
            self.channel_dim,
            self.channel_dim * self.channel_multiplier,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv_1 = nn.Conv2d(
            self.channel_dim * self.channel_multiplier,
            self.channel_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        x = self.layer_norm(x)
        x = x.permute(0, 3, 1, 2)
        res = self.conv(x)
        res = F.gelu(res, approximate='tanh')
        x += self.conv_1(res)
        x = x.permute(0, 2, 3, 1)
        return x

class ExtraConvs(nn.Module):
    """Additional CNN."""
    def __init__(
            self,
            num_layers=5,
            channel_dim=256,
            channel_multiplier=4,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.channel_dim = channel_dim
        self.channel_multiplier = channel_multiplier

        self.blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            self.blocks.append(
                ExtraConvBlock(self.channel_dim, self.channel_multiplier)
            )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class ConvChannelsMixer(nn.Module):
    """Linear activation block for PIPs's MLP Mixer."""
    def __init__(self, in_channels):
        super().__init__()
        self.mlp2_up = nn.Linear(in_channels, in_channels * 4)
        self.mlp2_down = nn.Linear(in_channels * 4, in_channels)

    def forward(self, x):
        x = self.mlp2_up(x)
        x = F.gelu(x, approximate='tanh')
        x = self.mlp2_down(x)
        return x

class PIPsConvBlock(nn.Module):
    """Convolutional block for PIPs's MLP Mixer."""
    def __init__(
            self, in_channels, kernel_shape=3, use_causal_conv=False, block_idx=None
    ):
        super().__init__()
        self.use_causal_conv = use_causal_conv
        self.block_name = f'block_{block_idx}'
        self.kernel_shape = kernel_shape
        self.step1_size = 512
        self.step2_size = 2048

        self.layer_norm = nn.LayerNorm(
            normalized_shape=in_channels, elementwise_affine=True, bias=False
        )
        self.mlp1_up = nn.Conv1d(
            in_channels,
            in_channels * 4,
            kernel_shape,
            stride=1,
            padding=0 if self.use_causal_conv else 1,
            groups=in_channels,
        )
        self.mlp1_up_1 = nn.Conv1d(
            in_channels * 4,
            in_channels * 4,
            kernel_shape,
            stride=1,
            padding=0 if self.use_causal_conv else 1,
            groups=in_channels * 4,
        )
        self.layer_norm_1 = nn.LayerNorm(
            normalized_shape=in_channels, elementwise_affine=True, bias=False
        )
        self.conv_channels_mixer = ConvChannelsMixer(in_channels)

    def process_step1(self, x, causal_context_1):
        x = self.layer_norm(x)
        x = torch.cat([causal_context_1, x], dim=-2)
        new_causal_context = x[..., -(self.kernel_shape - 1):, :]
        x = x.permute(0, 2, 1)
        x = F.pad(x, (2, 0))
        x = self.mlp1_up(x)
        x = F.gelu(x, approximate='tanh')
        return x, new_causal_context

    def process_step2(self, x, causal_context_2, new_causal_context):
        x = x.permute(0, 2, 1)
        num_extra = causal_context_2.shape[-2]
        x = torch.cat([causal_context_2, x[..., num_extra:, :]], dim=-2)
        new_causal_context = torch.cat([new_causal_context, x[..., -(self.kernel_shape - 1):, :]], dim=-1)
        x = x.permute(0, 2, 1)
        x = F.pad(x, (2, 0))
        x = self.mlp1_up_1(x)
        x = x.permute(0, 2, 1)
        x = x[..., num_extra:, :]
        x = x.view(*x.shape[:-1], -1, 4).sum(dim=-1)
        return x, new_causal_context

    def forward(self, x, causal_context_1, causal_context_2):
        to_skip = x
        x, new_causal_context = self.process_step1(x, causal_context_1)
        x, new_causal_context = self.process_step2(x, causal_context_2, new_causal_context)
        x = x + to_skip
        to_skip = x
        x = self.layer_norm_1(x)
        x = self.conv_channels_mixer(x)
        x = x + to_skip
        return x, new_causal_context

class PIPSMLPMixer(nn.Module):
    """Depthwise-conv version of PIPs's MLP Mixer."""
    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            hidden_dim: int = 512,
            num_blocks: int = 12,
            kernel_shape: int = 3,
            use_causal_conv: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.use_causal_conv = use_causal_conv
        self.linear = nn.Linear(input_channels, self.hidden_dim)
        self.layer_norm = nn.LayerNorm(
            normalized_shape=hidden_dim, elementwise_affine=True, bias=False
        )
        self.linear_1 = nn.Linear(hidden_dim, output_channels)
        self.blocks = nn.ModuleList([
            PIPsConvBlock(
                hidden_dim, kernel_shape, self.use_causal_conv, block_idx=i
            )
            for i in range(num_blocks)
        ])

    def forward(self, x, causal_context):
        x = self.linear(x)
        step1_size = self.blocks[0].step1_size
        causal_context1 = causal_context[..., :step1_size]
        causal_context2 = causal_context[..., step1_size:]
        for i, block in enumerate(self.blocks):
            x, causal_context[i,...] = block(x, causal_context1[i,...], causal_context2[i,...])
        x = self.layer_norm(x)
        x = self.linear_1(x)
        return x, causal_context

class BlockV2(nn.Module):
    """Optimized ResNet V2 block with conditional InstanceNorm folding."""
    def __init__(
            self,
            channels_in: int,
            channels_out: int,
            stride: Union[int, Sequence[int]],
            use_projection: bool,
    ):
        super().__init__()
        self.padding = (1, 1, 1, 1) if stride == 1 else (0, 2, 0, 2)
        if stride not in [1, 2]:
            raise ValueError("Stride must be 1 or 2")

        self.use_projection = use_projection
        if self.use_projection:
            self.proj_conv = nn.Conv2d(
                in_channels=channels_in,
                out_channels=channels_out,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )

        self.conv_0 = nn.Conv2d(
            in_channels=channels_in,
            out_channels=channels_out,
            kernel_size=3,
            stride=stride,
            padding=0,
            bias=False,
        )
        self.conv_1 = nn.Conv2d(
            in_channels=channels_out,
            out_channels=channels_out,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        self.bn_0 = nn.InstanceNorm2d(
            num_features=channels_in,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=False,
        )
        self.bn_1 = nn.InstanceNorm2d(
            num_features=channels_out,
            eps=1e-05,
            momentum=0.1,
            affine=True,
            track_running_stats=False,
        )

        # Attempt to fold InstanceNorm into Conv2d
        self.conv_0, self.bn_0 = fold_instance_norm(self.conv_0, self.bn_0)
        self.conv_1, self.bn_1 = fold_instance_norm(self.conv_1, self.bn_1)

    def forward(self, inputs):
        x = shortcut = inputs

        if self.bn_0 is not None:
            x = self.bn_0(x)
        x = torch.relu(x)
        if self.use_projection:
            shortcut = self.proj_conv(x)
        x = self.conv_0(F.pad(x, self.padding))

        if self.bn_1 is not None:
            x = self.bn_1(x)
        x = torch.relu(x)
        x = self.conv_1(x)

        return x + shortcut

class BlockGroup(nn.Module):
    """Higher level block for ResNet implementation."""
    def __init__(
            self,
            channels_in: int,
            channels_out: int,
            num_blocks: int,
            stride: Union[int, Sequence[int]],
            use_projection: bool,
    ):
        super().__init__()
        blocks = []
        for i in range(num_blocks):
            blocks.append(
                BlockV2(
                    channels_in=channels_in if i == 0 else channels_out,
                    channels_out=channels_out,
                    stride=(1 if i else stride),
                    use_projection=(i == 0 and use_projection),
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, inputs):
        out = inputs
        for block in self.blocks:
            out = block(out)
        return out

class ResNet(nn.Module):
    """Optimized ResNet model."""
    def __init__(
            self,
            blocks_per_group: Sequence[int],
            channels_per_group: Sequence[int] = (64, 128, 256, 512),
            use_projection: Sequence[bool] = (True, True, True, True),
            strides: Sequence[int] = (1, 2, 2, 2),
    ):
        super().__init__()
        self.initial_conv = nn.Conv2d(
            in_channels=3,
            out_channels=channels_per_group[0],
            kernel_size=(7, 7),
            stride=2,
            padding=0,
            bias=False,
        )

        block_groups = []
        for i, _ in enumerate(strides):
            block_groups.append(
                BlockGroup(
                    channels_in=channels_per_group[i - 1] if i > 0 else 64,
                    channels_out=channels_per_group[i],
                    num_blocks=blocks_per_group[i],
                    stride=strides[i],
                    use_projection=use_projection[i],
                )
            )
        self.block_groups = nn.ModuleList(block_groups)

    def forward(self, inputs):
        result = {}
        out = inputs
        out = self.initial_conv(F.pad(out, (2, 4, 2, 4)))
        result['initial_conv'] = out

        for block_id, block_group in enumerate(self.block_groups):
            out = block_group(out)
            result[f'resnet_unit_{block_id}'] = out
        return result

    def optimize_for_ipex(self, dtype=torch.float32):
        """Optimize the model using IPEX."""
        self.eval()
        self.to(dtype)
        optimized_model = ipex.optimize(self, dtype=dtype)
        return optimized_model