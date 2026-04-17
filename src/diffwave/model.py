# Copyright 2020 LMNT, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

# 尝试导入Mamba
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False


Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


def Conv1d(*args, **kwargs):
  layer = nn.Conv1d(*args, **kwargs)
  nn.init.kaiming_normal_(layer.weight)
  return layer


@torch.jit.script
def silu(x):
  return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
  def __init__(self, max_steps):
    super().__init__()
    self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
    self.projection1 = Linear(128, 512)
    self.projection2 = Linear(512, 512)

  def forward(self, diffusion_step):
    if diffusion_step.dtype in [torch.int32, torch.int64]:
      x = self.embedding[diffusion_step]
    else:
      x = self._lerp_embedding(diffusion_step)
    x = self.projection1(x)
    x = silu(x)
    x = self.projection2(x)
    x = silu(x)
    return x

  def _lerp_embedding(self, t):
    low_idx = torch.floor(t).long()
    high_idx = torch.ceil(t).long()
    low = self.embedding[low_idx]
    high = self.embedding[high_idx]
    return low + (high - low) * (t - low_idx)

  def _build_embedding(self, max_steps):
    steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
    dims = torch.arange(64).unsqueeze(0)          # [1,64]
    table = steps * 10.0**(dims * 4.0 / 63.0)     # [T,64]
    table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
    return table


class SpectrogramUpsampler(nn.Module):
  def __init__(self, n_mels):
    super().__init__()
    self.conv1 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
    self.conv2 = ConvTranspose2d(1, 1,  [3, 32], stride=[1, 16], padding=[1, 8])

  def forward(self, x):
    x = torch.unsqueeze(x, 1)
    x = self.conv1(x)
    x = F.leaky_relu(x, 0.4)
    x = self.conv2(x)
    x = F.leaky_relu(x, 0.4)
    x = torch.squeeze(x, 1)
    return x


class BiMamba(nn.Module):
  """双向Mamba模块"""

  def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
    super().__init__()
    if not MAMBA_AVAILABLE:
      raise ImportError("mamba-ssm is required when use_mamba=True")

    self.mamba_forward = Mamba(
        d_model=d_model,
        d_state=d_state,
        d_conv=4,
        expand=expand)
    self.mamba_backward = Mamba(
        d_model=d_model,
        d_state=d_state,
        d_conv=4,
        expand=expand)
    self.norm = nn.LayerNorm(d_model)
    self.proj = nn.Linear(d_model * 2, d_model)

  def forward(self, x):
    x = x.transpose(1, 2)
    forward_out = self.mamba_forward(x)
    backward_out = self.mamba_backward(torch.flip(x, dims=[1]))
    backward_out = torch.flip(backward_out, dims=[1])
    out = self.proj(torch.cat([forward_out, backward_out], dim=-1))
    out = self.norm(out)
    return out.transpose(1, 2)


class FiLMLayer(nn.Module):
    """FiLM条件调制层"""
    
    def __init__(self, condition_dim: int, hidden_dim: int, target_dim: int):
        """
        Args:
            condition_dim: 物理参数维度 (5)
            hidden_dim: 隐藏层维度
            target_dim: 目标特征维度（通道数）
        """
        super().__init__()
        self.mapping = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.scale = nn.Linear(hidden_dim, target_dim)
        self.shift = nn.Linear(hidden_dim, target_dim)
    
    def forward(self, x, condition):
        """
        Args:
            x: [B, C, T] 特征张量
            condition: [B, condition_dim] 物理参数
        Returns:
            调制后的特征 [B, C, T]
        """
        h = self.mapping(condition)
        gamma = self.scale(h).unsqueeze(-1)  # [B, C, 1]
        beta = self.shift(h).unsqueeze(-1)   # [B, C, 1]
        return gamma * x + beta


class HybridBlock(nn.Module):
    """混合块：Dilated Conv1d (局部) + BiMamba (全局)"""
    
    def __init__(self, 
                 residual_channels: int, 
                 dilation: int,
                 n_mels: int = 80,
                 uncond: bool = False,
                 condition_dim: int = 5,
                 use_mamba: bool = True,
                 mamba_d_state: int = 16,
                 mamba_expand: int = 2,
                 film_hidden_dim: int = 256):
        super().__init__()
        
        # 局部特征：扩张卷积
        self.dilated_conv = Conv1d(
            residual_channels, 
            2 * residual_channels, 
            3, 
            padding=dilation, 
            dilation=dilation
        )
        
        # 全局特征：BiMamba（可选）
        self.use_mamba = use_mamba
        if self.use_mamba:
            self.bi_mamba = BiMamba(
                d_model=residual_channels,
                d_state=mamba_d_state,
                expand=mamba_expand
            )
            self.mamba_projection = Conv1d(residual_channels, 2 * residual_channels, 1)
        
        # 扩散步投影
        self.diffusion_projection = Linear(512, residual_channels)
        
        # FiLM条件调制
        self.film = FiLMLayer(
            condition_dim=condition_dim,
            hidden_dim=film_hidden_dim,
            target_dim=2 * residual_channels
        )

        # Mel-Spectrogram conditioner projection (keep for backward compatibility if needed, else optional)
        if not uncond: 
             self.conditioner_projection = Conv1d(n_mels, 2 * residual_channels, 1)
        else:
             self.conditioner_projection = None
        
        # 输出投影
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)
    
    def forward(self, x, diffusion_step, conditioner=None, physical_params=None):
        """
        Args:
            x: [B, C, T] 输入特征
            diffusion_step: [B, 512] 扩散步嵌入
            conditioner: Mel spectrogram (optional)
            physical_params: [B, condition_dim] 物理参数
        """
        # 扩散步注入
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step
        
        # 局部特征（扩张卷积）
        y_local = self.dilated_conv(y)
        
        # Spectrogram conditioning (if available)
        if self.conditioner_projection is not None and conditioner is not None:
             y_local = y_local + self.conditioner_projection(conditioner)

        # 全局特征（BiMamba）
        if self.use_mamba:
            y_global = self.bi_mamba(y)
            y = y_local + self.mamba_projection(y_global)
        else:
            y = y_local
        
        # FiLM条件调制
        if physical_params is not None:
            y = self.film(y, physical_params)
        
        # 门控激活
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        
        # 输出投影
        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        
        return (x + residual) / sqrt(2.0), skip


class MambaDiffWave(nn.Module):
  def __init__(self, params):
    super().__init__()
    self.params = params
    if getattr(params, 'use_mamba', False) and not MAMBA_AVAILABLE:
      raise ImportError("mamba-ssm is required when use_mamba=True. Install mamba-ssm or set --use_mamba false for smoke tests.")
    self.audio_channels = getattr(params, 'audio_channels', 1)
    self.input_projection = Conv1d(self.audio_channels, params.residual_channels, 1)
    self.diffusion_embedding = DiffusionEmbedding(len(params.noise_schedule))
    if self.params.unconditional: # use unconditional model
      self.spectrogram_upsampler = None
    else:
      self.spectrogram_upsampler = SpectrogramUpsampler(params.n_mels)

    self.residual_layers = nn.ModuleList([
        HybridBlock(
            residual_channels=params.residual_channels, 
            dilation=2**(i % params.dilation_cycle_length), 
            n_mels=params.n_mels,
            uncond=params.unconditional,
            condition_dim=params.condition_dim,
            use_mamba=params.use_mamba,
            mamba_d_state=params.mamba_d_state,
            mamba_expand=params.mamba_expand,
            film_hidden_dim=params.film_hidden_dim
        )
        for i in range(params.residual_layers)
    ])
    self.skip_projection = Conv1d(params.residual_channels, params.residual_channels, 1)
    self.output_projection = Conv1d(params.residual_channels, self.audio_channels, 1)
    nn.init.zeros_(self.output_projection.weight)

  def forward(self, audio, diffusion_step, spectrogram=None, physical_params=None):
    # spectrogram can be passed as None even if not unconditional (e.g. if using physical params instead)
    # But existing code checks self.spectrogram_upsampler
    
    if audio.dim() == 2:
      audio = audio.unsqueeze(1)
    x = self.input_projection(audio)
    x = F.relu(x)

    diffusion_step = self.diffusion_embedding(diffusion_step)
    
    if self.spectrogram_upsampler and spectrogram is not None:
      spectrogram = self.spectrogram_upsampler(spectrogram)

    skip = None
    for layer in self.residual_layers:
      x, skip_connection = layer(x, diffusion_step, conditioner=spectrogram, physical_params=physical_params)
      skip = skip_connection if skip is None else skip_connection + skip

    x = skip / sqrt(len(self.residual_layers))
    x = self.skip_projection(x)
    x = F.relu(x)
    x = self.output_projection(x)
    return x

# Alias for backward compatibility
DiffWave = MambaDiffWave
