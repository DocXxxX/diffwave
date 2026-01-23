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


class AttrDict(dict):
  def __init__(self, *args, **kwargs):
      super(AttrDict, self).__init__(*args, **kwargs)
      self.__dict__ = self

  def override(self, attrs):
    if isinstance(attrs, dict):
      self.__dict__.update(**attrs)
    elif isinstance(attrs, (list, tuple, set)):
      for attr in attrs:
        self.override(attr)
    elif attrs is not None:
      raise NotImplementedError
    return self


params = AttrDict(
    # Training params
    batch_size=64,
    learning_rate=2e-4,
    max_grad_norm=None,

    # Data params
    sample_rate=8000,      # 爆破振动数据采样率 (原22050)
    n_mels=80,
    n_fft=1024,
    hop_samples=256,
    crop_mel_frames=62,  # Probably an error in paper.
    
    # 【新增】物理条件参数
    condition_dim=5,       # 物理参数维度：Q_max, Distance_R, Elev_Diff, Hole_Num, Delay_Int
    use_physics_condition=True,  # 使用物理参数条件

    # Model params
    residual_layers=30,
    residual_channels=64,
    dilation_cycle_length=10,
    unconditional=False,
    noise_schedule=np.linspace(1e-4, 0.05, 50).tolist(),
    inference_noise_schedule=[0.0001, 0.001, 0.01, 0.05, 0.2, 0.5],
    
    # 【新增】Mamba混合架构参数
    use_mamba=True,        # 启用Mamba模块
    mamba_d_model=64,      # Mamba隐藏层维度
    mamba_d_state=16,      # Mamba状态空间维度
    mamba_expand=2,        # Mamba扩展因子

    # 【新增】FiLM条件调制参数  
    film_hidden_dim=256,   # FiLM映射网络隐藏层维度
    diffusion_step_embed_dim_in=512,  # 扩散步嵌入维度

    # unconditional sample len
    audio_len = 8000*5, # unconditional_synthesis_samples
)
