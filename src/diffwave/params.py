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


BLAST_AUGMENT_LEVELS = {
    'off': {
        'blast_augment': False,
        'blast_peak_jitter': 0,
        'blast_time_shift': 0,
        'blast_gain_min': 1.0,
        'blast_gain_max': 1.0,
    },
    'conservative': {
        'blast_augment': True,
        'blast_peak_jitter': 2000,
        'blast_time_shift': 500,
        'blast_gain_min': 0.95,
        'blast_gain_max': 1.05,
    },
    'medium': {
        'blast_augment': True,
        'blast_peak_jitter': 4000,
        'blast_time_shift': 1000,
        'blast_gain_min': 0.90,
        'blast_gain_max': 1.10,
    },
}


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


def apply_blast_augment_level(target_params, level):
  if level is None:
    return target_params
  if level not in BLAST_AUGMENT_LEVELS:
    raise ValueError(f'Invalid blast_augment_level: {level}')
  target_params.override(BLAST_AUGMENT_LEVELS[level])
  target_params.blast_augment_level = level
  return target_params


params = AttrDict(
    # Training params
    batch_size=8,
    learning_rate=2e-4,
    max_grad_norm=None,
    loss_type='l1',
    validation_interval=500,
    validation_batches=8,
    checkpoint_interval=None,
    num_workers=4,
    pin_memory=False,
    split_seed=2021,
    val_ratio=0.15,
    use_wandb=False,
    wandb_project='blast-diffwave',
    wandb_run_name=None,

    # Data params
    sample_rate=8000,      # 爆破振动数据采样率 (原22050)
    audio_channels=3,
    audio_clip=None,
    blast_norm_mode='robust_log_scale',
    blast_condition_mode='enhanced',
    blast_split_mode='event',
    predict_amplitude_scale=True,
    lambda_scale=0.05,
    n_mels=80,
    n_fft=1024,
    hop_samples=256,
    crop_mel_frames=62,  # Probably an error in paper.
    data_format='blast_csv',
    default_blast_data_dir='dataset/SZ_blast',
    default_blast_params_csv='Final_20260411 Parameter Table.csv',
    blast_augment=True,
    blast_augment_level='conservative',
    blast_peak_jitter=2000,
    blast_time_shift=500,
    blast_gain_min=0.95,
    blast_gain_max=1.05,
    
    # 【新增】物理条件参数
    condition_dim=8,       # Updated from blast_stats for enhanced blast conditions.
    use_physics_condition=True,  # 使用物理参数条件
    use_physics_loss=False,
    lambda_mr_stft=0.01,
    lambda_band_energy=0.02,
    lambda_envelope=0.02,
    lambda_peak_rms=0.01,
    lambda_cumulative_energy=0.02,
    aux_loss_warmup_steps=2000,
    aux_loss_timestep_max_ratio=0.6,
    aux_loss_min_snr=0.1,
    gen_eval_interval=2000,
    gen_eval_subset_size=6,
    gen_eval_samples_per_condition=1,
    full_gen_eval_samples_per_condition=3,
    gen_eval_seed=20260425,
    gen_eval_fast_sampling=True,

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
    audio_len=20000,
    sample_clamp=False,
)
