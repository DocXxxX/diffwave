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
import os
import random
import torch
import torch.nn.functional as F
import soundfile as sf
import re
import pandas as pd
from scipy import signal
from glob import glob
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, Tuple, Optional

from diffwave.preprocess_params import preprocess_params


class BlastVibrationDataset(torch.utils.data.Dataset):
    """爆破振动条件数据集"""
    
    def __init__(self, 
                 data_dirs: list, 
                 params_csv_path: str,
                 sample_rate: int = 8000):
        """
        初始化数据集
        
        Args:
            data_dirs: 波形文件所在目录列表
            params_csv_path: 监测参数表CSV路径
            sample_rate: 采样率
        """
        super().__init__()
        self.sample_rate = sample_rate
        
        # 加载清洗后的参数字典
        self.params_dict = preprocess_params(params_csv_path)
        
        # 遍历数据文件夹，获取所有波形文件
        self.filenames = []
        for path in data_dirs:
            # 查找所有CSV文件
            self.filenames += glob(f'{path}/**/*.csv', recursive=True)
        
        # 过滤掉参数表中不存在的文件
        self.valid_files = []
        for f in self.filenames:
            key = self._parse_filename(f)
            if key and key in self.params_dict:
                self.valid_files.append(f)
            else:
                # 仅在调试时打印，避免刷屏
                pass 
                # print(f"[警告] 跳过文件 {f}：参数表中未找到对应记录")
        
        print(f"[信息] 有效样本数：{len(self.valid_files)}/{len(self.filenames)}")
    
    def _parse_filename(self, filepath: str) -> Optional[Tuple[str, int]]:
        """
        从文件名解析Event_ID和Monitor_ID
        
        Args:
            filepath: 文件路径，如 BL20251106A_ID6.csv
            
        Returns:
            (event_id, monitor_id) 元组或None
        """
        basename = os.path.basename(filepath)
        # 匹配模式：BL20251106A_ID6.csv
        match = re.match(r'(.+)_ID(\d+)\.csv', basename)
        if match:
            event_id = match.group(1)
            monitor_id = int(match.group(2))
            return (event_id, monitor_id)
        return None
    
    def _load_waveform(self, filepath: str) -> np.ndarray:
        """
        加载并预处理波形数据
        
        Args:
            filepath: CSV文件路径
            
        Returns:
            预处理后的波形数组
        """
        try:
            df = pd.read_csv(filepath)
            
            # 提取Z列（垂向分量）或计算矢量和
            if 'Z' in df.columns:
                waveform = df['Z'].values.astype(np.float32)
            elif all(col in df.columns for col in ['X', 'Y', 'Z']):
                # 计算三向矢量和
                waveform = np.sqrt(
                    df['X'].values**2 + df['Y'].values**2 + df['Z'].values**2
                ).astype(np.float32)
            else:
                # 使用第一列作为波形数据
                waveform = df.iloc[:, 0].values.astype(np.float32)
            
            # 去趋势
            waveform = signal.detrend(waveform)
            
            # Z-Score归一化
            mean = np.mean(waveform)
            std = np.std(waveform)
            if std > 1e-8:
                waveform = (waveform - mean) / std
                
            return waveform
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            return np.zeros(self.sample_rate, dtype=np.float32) # Return silent audio on error
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        filepath = self.valid_files[idx]
        
        # 解析文件名获取Key
        key = self._parse_filename(filepath)
        
        # 获取物理参数
        physical_params = self.params_dict[key]
        
        # 加载波形数据
        waveform = self._load_waveform(filepath)
        
        return {
            'audio': torch.from_numpy(waveform),
            'physical_params': torch.from_numpy(physical_params),
            'spectrogram': torch.zeros(1) # 不使用频谱图条件，提供占位符
        }


class BlastCollator:
    """爆破振动数据批处理器"""
    
    def __init__(self, params):
        self.params = params
    
    def collate(self, minibatch):
        target_len = self.params.audio_len
        
        valid_records = []
        for record in minibatch:
            audio = record['audio']
            
            # 处理长度
            if len(audio) < target_len:
                # 填充
                audio = F.pad(audio, (0, target_len - len(audio)), mode='constant', value=0)
            elif len(audio) > target_len:
                # 随机裁剪
                start = random.randint(0, len(audio) - target_len)
                audio = audio[start:start + target_len]
            
            record['audio'] = audio
            valid_records.append(record)
        
        if not valid_records:
            # Should not happen in normal training loop if dataset is clean
            # Return empty structure or handle appropriately
            return {} 
        
        audio = torch.stack([r['audio'] for r in valid_records])
        physical_params = torch.stack([r['physical_params'] for r in valid_records])
        
        return {
            'audio': audio,
            'physical_params': physical_params,
            'spectrogram': None
        }


def from_blast_data(data_dirs, params_csv_path, params, is_distributed=False):
    """
    创建爆破振动数据加载器
    
    Args:
        data_dirs: 数据目录列表
        params_csv_path: 参数表路径
        params: 模型参数
        is_distributed: 是否分布式训练
    """
    dataset = BlastVibrationDataset(data_dirs, params_csv_path, params.sample_rate)
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=params.batch_size,
        collate_fn=BlastCollator(params).collate,
        shuffle=not is_distributed,
        num_workers=min(os.cpu_count(), 4),
        sampler=DistributedSampler(dataset) if is_distributed else None,
        pin_memory=True,
        drop_last=True
    )


class ConditionalDataset(torch.utils.data.Dataset):
  def __init__(self, paths):
    super().__init__()
    self.filenames = []
    for path in paths:
      self.filenames += glob(f'{path}/**/*.wav', recursive=True)

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    audio_filename = self.filenames[idx]
    spec_filename = f'{audio_filename}.spec.npy'
    signal, _ = sf.read(audio_filename, dtype='float32')
    spectrogram = np.load(spec_filename)
    return {
        'audio': torch.from_numpy(signal),
        'spectrogram': spectrogram.T
    }


class UnconditionalDataset(torch.utils.data.Dataset):
  def __init__(self, paths):
    super().__init__()
    self.filenames = []
    for path in paths:
      self.filenames += glob(f'{path}/**/*.wav', recursive=True)

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    audio_filename = self.filenames[idx]
    spec_filename = f'{audio_filename}.spec.npy'
    signal, _ = sf.read(audio_filename, dtype='float32')
    return {
        'audio': torch.from_numpy(signal),
        'spectrogram': None
    }


class Collator:
  def __init__(self, params):
    self.params = params

  def collate(self, minibatch):
    samples_per_frame = self.params.hop_samples
    for record in minibatch:
      if self.params.unconditional:
          # Filter out records that aren't long enough.
          if len(record['audio']) < self.params.audio_len:
            del record['spectrogram']
            del record['audio']
            continue

          start = random.randint(0, record['audio'].shape[-1] - self.params.audio_len)
          end = start + self.params.audio_len
          record['audio'] = record['audio'][start:end]
          record['audio'] = np.pad(record['audio'], (0, (end - start) - len(record['audio'])), mode='constant')
      else:
          # Filter out records that aren't long enough.
          if len(record['spectrogram']) < self.params.crop_mel_frames:
            del record['spectrogram']
            del record['audio']
            continue

          start = random.randint(0, record['spectrogram'].shape[0] - self.params.crop_mel_frames)
          end = start + self.params.crop_mel_frames
          record['spectrogram'] = record['spectrogram'][start:end].T

          start *= samples_per_frame
          end *= samples_per_frame
          record['audio'] = record['audio'][start:end]
          record['audio'] = np.pad(record['audio'], (0, (end-start) - len(record['audio'])), mode='constant')

    audio = np.stack([record['audio'] for record in minibatch if 'audio' in record])
    if self.params.unconditional:
        return {
            'audio': torch.from_numpy(audio),
            'spectrogram': None,
        }
    spectrogram = np.stack([record['spectrogram'] for record in minibatch if 'spectrogram' in record])
    return {
        'audio': torch.from_numpy(audio),
        'spectrogram': torch.from_numpy(spectrogram),
    }

  # for gtzan
  def collate_gtzan(self, minibatch):
    ldata = []
    mean_audio_len = self.params.audio_len # change to fit in gpu memory
    # audio total generated time = audio_len * sample_rate
    # GTZAN statistics
    # max len audio 675808; min len audio sample 660000; mean len audio sample 662117
    # max audio sample 1; min audio sample -1; mean audio sample -0.0010 (normalized)
    # sample rate of all is 22050
    for data in minibatch:
      if data[0].shape[-1] < mean_audio_len:  # pad
        data_audio = F.pad(data[0], (0, mean_audio_len - data[0].shape[-1]), mode='constant', value=0)
      elif data[0].shape[-1] > mean_audio_len:  # crop
        start = random.randint(0, data[0].shape[-1] - mean_audio_len)
        end = start + mean_audio_len
        data_audio = data[0][:, start:end]
      else:
        data_audio = data[0]
      ldata.append(data_audio)
    audio = torch.cat(ldata, dim=0)
    return {
          'audio': audio,
          'spectrogram': None,
    }


def from_path(data_dirs, params, is_distributed=False):
  if params.unconditional:
    dataset = UnconditionalDataset(data_dirs)
  else:#with condition
    dataset = ConditionalDataset(data_dirs)
  return torch.utils.data.DataLoader(
      dataset,
      batch_size=params.batch_size,
      collate_fn=Collator(params).collate,
      shuffle=not is_distributed,
      num_workers=os.cpu_count(),
      sampler=DistributedSampler(dataset) if is_distributed else None,
      pin_memory=True,
      drop_last=True)


def from_gtzan(params, is_distributed=False):
  dataset = torchaudio.datasets.GTZAN('./data', download=True)
  return torch.utils.data.DataLoader(
      dataset,
      batch_size=params.batch_size,
      collate_fn=Collator(params).collate_gtzan,
      shuffle=not is_distributed,
      num_workers=os.cpu_count(),
      sampler=DistributedSampler(dataset) if is_distributed else None,
      pin_memory=True,
      drop_last=True)
