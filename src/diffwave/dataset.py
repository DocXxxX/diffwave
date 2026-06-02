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

import json
import numpy as np
import os
import random
import torch
import torch.nn.functional as F
import re
import pandas as pd
from glob import glob
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, Tuple, Optional, List

try:
  import soundfile as sf
except ImportError:
  sf = None

try:
  import torchaudio
except Exception:
  torchaudio = None

from diffwave.preprocess_params import (
    BLAST_PARAM_COLUMNS,
    load_and_clean_params,
    validate_params,
)


BLAST_CHANNEL_COLUMNS = ['CH1', 'CH2', 'OT']
BLAST_MATCH_REPORT = 'blast_match_report.csv'
BLAST_STATS_FILE = 'blast_stats.json'
GLOBAL_ZSCORE_NORM = 'global_zscore_clip'
ROBUST_LOG_SCALE_NORM = 'robust_log_scale'
BLAST_CONDITION_LEGACY = 'legacy'
BLAST_CONDITION_ENHANCED = 'enhanced'
SCALE_EPS = 1e-8
DERIVED_CONDITION_COLUMNS = [
    'Scaled_Distance_Sqrt_Qmax',
    'Scaled_Distance_Cuberoot_Qmax',
    'Avg_Charge_Per_Hole',
    'Log_Q_max',
    'Log_Q_total',
    'Log_Distance_R',
]


def _parse_sz_filename(filepath: str) -> Optional[Dict]:
  basename = os.path.basename(filepath)
  match = re.match(r'^SZ_(\d{8})_(\d{6})_([A-Z]+)_(\d+)\.csv$', basename)
  if not match:
    return None
  return {
      'date_key': match.group(1),
      'time_key': match.group(2),
      'instrument': match.group(3),
      'monitor_id': int(match.group(4)),
  }


def _param_groups(params_df: pd.DataFrame) -> Dict[Tuple[str, int], pd.DataFrame]:
  groups = {}
  for key, group in params_df.groupby(['DateKey', 'Monitor_ID'], dropna=False):
    groups[(str(key[0]), int(key[1]))] = group
  return groups


def _param_vector(row: pd.Series) -> Optional[np.ndarray]:
  if any(pd.isna(row.get(col, np.nan)) for col in BLAST_PARAM_COLUMNS):
    return None
  return row[BLAST_PARAM_COLUMNS].to_numpy(dtype=np.float32)


def _safe_positive(value: float) -> float:
  return max(float(value), SCALE_EPS)


def _one_hot_columns(prefix: str, values: List) -> List[str]:
  return [f'{prefix}_{value}' for value in values]


def _mode_value(values: List, fallback):
  if not values:
    return fallback
  counts = {}
  for value in values:
    counts[value] = counts.get(value, 0) + 1
  return sorted(counts.items(), key=lambda item: (-item[1], str(item[0])))[0][0]


def _derived_condition_values(physical_params: np.ndarray) -> np.ndarray:
  values = {name: float(value) for name, value in zip(BLAST_PARAM_COLUMNS, physical_params)}
  q_max = _safe_positive(values['Q_max'])
  q_total = _safe_positive(values['Q_total'])
  hole_num = _safe_positive(values['Hole_Num'])
  distance = _safe_positive(values['Distance_R'])
  return np.asarray([
      distance / np.sqrt(q_max),
      distance / np.cbrt(q_max),
      q_total / hole_num,
      np.log(q_max),
      np.log(q_total),
      np.log(distance),
  ], dtype=np.float32)


def build_blast_condition_schema(records: List[Dict], condition_mode: str = BLAST_CONDITION_ENHANCED) -> Dict:
  condition_mode = condition_mode or BLAST_CONDITION_LEGACY
  if condition_mode == BLAST_CONDITION_LEGACY:
    return {
        'mode': BLAST_CONDITION_LEGACY,
        'columns': list(BLAST_PARAM_COLUMNS),
        'base_param_columns': list(BLAST_PARAM_COLUMNS),
        'derived_columns': [],
        'monitor_ids': [],
        'instruments': [],
        'default_monitor_id': None,
        'default_instrument': None,
    }
  if condition_mode != BLAST_CONDITION_ENHANCED:
    raise ValueError(f'Invalid blast_condition_mode: {condition_mode}')

  monitor_ids = sorted({int(record['monitor_id']) for record in records})
  instruments = sorted({str(record['instrument']) for record in records})
  columns = (
      list(BLAST_PARAM_COLUMNS) +
      list(DERIVED_CONDITION_COLUMNS) +
      _one_hot_columns('Monitor_ID', monitor_ids) +
      _one_hot_columns('Instrument', instruments))
  return {
      'mode': BLAST_CONDITION_ENHANCED,
      'columns': columns,
      'base_param_columns': list(BLAST_PARAM_COLUMNS),
      'derived_columns': list(DERIVED_CONDITION_COLUMNS),
      'monitor_ids': monitor_ids,
      'instruments': instruments,
      'default_monitor_id': _mode_value([int(record['monitor_id']) for record in records], monitor_ids[0] if monitor_ids else None),
      'default_instrument': _mode_value([str(record['instrument']) for record in records], instruments[0] if instruments else None),
  }


def condition_schema_from_stats(stats: Optional[Dict]) -> Dict:
  if stats is not None and 'condition_schema' in stats:
    return stats['condition_schema']
  columns = list(stats.get('param_columns', BLAST_PARAM_COLUMNS)) if stats is not None else list(BLAST_PARAM_COLUMNS)
  return {
      'mode': BLAST_CONDITION_LEGACY,
      'columns': columns,
      'base_param_columns': list(BLAST_PARAM_COLUMNS),
      'derived_columns': [],
      'monitor_ids': [],
      'instruments': [],
      'default_monitor_id': None,
      'default_instrument': None,
  }


def build_blast_condition_vector(physical_params,
                                 monitor_id=None,
                                 instrument=None,
                                 stats: Optional[Dict] = None,
                                 condition_schema: Optional[Dict] = None) -> np.ndarray:
  schema = condition_schema or condition_schema_from_stats(stats)
  physical_params = np.asarray(physical_params, dtype=np.float32)
  if physical_params.ndim != 1:
    raise ValueError(f'Expected a 1-D blast parameter vector, got shape {physical_params.shape}.')

  mode = schema.get('mode', BLAST_CONDITION_LEGACY)
  columns = schema.get('columns', BLAST_PARAM_COLUMNS)
  if mode == BLAST_CONDITION_LEGACY:
    if len(physical_params) != len(columns):
      raise ValueError(f'Expected {len(columns)} blast parameters, got {len(physical_params)}.')
    return physical_params.astype(np.float32)

  if len(physical_params) != len(BLAST_PARAM_COLUMNS):
    if len(physical_params) == len(columns):
      return physical_params.astype(np.float32)
    raise ValueError(
        f'Expected {len(BLAST_PARAM_COLUMNS)} base blast parameters or {len(columns)} enhanced conditions, got {len(physical_params)}.')

  monitor_ids = [int(value) for value in schema.get('monitor_ids', [])]
  instruments = [str(value) for value in schema.get('instruments', [])]
  if monitor_id is None:
    monitor_id = schema.get('default_monitor_id')
  if instrument is None:
    instrument = schema.get('default_instrument')
  monitor_id = int(monitor_id) if monitor_id is not None else None
  instrument = str(instrument) if instrument is not None else None

  monitor_values = np.asarray([1.0 if monitor_id == value else 0.0 for value in monitor_ids], dtype=np.float32)
  instrument_values = np.asarray([1.0 if instrument == value else 0.0 for value in instruments], dtype=np.float32)
  vector = np.concatenate([
      physical_params.astype(np.float32),
      _derived_condition_values(physical_params),
      monitor_values,
      instrument_values,
  ]).astype(np.float32)
  if len(vector) != len(columns):
    raise ValueError(f'Condition vector length {len(vector)} does not match schema length {len(columns)}.')
  return vector


def _resolve_ambiguous_param_row(matches: pd.DataFrame, meta: Dict) -> Optional[pd.Series]:
  date_key = meta['date_key']
  monitor_id = meta['monitor_id']

  target_event_id = None
  if date_key == '20251124' and monitor_id in {2, 3, 5, 6, 7}:
    target_event_id = 'BL20251124A' if meta['time_key'] < '125000' else 'BL20251124B'
  elif date_key == '20251201' and monitor_id in {1, 2}:
    target_event_id = 'BL20251201A'

  if target_event_id is None:
    return None

  resolved = matches[matches['Event_ID'] == target_event_id]
  if len(resolved) != 1:
    return None
  return resolved.iloc[0]


def build_blast_records(data_dirs: List[str], params_csv_path: str, report_path: Optional[str] = None) -> List[Dict]:
  params_df = load_and_clean_params(params_csv_path)
  validate_params(params_df)
  groups = _param_groups(params_df)

  filenames = []
  for path in data_dirs:
    filenames += glob(f'{path}/**/*.csv', recursive=True)

  records = []
  report_rows = []
  for filename in sorted(filenames):
    meta = _parse_sz_filename(filename)
    row = {
        'filename': filename,
        'status': 'skipped',
        'reason': '',
        'date_key': '',
        'monitor_id': '',
        'event_id': '',
    }

    if meta is None:
      row['reason'] = 'filename_parse_failed'
      report_rows.append(row)
      continue

    key = (meta['date_key'], meta['monitor_id'])
    row['date_key'] = meta['date_key']
    row['monitor_id'] = meta['monitor_id']
    matches = groups.get(key)
    if matches is None:
      row['reason'] = 'no_parameter_row'
      report_rows.append(row)
      continue
    if len(matches) != 1:
      param_row = _resolve_ambiguous_param_row(matches, meta)
      if param_row is None:
        row['reason'] = 'ambiguous_parameter_rows'
        row['event_id'] = '|'.join(matches['Event_ID'].astype(str).tolist())
        report_rows.append(row)
        continue
    else:
      param_row = matches.iloc[0]

    params_vector = _param_vector(param_row)
    row['event_id'] = param_row['Event_ID']
    if params_vector is None:
      row['reason'] = 'missing_parameter_value'
      report_rows.append(row)
      continue

    row['status'] = 'matched'
    row['reason'] = ''
    report_rows.append(row)
    records.append({
        'path': filename,
        'event_id': param_row['Event_ID'],
        'date_key': meta['date_key'],
        'time_key': meta['time_key'],
        'instrument': meta['instrument'],
        'monitor_id': meta['monitor_id'],
        'physical_params': params_vector,
    })

  if report_path:
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    pd.DataFrame(report_rows).to_csv(report_path, index=False, encoding='utf-8-sig')

  skipped = len(report_rows) - len(records)
  print(f"[INFO] Matched blast samples: {len(records)}/{len(report_rows)}; skipped {skipped}.")
  return records


def _load_blast_waveform(filepath: str) -> np.ndarray:
  df = pd.read_csv(filepath)
  missing_cols = [col for col in BLAST_CHANNEL_COLUMNS if col not in df.columns]
  if missing_cols:
    raise ValueError(f'{filepath} missing columns: {missing_cols}')

  audio = df[BLAST_CHANNEL_COLUMNS].apply(pd.to_numeric, errors='coerce').fillna(0.0)
  audio = audio.to_numpy(dtype=np.float32).T
  return np.nan_to_num(audio, copy=True)


def _scale_columns(channel_columns: List[str]) -> List[str]:
  return [f'log_rms_{col}' for col in channel_columns] + [
      f'log_peak_{col}' for col in channel_columns]


def _safe_std(values: np.ndarray, axis=0) -> np.ndarray:
  std = values.std(axis=axis)
  return np.maximum(std, 1e-6)


def _center_blast_audio(audio: np.ndarray) -> np.ndarray:
  center = np.median(audio, axis=1, keepdims=True)
  return audio - center


def _scale_target(audio: np.ndarray) -> np.ndarray:
  centered = _center_blast_audio(audio.astype(np.float64))
  rms = np.sqrt(np.mean(np.square(centered), axis=1))
  peak = np.max(np.abs(centered), axis=1)
  return np.concatenate([
      np.log(np.maximum(rms, SCALE_EPS)),
      np.log(np.maximum(peak, SCALE_EPS)),
  ]).astype(np.float32)


def _robust_normalize(audio: np.ndarray) -> np.ndarray:
  centered = _center_blast_audio(audio.astype(np.float64))
  abs_centered = np.abs(centered)
  clip_value = np.percentile(abs_centered, 99, axis=1, keepdims=True)
  clip_value = np.maximum(clip_value, SCALE_EPS)
  clipped = np.clip(centered, -clip_value, clip_value)
  scale = np.sqrt(np.mean(np.square(clipped), axis=1, keepdims=True))
  scale = np.maximum(scale, SCALE_EPS)
  return (centered / scale).astype(np.float32)


def _crop_or_pad_peak(audio: np.ndarray, target_len: int) -> np.ndarray:
  length = audio.shape[1]
  if length == target_len:
    return audio
  if length < target_len:
    pad_width = target_len - length
    return np.pad(audio, ((0, 0), (0, pad_width)), mode='constant')

  peak_index = int(np.argmax(np.max(np.abs(audio), axis=0)))
  start = peak_index - target_len // 2
  start = min(max(start, 0), length - target_len)
  return audio[:, start:start + target_len]


def _crop_or_pad_peak_random(audio: np.ndarray, target_len: int, peak_jitter: int) -> np.ndarray:
  length = audio.shape[1]
  if length == target_len:
    return audio
  if length < target_len:
    pad_width = target_len - length
    return np.pad(audio, ((0, 0), (0, pad_width)), mode='constant')

  peak_index = int(np.argmax(np.max(np.abs(audio), axis=0)))
  center_start = peak_index - target_len // 2
  max_start = length - target_len
  center_start = min(max(center_start, 0), max_start)
  peak_jitter = max(0, int(peak_jitter))
  start_min = max(0, center_start - peak_jitter)
  start_max = min(max_start, center_start + peak_jitter)
  start = random.randint(start_min, start_max) if start_max > start_min else center_start
  return audio[:, start:start + target_len]


def _time_shift_audio(audio: np.ndarray, max_shift: int) -> np.ndarray:
  max_shift = max(0, int(max_shift))
  if max_shift == 0:
    return audio

  shift = random.randint(-max_shift, max_shift)
  if shift == 0:
    return audio

  shifted = np.zeros_like(audio)
  if abs(shift) >= audio.shape[1]:
    return shifted
  if shift > 0:
    shifted[:, shift:] = audio[:, :-shift]
  else:
    shifted[:, :shift] = audio[:, -shift:]
  return shifted


def _augment_blast_waveform(audio: np.ndarray, params) -> np.ndarray:
  audio = _crop_or_pad_peak_random(
      audio,
      params.audio_len,
      getattr(params, 'blast_peak_jitter', 0))
  audio = _time_shift_audio(audio, getattr(params, 'blast_time_shift', 0))

  gain_min = float(getattr(params, 'blast_gain_min', 1.0))
  gain_max = float(getattr(params, 'blast_gain_max', 1.0))
  if gain_min > gain_max:
    raise ValueError(f'blast_gain_min ({gain_min}) cannot exceed blast_gain_max ({gain_max}).')
  if gain_min != 1.0 or gain_max != 1.0:
    audio = audio * random.uniform(gain_min, gain_max)
  return audio


def _compute_blast_stats(records: List[Dict], params) -> Dict:
  if not records:
    raise ValueError('No matched blast records are available for training.')

  channel_sum = np.zeros(len(BLAST_CHANNEL_COLUMNS), dtype=np.float64)
  channel_sumsq = np.zeros(len(BLAST_CHANNEL_COLUMNS), dtype=np.float64)
  channel_count = 0

  for record in records:
    audio = _crop_or_pad_peak(_load_blast_waveform(record['path']), params.audio_len)
    channel_sum += audio.sum(axis=1)
    channel_sumsq += np.square(audio.astype(np.float64)).sum(axis=1)
    channel_count += audio.shape[1]

  channel_mean = channel_sum / channel_count
  channel_var = channel_sumsq / channel_count - channel_mean ** 2
  channel_std = np.sqrt(np.maximum(channel_var, 1e-12))

  condition_mode = getattr(params, 'blast_condition_mode', BLAST_CONDITION_LEGACY)
  condition_schema = build_blast_condition_schema(records, condition_mode)
  param_values = np.stack([
      build_blast_condition_vector(
          record['physical_params'],
          monitor_id=record.get('monitor_id'),
          instrument=record.get('instrument'),
          condition_schema=condition_schema)
      for record in records
  ])
  param_mean = param_values.mean(axis=0)
  param_std = param_values.std(axis=0)
  param_std = np.maximum(param_std, 1e-6)

  scale_values = []
  for record in records:
    audio = _crop_or_pad_peak(_load_blast_waveform(record['path']), params.audio_len)
    scale_values.append(_scale_target(audio))
  scale_values = np.stack(scale_values)
  scale_mean = scale_values.mean(axis=0)
  scale_std = _safe_std(scale_values, axis=0)

  return {
      'channel_columns': BLAST_CHANNEL_COLUMNS,
      'param_columns': condition_schema['columns'],
      'base_param_columns': BLAST_PARAM_COLUMNS,
      'condition_schema': condition_schema,
      'scale_columns': _scale_columns(BLAST_CHANNEL_COLUMNS),
      'channel_mean': channel_mean.astype(float).tolist(),
      'channel_std': channel_std.astype(float).tolist(),
      'param_mean': param_mean.astype(float).tolist(),
      'param_std': param_std.astype(float).tolist(),
      'scale_mean': scale_mean.astype(float).tolist(),
      'scale_std': scale_std.astype(float).tolist(),
      'norm_mode': getattr(params, 'blast_norm_mode', GLOBAL_ZSCORE_NORM),
      'sample_rate': int(params.sample_rate),
      'audio_len': int(params.audio_len),
      'split_mode': getattr(params, 'blast_split_mode', 'record'),
  }


def save_blast_stats(stats: Dict, stats_path: str) -> None:
  os.makedirs(os.path.dirname(stats_path), exist_ok=True)
  with open(stats_path, 'w', encoding='utf-8') as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)


def load_blast_stats(stats_path: str) -> Dict:
  with open(stats_path, 'r', encoding='utf-8') as f:
    return json.load(f)


class BlastVibrationDataset(torch.utils.data.Dataset):
  """爆破振动条件数据集"""

  def __init__(self, records: List[Dict], params, stats: Dict, is_training: bool = False):
    super().__init__()
    self.records = records
    self.params = params
    self.stats = stats
    self.is_training = is_training
    self.channel_mean = np.asarray(stats['channel_mean'], dtype=np.float32)[:, None]
    self.channel_std = np.asarray(stats['channel_std'], dtype=np.float32)[:, None]
    self.param_mean = np.asarray(stats['param_mean'], dtype=np.float32)
    self.param_std = np.asarray(stats['param_std'], dtype=np.float32)
    scale_dim = len(BLAST_CHANNEL_COLUMNS) * 2
    self.scale_mean = np.asarray(stats.get('scale_mean', [0.0] * scale_dim), dtype=np.float32)
    self.scale_std = np.asarray(stats.get('scale_std', [1.0] * scale_dim), dtype=np.float32)
    self.scale_std = np.maximum(self.scale_std, 1e-6)
    self.norm_mode = getattr(params, 'blast_norm_mode', stats.get('norm_mode', GLOBAL_ZSCORE_NORM))

  def __len__(self):
    return len(self.records)

  def __getitem__(self, idx):
    record = self.records[idx]
    raw_audio = _load_blast_waveform(record['path'])
    if self.is_training and getattr(self.params, 'blast_augment', False):
      audio = _augment_blast_waveform(raw_audio, self.params)
    else:
      audio = _crop_or_pad_peak(raw_audio, self.params.audio_len)
    scale_target = _scale_target(audio)
    if self.norm_mode == ROBUST_LOG_SCALE_NORM:
      audio = _robust_normalize(audio)
    else:
      audio = (audio - self.channel_mean) / self.channel_std
    audio_clip = getattr(self.params, 'audio_clip', None)
    if audio_clip is not None and audio_clip > 0:
      audio = np.clip(audio, -audio_clip, audio_clip)
    physical_params = build_blast_condition_vector(
        record['physical_params'],
        monitor_id=record.get('monitor_id'),
        instrument=record.get('instrument'),
        stats=self.stats)
    physical_params = (physical_params - self.param_mean) / self.param_std
    scale_target = (scale_target - self.scale_mean) / self.scale_std

    return {
        'audio': torch.from_numpy(audio.astype(np.float32)),
        'physical_params': torch.from_numpy(physical_params.astype(np.float32)),
        'scale_target': torch.from_numpy(scale_target.astype(np.float32)),
        'spectrogram': None,
        'path': record['path'],
    }


class BlastCollator:
  """爆破振动数据批处理器"""

  def __init__(self, params):
    self.params = params

  def collate(self, minibatch):
    audio = torch.stack([record['audio'] for record in minibatch])
    physical_params = torch.stack([record['physical_params'] for record in minibatch])
    scale_target = torch.stack([record['scale_target'] for record in minibatch])

    return {
        'audio': audio,
        'physical_params': physical_params,
        'scale_target': scale_target,
        'spectrogram': None,
        'path': [record['path'] for record in minibatch],
    }


def _split_records_by_record(records: List[Dict], val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
  records = list(records)
  random.Random(seed).shuffle(records)
  if val_ratio <= 0 or len(records) < 2:
    return records, []
  val_count = max(1, int(round(len(records) * val_ratio)))
  val_count = min(val_count, len(records) - 1)
  return records[val_count:], records[:val_count]


def _split_records_by_event(records: List[Dict], val_ratio: float, seed: int) -> Tuple[List[Dict], List[Dict]]:
  records = list(records)
  if val_ratio <= 0 or len(records) < 2:
    return records, []

  event_ids = sorted({str(record.get('event_id', '')) for record in records})
  if len(event_ids) < 2:
    return _split_records_by_record(records, val_ratio, seed)

  random.Random(seed).shuffle(event_ids)
  target_val_count = max(1, int(round(len(records) * val_ratio)))
  val_events = set()
  val_count = 0
  for event_id in event_ids:
    event_count = sum(1 for record in records if str(record.get('event_id', '')) == event_id)
    if len(val_events) > 0 and val_count >= target_val_count:
      break
    if len(val_events) >= len(event_ids) - 1:
      break
    val_events.add(event_id)
    val_count += event_count

  train_records = [record for record in records if str(record.get('event_id', '')) not in val_events]
  val_records = [record for record in records if str(record.get('event_id', '')) in val_events]
  if not train_records or not val_records:
    return _split_records_by_record(records, val_ratio, seed)
  return train_records, val_records


def _split_records(records: List[Dict], val_ratio: float, seed: int, split_mode: str = 'record') -> Tuple[List[Dict], List[Dict]]:
  if split_mode == 'event':
    return _split_records_by_event(records, val_ratio, seed)
  if split_mode != 'record':
    raise ValueError(f'Invalid blast_split_mode: {split_mode}')
  return _split_records_by_record(records, val_ratio, seed)


def from_blast_data(data_dirs, params_csv_path, params, is_distributed=False):
  """
  创建爆破振动数据加载器

  Args:
      data_dirs: 数据目录列表
      params_csv_path: 参数表路径
      params: 模型参数
      is_distributed: 是否分布式训练
  """
  train_loader, _ = create_blast_dataloaders(
      data_dirs,
      params_csv_path,
      params,
      is_distributed=is_distributed)
  return train_loader


def create_blast_dataloaders(data_dirs, params_csv_path, params, model_dir=None, is_distributed=False):
  report_path = os.path.join(model_dir, BLAST_MATCH_REPORT) if model_dir else None
  stats_path = os.path.join(model_dir, BLAST_STATS_FILE) if model_dir else None
  records = build_blast_records(data_dirs, params_csv_path, report_path=report_path)
  train_records, val_records = _split_records(
      records,
      getattr(params, 'val_ratio', 0.15),
      getattr(params, 'split_seed', 2021),
      getattr(params, 'blast_split_mode', 'record'))
  stats = _compute_blast_stats(train_records, params)
  params.condition_dim = len(stats['param_mean'])
  if stats_path:
    save_blast_stats(stats, stats_path)

  train_dataset = BlastVibrationDataset(train_records, params, stats, is_training=True)
  val_dataset = BlastVibrationDataset(val_records, params, stats, is_training=False) if val_records else None
  train_sampler = DistributedSampler(train_dataset) if is_distributed else None
  num_workers = min(os.cpu_count() or 1, getattr(params, 'num_workers', 4))

  train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=params.batch_size,
      collate_fn=BlastCollator(params).collate,
      shuffle=(train_sampler is None),
      num_workers=num_workers,
      sampler=train_sampler,
      pin_memory=bool(getattr(params, 'pin_memory', False)),
      drop_last=True)

  val_loader = None
  if val_dataset is not None:
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=params.batch_size,
        collate_fn=BlastCollator(params).collate,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(getattr(params, 'pin_memory', False)),
        drop_last=False)

  return train_loader, val_loader


class ConditionalDataset(torch.utils.data.Dataset):
  def __init__(self, paths):
    super().__init__()
    self.filenames = []
    for path in paths:
      self.filenames += glob(f'{path}/**/*.wav', recursive=True)

  def __len__(self):
    return len(self.filenames)

  def __getitem__(self, idx):
    if sf is None:
      raise ImportError('soundfile is required to load wav datasets.')
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
    if sf is None:
      raise ImportError('soundfile is required to load wav datasets.')
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
  if torchaudio is None:
    raise ImportError('torchaudio is required to load GTZAN.')
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
