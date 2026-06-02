import csv
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
  from scipy.signal import hilbert, stft
except Exception:
  hilbert = None
  stft = None

from diffwave.dataset import _crop_or_pad_peak, _load_blast_waveform
from diffwave.inference import generate_audio


DEFAULT_FREQ_BANDS = [0, 20, 40, 80, 130, 250, 500, 1000, 2000, 4000]
DEFAULT_STFT_FFT_SIZES = [256, 512, 1024]
GEN_SCORE_WEIGHTS = {
    'rms_error': 0.12,
    'max_abs_error': 0.08,
    'peak_to_peak_error': 0.08,
    'band_energy_error': 0.14,
    'stft_lsd_error': 0.15,
    'envelope_error': 0.14,
    'cumulative_energy_error': 0.10,
    'peak_time_error': 0.09,
    'channel_corr_error': 0.05,
    'dominant_freq_error': 0.05,
}
EPS = 1e-8


def validate_signal_config(sample_rate: int,
                           audio_len: int,
                           bands: Optional[List[float]] = None,
                           fft_sizes: Optional[List[int]] = None) -> Tuple[List[float], List[int]]:
  if sample_rate <= 0:
    raise ValueError(f'sample_rate must be positive, got {sample_rate}.')
  if audio_len <= 1:
    raise ValueError(f'audio_len must be greater than 1, got {audio_len}.')

  nyquist = sample_rate / 2
  bands = list(DEFAULT_FREQ_BANDS if bands is None else bands)
  bands = sorted(set(float(edge) for edge in bands if edge >= 0))
  if not bands or bands[0] != 0:
    bands = [0.0] + bands
  bands = [min(edge, nyquist) for edge in bands if edge <= nyquist or edge == bands[-1]]
  bands = sorted(set(bands))
  if bands[-1] < nyquist:
    bands.append(float(nyquist))
  if len(bands) < 3:
    raise ValueError(
        f'At least two valid frequency bands are required for sample_rate={sample_rate}.')

  fft_sizes = list(DEFAULT_STFT_FFT_SIZES if fft_sizes is None else fft_sizes)
  valid_fft = [int(size) for size in fft_sizes if int(size) > 1 and int(size) <= audio_len]
  if not valid_fft:
    fallback = 2 ** int(math.floor(math.log2(audio_len)))
    if fallback < 16:
      raise ValueError(f'No valid STFT n_fft for audio_len={audio_len}.')
    valid_fft = [fallback]
  for n_fft in valid_fft:
    hop = n_fft // 4
    if hop <= 0 or hop >= n_fft:
      raise ValueError(f'Invalid STFT hop for n_fft={n_fft}.')
  return bands, valid_fft


def _clip01(value) -> Tuple[float, int]:
  value = float(value)
  if not np.isfinite(value):
    return 1.0, 1
  return float(np.clip(value, 0.0, 1.0)), 0


def _log_ratio_error(generated: np.ndarray, real: np.ndarray) -> Tuple[float, int]:
  err = np.abs(np.log((generated + EPS) / (real + EPS))) / np.log(4.0)
  return _clip01(np.mean(err))


def _dominant_frequency(values: np.ndarray, sample_rate: int) -> float:
  if values.shape[-1] < 3:
    return np.nan
  centered = values - np.mean(values)
  spectrum = np.abs(np.fft.rfft(centered))
  if len(spectrum) <= 1:
    return np.nan
  freqs = np.fft.rfftfreq(len(centered), d=1.0 / sample_rate)
  return float(freqs[int(np.argmax(spectrum[1:]) + 1)])


def _band_energy(audio: np.ndarray, sample_rate: int, bands: List[float]) -> np.ndarray:
  spectrum = np.fft.rfft(audio, axis=1)
  power = np.square(np.abs(spectrum))
  freqs = np.fft.rfftfreq(audio.shape[1], d=1.0 / sample_rate)
  values = []
  for low, high in zip(bands[:-1], bands[1:]):
    if high == bands[-1]:
      mask = (freqs >= low) & (freqs <= high)
    else:
      mask = (freqs >= low) & (freqs < high)
    if not np.any(mask):
      values.append(np.zeros(audio.shape[0], dtype=np.float64))
    else:
      values.append(power[:, mask].sum(axis=1))
  energy = np.stack(values, axis=1)
  return energy / (energy.sum(axis=1, keepdims=True) + EPS)


def _stft_lsd(generated: np.ndarray, real: np.ndarray, sample_rate: int, fft_sizes: List[int]) -> Tuple[float, int]:
  if stft is None:
    return 1.0, 1
  errors = []
  for n_fft in fft_sizes:
    hop = n_fft // 4
    for channel in range(generated.shape[0]):
      _, _, gen_spec = stft(
          generated[channel],
          fs=sample_rate,
          nperseg=n_fft,
          noverlap=n_fft - hop,
          nfft=n_fft,
          boundary=None,
          padded=False)
      _, _, real_spec = stft(
          real[channel],
          fs=sample_rate,
          nperseg=n_fft,
          noverlap=n_fft - hop,
          nfft=n_fft,
          boundary=None,
          padded=False)
      gen_db = 20.0 * np.log10(np.abs(gen_spec) + EPS)
      real_db = 20.0 * np.log10(np.abs(real_spec) + EPS)
      errors.append(np.mean(np.abs(gen_db - real_db)) / 20.0)
  return _clip01(np.mean(errors))


def _envelope(audio: np.ndarray) -> np.ndarray:
  if hilbert is not None:
    return np.abs(hilbert(audio, axis=1))
  return np.abs(audio)


def _envelope_error(generated: np.ndarray, real: np.ndarray) -> Tuple[float, int]:
  gen_env = _envelope(generated)
  real_env = _envelope(real)
  errors = []
  invalid = 0
  for channel in range(generated.shape[0]):
    g = gen_env[channel] - np.mean(gen_env[channel])
    r = real_env[channel] - np.mean(real_env[channel])
    denom = np.linalg.norm(g) * np.linalg.norm(r)
    if denom <= EPS:
      corr = 1.0 if np.linalg.norm(g - r) <= EPS else 0.0
    else:
      corr = float(np.dot(g, r) / denom)
    value, bad = _clip01(1.0 - max(0.0, corr))
    errors.append(value)
    invalid += bad
  return float(np.mean(errors)), invalid


def _cumulative_energy_error(generated: np.ndarray, real: np.ndarray) -> Tuple[float, int]:
  gen_energy = np.cumsum(np.square(generated), axis=1)
  real_energy = np.cumsum(np.square(real), axis=1)
  gen_total = gen_energy[:, -1:]
  real_total = real_energy[:, -1:]
  invalid = int(np.any(gen_total <= EPS) or np.any(real_total <= EPS))
  gen_energy = gen_energy / (gen_total + EPS)
  real_energy = real_energy / (real_total + EPS)
  value, bad = _clip01(np.mean(np.abs(gen_energy - real_energy)))
  return value, bad + invalid


def _peak_time(audio: np.ndarray, sample_rate: int) -> float:
  index = int(np.argmax(np.max(np.abs(audio), axis=0)))
  return index / sample_rate


def _channel_corr(audio: np.ndarray) -> np.ndarray:
  if audio.shape[0] < 2:
    return np.eye(audio.shape[0])
  corr = np.corrcoef(audio)
  return np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)


def compute_generation_metrics(generated: np.ndarray,
                               real: np.ndarray,
                               sample_rate: int,
                               bands: Optional[List[float]] = None,
                               fft_sizes: Optional[List[int]] = None) -> Dict[str, float]:
  if generated.shape != real.shape:
    raise ValueError(f'generated shape {generated.shape} does not match real shape {real.shape}.')
  bands, fft_sizes = validate_signal_config(sample_rate, real.shape[1], bands, fft_sizes)

  generated = np.nan_to_num(generated.astype(np.float64), copy=False)
  real = np.nan_to_num(real.astype(np.float64), copy=False)
  invalid_count = 0

  gen_rms = np.sqrt(np.mean(np.square(generated), axis=1))
  real_rms = np.sqrt(np.mean(np.square(real), axis=1))
  rms_error, bad = _log_ratio_error(gen_rms, real_rms)
  invalid_count += bad

  gen_max_abs = np.max(np.abs(generated), axis=1)
  real_max_abs = np.max(np.abs(real), axis=1)
  max_abs_error, bad = _log_ratio_error(gen_max_abs, real_max_abs)
  invalid_count += bad

  gen_ptp = np.ptp(generated, axis=1)
  real_ptp = np.ptp(real, axis=1)
  peak_to_peak_error, bad = _log_ratio_error(gen_ptp, real_ptp)
  invalid_count += bad

  gen_dom = np.array([_dominant_frequency(channel, sample_rate) for channel in generated])
  real_dom = np.array([_dominant_frequency(channel, sample_rate) for channel in real])
  dom_den = np.maximum(real_dom, 20.0)
  dominant_frequency_error, bad = _clip01(np.mean(np.abs(gen_dom - real_dom) / (dom_den + EPS)))
  invalid_count += bad

  gen_band = _band_energy(generated, sample_rate, bands)
  real_band = _band_energy(real, sample_rate, bands)
  band_energy_error, bad = _clip01(np.mean(0.5 * np.sum(np.abs(gen_band - real_band), axis=1)))
  invalid_count += bad

  stft_lsd_error, bad = _stft_lsd(generated, real, sample_rate, fft_sizes)
  invalid_count += bad

  envelope_error, bad = _envelope_error(generated, real)
  invalid_count += bad

  cumulative_energy_error, bad = _cumulative_energy_error(generated, real)
  invalid_count += bad

  peak_time_error, bad = _clip01(abs(_peak_time(generated, sample_rate) - _peak_time(real, sample_rate)) / 0.25)
  invalid_count += bad

  gen_corr = _channel_corr(generated)
  real_corr = _channel_corr(real)
  if generated.shape[0] >= 2:
    upper = np.triu_indices(generated.shape[0], k=1)
    corr_error = np.mean(np.abs(gen_corr[upper] - real_corr[upper])) / 2.0
  else:
    corr_error = 0.0
  channel_corr_error, bad = _clip01(corr_error)
  invalid_count += bad

  metrics = {
      'rms_error': rms_error,
      'max_abs_error': max_abs_error,
      'peak_to_peak_error': peak_to_peak_error,
      'band_energy_error': band_energy_error,
      'stft_lsd_error': stft_lsd_error,
      'envelope_error': envelope_error,
      'cumulative_energy_error': cumulative_energy_error,
      'peak_time_error': peak_time_error,
      'channel_corr_error': channel_corr_error,
      'dominant_freq_error': dominant_frequency_error,
      'gen_invalid_metric_count': float(invalid_count),
  }
  score = sum(GEN_SCORE_WEIGHTS[name] * metrics[name] for name in GEN_SCORE_WEIGHTS)
  metrics['gen_score'] = float(np.clip(score, 0.0, 1.0))
  return metrics


class BlastGenerationEvaluator:
  def __init__(self, records: List[Dict], stats: Dict, params):
    self.records = list(records)
    self.stats = stats
    self.params = params
    self.sample_rate = int(stats.get('sample_rate', params.sample_rate))
    self.audio_len = int(stats.get('audio_len', params.audio_len))
    if self.sample_rate != int(params.sample_rate):
      raise ValueError(
          f"sample_rate mismatch: params={params.sample_rate}, stats={self.sample_rate}.")
    self.bands, self.fft_sizes = validate_signal_config(
        self.sample_rate,
        self.audio_len,
        getattr(params, 'gen_eval_freq_bands', None),
        getattr(params, 'gen_eval_stft_fft_sizes', None))

  def _subset(self, subset_size: Optional[int], seed: int) -> List[Dict]:
    if subset_size is None or subset_size <= 0 or subset_size >= len(self.records):
      return list(self.records)
    rng = np.random.default_rng(seed)
    indices = sorted(rng.choice(len(self.records), size=subset_size, replace=False).tolist())
    return [self.records[index] for index in indices]

  def evaluate_model(self,
                     model,
                     device,
                     subset_size: Optional[int],
                     samples_per_condition: int,
                     fast_sampling: bool,
                     seed: int,
                     output_csv: Optional[str] = None) -> Dict[str, float]:
    if not self.records:
      return {}

    was_training = model.training
    model.eval()
    rows = []
    condition_means = []
    condition_bests = []
    records = self._subset(subset_size, seed)
    samples_per_condition = max(1, int(samples_per_condition))

    with torch.no_grad():
      for record_index, record in enumerate(records):
        real = _crop_or_pad_peak(_load_blast_waveform(record['path']), self.audio_len)
        sample_metrics = []
        for sample_index in range(samples_per_condition):
          generator = None
          if device.type == 'cuda':
            generator = torch.Generator(device=device)
          else:
            generator = torch.Generator()
          generator.manual_seed(int(seed) + record_index * 1009 + sample_index)
          generated, _ = generate_audio(
              model,
              physical_params=record['physical_params'],
              stats=self.stats,
              device=device,
              fast_sampling=fast_sampling,
              monitor_id=record.get('monitor_id'),
              instrument=record.get('instrument'),
              generator=generator)
          generated = generated.detach().cpu().numpy()[0]
          metrics = compute_generation_metrics(
              generated,
              real,
              self.sample_rate,
              bands=self.bands,
              fft_sizes=self.fft_sizes)
          sample_metrics.append(metrics)
          row = {
              'path': record['path'],
              'event_id': record.get('event_id', ''),
              'monitor_id': record.get('monitor_id', ''),
              'sample_index': sample_index,
          }
          row.update(metrics)
          rows.append(row)
        metric_names = list(sample_metrics[0].keys())
        mean_metrics = {
            name: float(np.mean([metrics[name] for metrics in sample_metrics]))
            for name in metric_names
        }
        best_metrics = min(sample_metrics, key=lambda metrics: metrics['gen_score'])
        condition_means.append(mean_metrics)
        condition_bests.append(best_metrics)

    if was_training:
      model.train()

    if output_csv:
      os.makedirs(os.path.dirname(output_csv), exist_ok=True)
      fieldnames = list(rows[0].keys()) if rows else []
      with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if not condition_means:
      return {}
    metric_names = list(condition_means[0].keys())
    output = {}
    for name in metric_names:
      mean_values = np.asarray([metrics[name] for metrics in condition_means], dtype=np.float64)
      best_values = np.asarray([metrics[name] for metrics in condition_bests], dtype=np.float64)
      output[f'{name}_mean'] = float(np.mean(mean_values))
      output[f'{name}_best'] = float(np.mean(best_values))
    return output
