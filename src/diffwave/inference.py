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

import os

import numpy as np
import torch

from argparse import ArgumentParser

try:
  import soundfile as sf
except ImportError:
  sf = None

from diffwave.dataset import BLAST_CHANNEL_COLUMNS, BLAST_STATS_FILE, load_blast_stats
from diffwave.params import AttrDict, params as base_params
from diffwave.model import DiffWave


models = {}


def _default_device(device):
  if isinstance(device, torch.device):
    return device
  if device:
    return torch.device(device)
  return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _checkpoint_path(model_dir):
  if os.path.isdir(model_dir):
    return os.path.join(model_dir, 'weights.pt')
  return model_dir


def _checkpoint_dir(model_dir):
  return model_dir if os.path.isdir(model_dir) else os.path.dirname(model_dir)


def _load_stats(model_dir, stats_path=None):
  stats_path = stats_path or os.path.join(_checkpoint_dir(model_dir), BLAST_STATS_FILE)
  if stats_path and os.path.exists(stats_path):
    return load_blast_stats(stats_path)
  return None


def _checkpoint_params(checkpoint, overrides=None):
  model_params = AttrDict(base_params)
  checkpoint_params = checkpoint.get('params') or {}
  model_params.override(checkpoint_params)
  state = checkpoint.get('model', {})
  if 'input_projection.weight' in state:
    model_params.audio_channels = int(state['input_projection.weight'].shape[1])
  if 'output_projection.weight' in state:
    model_params.audio_channels = int(state['output_projection.weight'].shape[0])
  if 'predict_amplitude_scale' not in checkpoint_params and not any(
      key.startswith('scale_predictor.') for key in state):
    model_params.predict_amplitude_scale = False
  model_params.override(overrides)
  return model_params


def _prepare_physical_params(physical_params, stats, device):
  if physical_params is None:
    return None, 1
  if isinstance(physical_params, (list, tuple)):
    physical_params = torch.tensor(physical_params).float()
  if isinstance(physical_params, np.ndarray):
    physical_params = torch.from_numpy(physical_params).float()
  if physical_params.dim() == 1:
    physical_params = physical_params.unsqueeze(0)
  if stats is not None:
    mean = torch.tensor(stats['param_mean']).float()
    std = torch.tensor(stats['param_std']).float()
    if physical_params.shape[-1] != mean.shape[0]:
      raise ValueError(f'Expected {mean.shape[0]} blast parameters, got {physical_params.shape[-1]}.')
    physical_params = (physical_params - mean) / std
  return physical_params.to(device), physical_params.shape[0]


def _denormalize_scale(predicted_scale, stats, channels):
  if predicted_scale is None or stats is None:
    return None
  if 'scale_mean' not in stats or 'scale_std' not in stats:
    return None
  mean = torch.tensor(stats['scale_mean'], device=predicted_scale.device, dtype=predicted_scale.dtype).view(1, -1)
  std = torch.tensor(stats['scale_std'], device=predicted_scale.device, dtype=predicted_scale.dtype).view(1, -1)
  raw_scale = predicted_scale * std + mean
  log_rms = raw_scale[:, :channels]
  return torch.exp(log_rms).view(predicted_scale.shape[0], channels, 1)


def _denormalize_audio(audio, stats, model=None, physical_params=None):
  if stats is None:
    return audio
  norm_mode = stats.get('norm_mode', 'global_zscore_clip')
  if norm_mode == 'robust_log_scale':
    scale = None
    if model is not None and hasattr(model, 'predict_scale') and physical_params is not None:
      scale = _denormalize_scale(model.predict_scale(physical_params), stats, audio.shape[1])
    if scale is None:
      if 'scale_mean' in stats:
        mean = torch.tensor(stats['scale_mean'], device=audio.device, dtype=audio.dtype).view(1, -1)
        scale = torch.exp(mean[:, :audio.shape[1]]).view(1, audio.shape[1], 1)
      else:
        scale = torch.ones(1, audio.shape[1], 1, device=audio.device, dtype=audio.dtype)
    return audio * scale
  mean = torch.tensor(stats['channel_mean'], device=audio.device, dtype=audio.dtype).view(1, -1, 1)
  std = torch.tensor(stats['channel_std'], device=audio.device, dtype=audio.dtype).view(1, -1, 1)
  return audio * std + mean


def _load_model(model_dir, params, device):
  checkpoint = torch.load(_checkpoint_path(model_dir), map_location=device)
  model_params = _checkpoint_params(checkpoint, params)
  model = DiffWave(model_params).to(device)
  missing, unexpected = model.load_state_dict(checkpoint['model'], strict=False)
  scale_missing = [key for key in missing if key.startswith('scale_predictor.')]
  other_missing = [key for key in missing if not key.startswith('scale_predictor.')]
  if other_missing or unexpected:
    raise RuntimeError(
        f'Checkpoint/model mismatch. Missing={other_missing}, unexpected={unexpected}.')
  if scale_missing:
    model.params.predict_amplitude_scale = False
  model.eval()
  return model


def generate_audio(model,
                   spectrogram=None,
                   physical_params=None,
                   stats=None,
                   device=None,
                   fast_sampling=False,
                   generator=None):
  device = _default_device(device)
  model = model.to(device)
  if stats is not None:
    if 'sample_rate' in stats and int(stats['sample_rate']) != int(model.params.sample_rate):
      raise ValueError(
          f"sample_rate mismatch: stats={stats['sample_rate']}, model={model.params.sample_rate}.")
    if 'audio_len' in stats and int(stats['audio_len']) != int(model.params.audio_len):
      raise ValueError(
          f"audio_len mismatch: stats={stats['audio_len']}, model={model.params.audio_len}.")
  physical_params, batch_size = _prepare_physical_params(physical_params, stats, device)

  if spectrogram is not None:
    if len(spectrogram.shape) == 2:
      spectrogram = spectrogram.unsqueeze(0)
    spectrogram = spectrogram.to(device)
    batch_size = spectrogram.shape[0]

  training_noise_schedule = np.array(model.params.noise_schedule)
  inference_noise_schedule = np.array(model.params.inference_noise_schedule) if fast_sampling else training_noise_schedule

  talpha = 1 - training_noise_schedule
  talpha_cum = np.cumprod(talpha)

  beta = inference_noise_schedule
  alpha = 1 - beta
  alpha_cum = np.cumprod(alpha)

  T = []
  for s in range(len(inference_noise_schedule)):
    for t in range(len(training_noise_schedule) - 1):
      if talpha_cum[t+1] <= alpha_cum[s] <= talpha_cum[t]:
        twiddle = (talpha_cum[t]**0.5 - alpha_cum[s]**0.5) / (talpha_cum[t]**0.5 - talpha_cum[t+1]**0.5)
        T.append(t + twiddle)
        break
  T = np.array(T, dtype=np.float32)

  audio = torch.randn(
      batch_size,
      getattr(model.params, 'audio_channels', 1),
      model.params.audio_len,
      device=device,
      generator=generator)

  for n in range(len(alpha) - 1, -1, -1):
    c1 = 1 / alpha[n]**0.5
    c2 = beta[n] / (1 - alpha_cum[n])**0.5
    diffusion_step = torch.full([batch_size], T[n], device=audio.device)
    pred = model(audio, diffusion_step, spectrogram, physical_params)
    audio = c1 * (audio - c2 * pred)
    if n > 0:
      noise = torch.randn(audio.shape, device=audio.device, dtype=audio.dtype, generator=generator)
      sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
      audio += sigma * noise
    if getattr(model.params, 'sample_clamp', False):
      audio = torch.clamp(audio, -1.0, 1.0)

  return _denormalize_audio(audio, stats, model=model, physical_params=physical_params), model.params.sample_rate


def predict(spectrogram=None, physical_params=None, model_dir=None, params=None, device=None, fast_sampling=False, stats_path=None):
  device = _default_device(device)
  cache_key = (model_dir, str(device))
  stats = _load_stats(model_dir, stats_path)

  if cache_key not in models:
    models[cache_key] = _load_model(model_dir, params, device)

  model = models[cache_key]
  with torch.no_grad():
    return generate_audio(
        model,
        spectrogram=spectrogram,
        physical_params=physical_params,
        stats=stats,
        device=device,
        fast_sampling=fast_sampling)


def write_blast_csv(output_path, audio, sample_rate):
  audio = audio.detach().cpu().numpy()
  if audio.ndim == 3:
    audio = audio[0]
  if audio.ndim == 1:
    audio = audio[None, :]
  audio = audio.T
  time = np.arange(audio.shape[0], dtype=np.float64) / sample_rate

  if audio.shape[1] == len(BLAST_CHANNEL_COLUMNS):
    header = 'time_s,' + ','.join(BLAST_CHANNEL_COLUMNS)
  else:
    header = 'time_s,' + ','.join(f'CH{i + 1}' for i in range(audio.shape[1]))
  np.savetxt(output_path, np.column_stack([time, audio]), delimiter=',', header=header, comments='')


def _parse_params(params_text):
  try:
    return [float(x.strip()) for x in params_text.split(',') if x.strip()]
  except ValueError as exc:
    raise ValueError('Expected comma-separated numeric blast parameters.') from exc


def main(args):
  if args.spectrogram_path:
    spectrogram = torch.from_numpy(np.load(args.spectrogram_path))
  else:
    spectrogram = None

  physical_params = None
  params_text = args.blast_params or args.physical_params
  if params_text:
    physical_params = torch.tensor(_parse_params(params_text))
    print(f"Generating with blast params: {physical_params}")

  audio, sr = predict(
      spectrogram,
      physical_params=physical_params,
      model_dir=args.model_dir,
      fast_sampling=args.fast,
      params=None,
      device=args.device,
      stats_path=args.stats_path)

  if args.output.lower().endswith('.csv') or args.blast_params:
    write_blast_csv(args.output, audio, sr)
  else:
    if sf is None:
      raise ImportError('soundfile is required to write non-CSV audio output.')
    sf.write(args.output, audio.cpu().numpy().squeeze().T, sr)


if __name__ == '__main__':
  parser = ArgumentParser(description='runs inference with a trained DiffWave model')
  parser.add_argument('model_dir',
      help='directory containing a trained model (or full path to weights.pt file)')
  parser.add_argument('--spectrogram_path', '-s',
      help='path to a spectrogram file generated by diffwave.preprocess')
  parser.add_argument('--blast_params',
      help='comma-separated parameters: Q_max,Q_total,Hole_Num,Delay_hole,Delay_row,Hole_Diameter,Distance_R,Elev_Diff')
  parser.add_argument('--physical_params', '-p',
      help='legacy alias for --blast_params')
  parser.add_argument('--stats_path',
      help='path to blast_stats.json; defaults to the model directory')
  parser.add_argument('--output', '-o', default='output.csv',
      help='output file name')
  parser.add_argument('--device',
      help='inference device, e.g. cuda, cuda:0, or cpu')
  parser.add_argument('--fast', '-f', action='store_true',
      help='fast sampling procedure')
  main(parser.parse_args())
