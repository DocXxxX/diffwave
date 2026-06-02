from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
  from scipy.signal import hilbert
except Exception:
  hilbert = None

from diffwave.blast_eval import compute_generation_metrics, validate_signal_config
from diffwave.dataset import (
    BLAST_CHANNEL_COLUMNS,
    BLAST_STATS_FILE,
    build_blast_records,
    load_blast_stats,
    _crop_or_pad_peak,
    _load_blast_waveform,
    _scale_target,
    _split_records,
)
from diffwave.inference import _load_model, _prepare_physical_params, generate_audio
from diffwave.params import params as base_params


def _default_device(device):
  if device:
    return torch.device(device)
  return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _default_stats_path(model_dir):
  model_dir = Path(model_dir)
  return model_dir / BLAST_STATS_FILE if model_dir.is_dir() else model_dir.parent / BLAST_STATS_FILE


def _import_pyplot():
  try:
    import matplotlib.pyplot as plt
  except ImportError as exc:
    raise ImportError('matplotlib is required for blast diagnostics: pip install matplotlib') from exc
  return plt


def _plots_available():
  try:
    _import_pyplot()
    return True
  except ImportError as exc:
    print(f'[WARN] {exc}; metrics CSV will still be written.')
    return False


def _envelope(audio):
  if hilbert is not None:
    return np.abs(hilbert(audio, axis=1))
  return np.abs(audio)


def _plot_waveforms(real, generated, sample_rate, output_path, title):
  plt = _import_pyplot()
  time = np.arange(real.shape[1], dtype=np.float64) / float(sample_rate)
  fig, axes = plt.subplots(real.shape[0], 1, figsize=(14, 3.0 * real.shape[0]), sharex=True)
  if real.shape[0] == 1:
    axes = [axes]
  for channel_index, ax in enumerate(axes):
    label = BLAST_CHANNEL_COLUMNS[channel_index] if channel_index < len(BLAST_CHANNEL_COLUMNS) else f'CH{channel_index + 1}'
    ax.plot(time, real[channel_index], linewidth=0.8, label='real')
    ax.plot(time, generated[channel_index], linewidth=0.8, alpha=0.8, label='generated')
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.25)
  axes[0].set_title(title)
  axes[0].legend(loc='upper right')
  axes[-1].set_xlabel('Time (s)')
  fig.tight_layout()
  fig.savefig(output_path, dpi=160)
  plt.close(fig)


def _plot_spectra(real, generated, sample_rate, output_path, title):
  plt = _import_pyplot()
  freqs = np.fft.rfftfreq(real.shape[1], d=1.0 / float(sample_rate))
  fig, axes = plt.subplots(real.shape[0], 1, figsize=(14, 3.0 * real.shape[0]), sharex=True)
  if real.shape[0] == 1:
    axes = [axes]
  for channel_index, ax in enumerate(axes):
    label = BLAST_CHANNEL_COLUMNS[channel_index] if channel_index < len(BLAST_CHANNEL_COLUMNS) else f'CH{channel_index + 1}'
    real_spec = np.abs(np.fft.rfft(real[channel_index] - np.mean(real[channel_index])))
    gen_spec = np.abs(np.fft.rfft(generated[channel_index] - np.mean(generated[channel_index])))
    ax.semilogy(freqs, real_spec + 1e-12, linewidth=0.8, label='real')
    ax.semilogy(freqs, gen_spec + 1e-12, linewidth=0.8, alpha=0.8, label='generated')
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.25)
  axes[0].set_title(title)
  axes[0].legend(loc='upper right')
  axes[-1].set_xlabel('Frequency (Hz)')
  fig.tight_layout()
  fig.savefig(output_path, dpi=160)
  plt.close(fig)


def _plot_envelopes(real, generated, sample_rate, output_path, title):
  plt = _import_pyplot()
  time = np.arange(real.shape[1], dtype=np.float64) / float(sample_rate)
  real_env = _envelope(real)
  gen_env = _envelope(generated)
  fig, axes = plt.subplots(real.shape[0], 1, figsize=(14, 3.0 * real.shape[0]), sharex=True)
  if real.shape[0] == 1:
    axes = [axes]
  for channel_index, ax in enumerate(axes):
    label = BLAST_CHANNEL_COLUMNS[channel_index] if channel_index < len(BLAST_CHANNEL_COLUMNS) else f'CH{channel_index + 1}'
    ax.plot(time, real_env[channel_index], linewidth=0.8, label='real')
    ax.plot(time, gen_env[channel_index], linewidth=0.8, alpha=0.8, label='generated')
    ax.set_ylabel(label)
    ax.grid(True, alpha=0.25)
  axes[0].set_title(title)
  axes[0].legend(loc='upper right')
  axes[-1].set_xlabel('Time (s)')
  fig.tight_layout()
  fig.savefig(output_path, dpi=160)
  plt.close(fig)


def _scale_prediction_metrics(model, record, real, stats, device):
  output = {
      'scale_log_rms_error': np.nan,
      'scale_log_peak_error': np.nan,
  }
  if stats is None or not hasattr(model, 'predict_scale') or getattr(model, 'scale_predictor', None) is None:
    return output
  if 'scale_mean' not in stats or 'scale_std' not in stats:
    return output

  try:
    condition, _ = _prepare_physical_params(
        record['physical_params'],
        stats,
        device,
        monitor_id=record.get('monitor_id'),
        instrument=record.get('instrument'))
    predicted = model.predict_scale(condition)
    if predicted is None:
      return output
    mean = torch.tensor(stats['scale_mean'], device=predicted.device, dtype=predicted.dtype).view(1, -1)
    std = torch.tensor(stats['scale_std'], device=predicted.device, dtype=predicted.dtype).view(1, -1)
    predicted_raw = (predicted * std + mean).detach().cpu().numpy()[0]
    target_raw = _scale_target(real)
    channels = real.shape[0]
    output['scale_log_rms_error'] = float(np.mean(np.abs(predicted_raw[:channels] - target_raw[:channels])))
    output['scale_log_peak_error'] = float(np.mean(np.abs(predicted_raw[channels:channels * 2] - target_raw[channels:channels * 2])))
  except Exception as exc:
    print(f"[WARN] Failed to compute scale prediction metrics for {record.get('path')}: {exc}")
  return output


def _select_records(records, subset_size, seed):
  if subset_size is None or subset_size <= 0 or subset_size >= len(records):
    return list(records)
  rng = np.random.default_rng(seed)
  indices = sorted(rng.choice(len(records), size=subset_size, replace=False).tolist())
  return [records[index] for index in indices]


def run_diagnostics(args):
  model_dir = Path(args.model_dir)
  stats_path = Path(args.stats_path) if args.stats_path else _default_stats_path(model_dir)
  stats = load_blast_stats(str(stats_path)) if stats_path.exists() else None
  device = _default_device(args.device)
  model = _load_model(str(model_dir), None, device)

  data_dirs = args.data_dirs or [base_params.default_blast_data_dir]
  params_csv = args.params_csv or base_params.default_blast_params_csv
  records = build_blast_records(data_dirs, params_csv)
  split_mode = args.split_mode
  if split_mode == 'auto':
    split_mode = stats.get('split_mode') if stats is not None and 'split_mode' in stats else 'record'
  _, val_records = _split_records(
      records,
      getattr(model.params, 'val_ratio', getattr(base_params, 'val_ratio', 0.15)),
      getattr(model.params, 'split_seed', getattr(base_params, 'split_seed', 2021)),
      split_mode)
  target_records = _select_records(val_records or records, args.subset_size, args.seed)

  sample_rate = int(stats.get('sample_rate', model.params.sample_rate)) if stats else int(model.params.sample_rate)
  audio_len = int(stats.get('audio_len', model.params.audio_len)) if stats else int(model.params.audio_len)
  bands, fft_sizes = validate_signal_config(sample_rate, audio_len)

  out_dir = Path(args.out_dir) if args.out_dir else model_dir / 'blast_diagnostics'
  waveform_dir = out_dir / 'real_vs_generated'
  spectra_dir = out_dir / 'spectra'
  envelope_dir = out_dir / 'envelopes'
  for directory in [out_dir, waveform_dir, spectra_dir, envelope_dir]:
    directory.mkdir(parents=True, exist_ok=True)

  rows = []
  plot_enabled = _plots_available()
  was_training = model.training
  model.eval()
  with torch.no_grad():
    for record_index, record in enumerate(target_records):
      real = _crop_or_pad_peak(_load_blast_waveform(record['path']), audio_len)
      for sample_index in range(max(1, int(args.samples_per_condition))):
        generator = torch.Generator(device=device) if device.type == 'cuda' else torch.Generator()
        generator.manual_seed(int(args.seed) + record_index * 1009 + sample_index)
        generated, _ = generate_audio(
            model,
            physical_params=record['physical_params'],
            stats=stats,
            device=device,
            fast_sampling=args.fast,
            generator=generator,
            monitor_id=record.get('monitor_id'),
            instrument=record.get('instrument'))
        generated = generated.detach().cpu().numpy()[0]
        metrics = compute_generation_metrics(generated, real, sample_rate, bands=bands, fft_sizes=fft_sizes)
        metrics.update(_scale_prediction_metrics(model, record, real, stats, device))
        row = {
            'path': record['path'],
            'event_id': record.get('event_id', ''),
            'monitor_id': record.get('monitor_id', ''),
            'instrument': record.get('instrument', ''),
            'sample_index': sample_index,
        }
        row.update(metrics)
        rows.append(row)

        stem = f"{record_index:03d}_{record.get('event_id', 'event')}_M{record.get('monitor_id', 'x')}_S{sample_index}"
        title = f"{record.get('event_id', '')} monitor={record.get('monitor_id', '')} sample={sample_index}"
        if plot_enabled:
          _plot_waveforms(real, generated, sample_rate, waveform_dir / f'{stem}.png', title)
          _plot_spectra(real, generated, sample_rate, spectra_dir / f'{stem}.png', title)
          _plot_envelopes(real, generated, sample_rate, envelope_dir / f'{stem}.png', title)
  if was_training:
    model.train()

  metrics_df = pd.DataFrame(rows)
  metrics_path = out_dir / 'metrics.csv'
  metrics_df.to_csv(metrics_path, index=False)
  group_cols = ['path', 'event_id', 'monitor_id', 'instrument']
  numeric_cols = [col for col in metrics_df.columns if col not in group_cols + ['sample_index']]
  condition_metrics = metrics_df.groupby(group_cols, as_index=False)[numeric_cols].mean()
  condition_metrics.to_csv(out_dir / 'condition_metrics.csv', index=False)
  print(f'[INFO] Wrote diagnostics metrics: {metrics_path}')
  print(f'[INFO] Wrote diagnostic figures to: {out_dir}')


def main():
  parser = ArgumentParser(description='diagnose generated blast vibration waveforms against validation records')
  parser.add_argument('model_dir',
      help='directory containing weights.pt and optional blast_stats.json')
  parser.add_argument('--data_dirs', nargs='*',
      help='blast CSV data directories; defaults to params.default_blast_data_dir')
  parser.add_argument('--params_csv',
      help='blast parameter CSV path; defaults to params.default_blast_params_csv')
  parser.add_argument('--stats_path',
      help='blast_stats.json path; defaults to the model directory')
  parser.add_argument('--out_dir',
      help='diagnostic output directory; defaults to model_dir/blast_diagnostics')
  parser.add_argument('--samples_per_condition', type=int, default=1)
  parser.add_argument('--subset_size', type=int, default=6)
  parser.add_argument('--seed', type=int, default=20260425)
  parser.add_argument('--split_mode', choices=['auto', 'record', 'event'], default='auto')
  parser.add_argument('--device',
      help='inference device, e.g. cuda, cuda:0, or cpu')
  parser.add_argument('--fast', action='store_true',
      help='use the fast inference noise schedule')
  run_diagnostics(parser.parse_args())


if __name__ == '__main__':
  main()
