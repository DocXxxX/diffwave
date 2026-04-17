from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd


CHANNEL_COLUMNS = ['CH1', 'CH2', 'OT']
PARAM_COLUMNS = [
  'Q_max',
  'Q_total',
  'Hole_Num',
  'Delay_hole',
  'Delay_row',
  'Hole_Diameter',
  'Distance_R',
  'Elev_Diff',
]


def _import_pyplot():
  try:
    import matplotlib.pyplot as plt
  except ImportError as exc:
    raise ImportError('matplotlib is required: pip install matplotlib') from exc
  return plt


def _manifest_params(sample_dir):
  manifest_path = sample_dir / 'samples_manifest.csv'
  if not manifest_path.exists():
    return {}
  manifest = pd.read_csv(manifest_path)
  if 'filename' not in manifest.columns:
    return {}
  missing_cols = [col for col in PARAM_COLUMNS if col not in manifest.columns]
  if missing_cols:
    return {}
  params_by_file = {}
  for _, row in manifest.iterrows():
    params_by_file[str(row['filename'])] = [float(row[col]) for col in PARAM_COLUMNS]
  return params_by_file


def _sample_paths(root):
  root = Path(root)
  for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
    sample_dir = run_dir / 'samples'
    if not sample_dir.exists():
      continue
    params_by_file = _manifest_params(sample_dir)
    for sample_path in sorted(sample_dir.glob('sample_*.csv')):
      yield run_dir.name, sample_path, params_by_file.get(sample_path.name)


def _read_sample(sample_path):
  df = pd.read_csv(sample_path)
  channels = [col for col in CHANNEL_COLUMNS if col in df.columns]
  if 'time_s' not in df.columns or not channels:
    raise ValueError(f'{sample_path} must contain time_s and at least one sample channel.')
  return df, channels


def _dominant_frequency(time_s, values):
  if len(values) < 3:
    return np.nan
  dt = float(np.median(np.diff(time_s)))
  if not np.isfinite(dt) or dt <= 0:
    return np.nan
  centered = values - np.mean(values)
  spectrum = np.abs(np.fft.rfft(centered))
  freqs = np.fft.rfftfreq(len(centered), d=dt)
  if len(spectrum) <= 1:
    return np.nan
  index = int(np.argmax(spectrum[1:]) + 1)
  return float(freqs[index])


def _param_key(params, sample_name):
  if params is None:
    return ('legacy_sample_name', sample_name)
  return tuple(round(float(value), 6) for value in params)


def _param_title(params):
  if params is None:
    return 'No parameter manifest found'
  return ', '.join(f'{name}={value:g}' for name, value in zip(PARAM_COLUMNS, params))


def _safe_name(text):
  return ''.join(ch if ch.isalnum() or ch in ['_', '-'] else '_' for ch in text)


def _metrics(run_id, sample_name, param_group, params, df, channels):
  rows = []
  time_s = df['time_s'].to_numpy(dtype=np.float64)
  for channel in channels:
    values = df[channel].to_numpy(dtype=np.float64)
    finite = np.isfinite(values)
    finite_values = values[finite]
    if len(finite_values) == 0:
      rows.append({
          'param_group': param_group,
          'run_id': run_id,
          'sample': sample_name,
          'channel': channel,
          'finite_ratio': 0.0,
      })
      continue
    row = {
        'param_group': param_group,
        'run_id': run_id,
        'sample': sample_name,
        'channel': channel,
    }
    if params is not None:
      row.update({name: value for name, value in zip(PARAM_COLUMNS, params)})
    max_abs = float(np.max(np.abs(finite_values)))
    rms = float(np.sqrt(np.mean(np.square(finite_values))))
    row.update({
        'finite_ratio': float(np.mean(finite)),
        'min': float(np.min(finite_values)),
        'max': float(np.max(finite_values)),
        'max_abs': max_abs,
        'rms': rms,
        'mean': float(np.mean(finite_values)),
        'std': float(np.std(finite_values)),
        'peak_to_peak': float(np.ptp(finite_values)),
        'crest_factor': max_abs / rms if rms > 0 else np.nan,
        'dominant_freq_hz': _dominant_frequency(time_s[finite], finite_values),
      })
    rows.append(row)
  return rows


def _plot_overlay(param_group, records, out_dir, max_points):
  plt = _import_pyplot()
  channels = records[0]['channels']
  fig, axes = plt.subplots(len(channels), 1, figsize=(14, 3.2 * len(channels)), sharex=True)
  if len(channels) == 1:
    axes = [axes]

  for record in records:
    df = record['df']
    stride = max(1, len(df) // max_points)
    x = df['time_s'].to_numpy()[::stride]
    for ax, channel in zip(axes, channels):
      ax.plot(x, df[channel].to_numpy()[::stride], linewidth=0.8, alpha=0.75, label=record['run_id'])

  for ax, channel in zip(axes, channels):
    ax.set_ylabel(channel)
    ax.grid(True, alpha=0.25)
  axes[-1].set_xlabel('Time (s)')
  axes[0].set_title(f'{param_group} overlay\n{_param_title(records[0]["params"])}')
  axes[0].legend(loc='upper right', fontsize=8, ncol=2)
  fig.tight_layout()
  fig.savefig(out_dir / 'overlay.png', dpi=180)
  plt.close(fig)


def _plot_individual(record, out_dir, max_points):
  plt = _import_pyplot()
  df = record['df']
  channels = record['channels']
  fig, axes = plt.subplots(len(channels), 1, figsize=(14, 3.0 * len(channels)), sharex=True)
  if len(channels) == 1:
    axes = [axes]
  stride = max(1, len(df) // max_points)
  x = df['time_s'].to_numpy()[::stride]
  for ax, channel in zip(axes, channels):
    ax.plot(x, df[channel].to_numpy()[::stride], linewidth=0.8)
    ax.set_ylabel(channel)
    ax.grid(True, alpha=0.25)
  axes[-1].set_xlabel('Time (s)')
  axes[0].set_title(f'{record["run_id"]} / {record["sample_name"]}\n{_param_title(record["params"])}')
  fig.tight_layout()
  fig.savefig(out_dir / f'{_safe_name(record["run_id"])}_{record["sample_name"]}.png', dpi=160)
  plt.close(fig)


def _plot_metric_summary(metrics_df, out_dir):
  plt = _import_pyplot()
  if metrics_df.empty:
    return
  summary = (
      metrics_df
      .groupby(['param_group', 'run_id', 'channel'], as_index=False)
      .agg(max_abs=('max_abs', 'mean'), rms=('rms', 'mean'), dominant_freq_hz=('dominant_freq_hz', 'mean'))
  )
  for metric in ['max_abs', 'rms', 'dominant_freq_hz']:
    for param_group, group in summary.groupby('param_group'):
      pivot = group.pivot(index='run_id', columns='channel', values=metric)
      ax = pivot.plot(kind='bar', figsize=(14, 5), width=0.8)
      ax.set_title(f'{param_group}: {metric} by run')
      ax.set_xlabel('Run')
      ax.set_ylabel(metric)
      ax.grid(True, axis='y', alpha=0.25)
      ax.legend(title='Channel')
      ax.figure.tight_layout()
      ax.figure.savefig(out_dir / f'{param_group}_{metric}_by_run.png', dpi=180)
      plt.close(ax.figure)


def compare_samples(root, out_dir, max_points):
  root = Path(root)
  out_dir = Path(out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)

  grouped = {}
  group_names = {}
  metric_rows = []
  for run_id, sample_path, params in _sample_paths(root):
    df, channels = _read_sample(sample_path)
    sample_name = sample_path.stem
    key = _param_key(params, sample_name)
    if key not in group_names:
      group_names[key] = f'param_{len(group_names) + 1:03d}'
    param_group = group_names[key]
    grouped.setdefault(key, []).append({
        'run_id': run_id,
        'path': sample_path,
        'sample_name': sample_name,
        'param_group': param_group,
        'params': params,
        'df': df,
        'channels': channels,
    })
    metric_rows.extend(_metrics(run_id, sample_name, param_group, params, df, channels))

  metrics_df = pd.DataFrame(metric_rows)
  metrics_path = out_dir / 'sample_metrics.csv'
  metrics_df.to_csv(metrics_path, index=False)

  group_index_rows = []
  for key, records in grouped.items():
    param_group = records[0]['param_group']
    group_dir = out_dir / param_group
    group_dir.mkdir(parents=True, exist_ok=True)
    _plot_overlay(param_group, records, group_dir, max_points)
    for record in records:
      _plot_individual(record, group_dir, max_points)
      row = {
          'param_group': param_group,
          'run_id': record['run_id'],
          'sample': record['sample_name'],
          'path': str(record['path']),
      }
      if record['params'] is not None:
        row.update({name: value for name, value in zip(PARAM_COLUMNS, record['params'])})
      group_index_rows.append(row)
    group_metrics = metrics_df[metrics_df['param_group'] == param_group]
    group_metrics.to_csv(group_dir / 'metrics.csv', index=False)
    pd.DataFrame(group_index_rows).query('param_group == @param_group').to_csv(
        group_dir / 'samples_index.csv',
        index=False)

  pd.DataFrame(group_index_rows).to_csv(out_dir / 'samples_index.csv', index=False)
  _plot_metric_summary(metrics_df, out_dir)

  print(f'[INFO] Found {len(grouped)} sample groups under {root}.')
  print(f'[INFO] Wrote metrics: {metrics_path}')
  print(f'[INFO] Wrote figures to: {out_dir}')


if __name__ == '__main__':
  parser = ArgumentParser(description='compare generated sweep samples visually')
  parser.add_argument('--root', default='sweep_runs/blast_mamba',
      help='sweep run root directory')
  parser.add_argument('--out_dir', default='sweep_runs/blast_mamba/sample_comparison',
      help='directory for comparison figures and metrics')
  parser.add_argument('--max_points', default=2500, type=int,
      help='maximum points per line in overlay plots')
  args = parser.parse_args()
  compare_samples(args.root, args.out_dir, args.max_points)
