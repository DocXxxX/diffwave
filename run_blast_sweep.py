from argparse import Namespace
import csv
import gc
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / 'src'
if str(SRC_DIR) not in sys.path:
  sys.path.insert(0, str(SRC_DIR))

import wandb
import torch
from diffwave.blast_eval import BlastGenerationEvaluator
from diffwave.inference import predict, write_blast_csv
from diffwave.learner import train
from diffwave.params import AttrDict, apply_blast_augment_level, params as base_params


SAMPLE_PARAM_COLUMNS = [
  'Q_max',
  'Q_total',
  'Hole_Num',
  'Delay_hole',
  'Delay_row',
  'Hole_Diameter',
  'Distance_R',
  'Elev_Diff',
]

DEFAULT_SAMPLE_PARAM_SETS = [
  [2.2, 151.8, 76.0, 28.6, 74.8, 57.5, 13.8, 6.5],
  [1.6, 110.0, 55.0, 25.0, 65.0, 42.0, 10.0, 4.0],
  [2.8, 210.0, 105.0, 32.0, 90.0, 76.0, 22.0, 10.0],
]

PARAM_OVERRIDE_NAMES = [
  'batch_size',
  'learning_rate',
  'max_grad_norm',
  'audio_len',
  'audio_channels',
  'audio_clip',
  'blast_norm_mode',
  'predict_amplitude_scale',
  'lambda_scale',
  'condition_dim',
  'residual_layers',
  'residual_channels',
  'dilation_cycle_length',
  'use_mamba',
  'mamba_d_state',
  'mamba_expand',
  'film_hidden_dim',
  'loss_type',
  'validation_interval',
  'validation_batches',
  'checkpoint_interval',
  'num_workers',
  'pin_memory',
  'split_seed',
  'val_ratio',
  'blast_augment',
  'blast_augment_level',
  'blast_peak_jitter',
  'blast_time_shift',
  'blast_gain_min',
  'blast_gain_max',
  'sample_clamp',
  'lambda_mr_stft',
  'lambda_band_energy',
  'lambda_envelope',
  'lambda_peak_rms',
  'aux_loss_warmup_steps',
  'aux_loss_timestep_max_ratio',
  'aux_loss_min_snr',
  'gen_eval_interval',
  'gen_eval_subset_size',
  'gen_eval_samples_per_condition',
  'full_gen_eval_samples_per_condition',
  'gen_eval_seed',
  'gen_eval_fast_sampling',
]

SAFETY_DEFAULTS = {
  'audio_clip': 8.0,
  'max_grad_norm': 1.0,
}


def _project_root():
  return PROJECT_ROOT


def _default_path(relative_path):
  return str(_project_root() / relative_path)


def _resolve_path(path):
  path = Path(str(path))
  if path.is_absolute():
    return str(path)
  return str(_project_root() / path)


def _as_bool(value):
  if isinstance(value, bool):
    return value
  if isinstance(value, (int, float)):
    return bool(value)
  value = str(value).lower()
  if value in ['true', '1', 'yes', 'y']:
    return True
  if value in ['false', '0', 'no', 'n']:
    return False
  raise ValueError(f'Invalid boolean value: {value}')


def _as_data_dirs(value):
  if value is None:
    return [_default_path(base_params.default_blast_data_dir)]
  if isinstance(value, str):
    value = [value]
  return [_resolve_path(path) for path in value]


def _config_value(config, name, default=None):
  if name in config:
    return config[name]
  return default


def _sample_param_sets(config):
  param_sets = _config_value(config, 'sample_blast_params')
  if param_sets is None:
    return DEFAULT_SAMPLE_PARAM_SETS
  if isinstance(param_sets, str):
    param_sets = [[float(value.strip()) for value in param_sets.split(',') if value.strip()]]
  if param_sets and isinstance(param_sets[0], (int, float)):
    param_sets = [param_sets]
  return [[float(value) for value in params] for params in param_sets]


def _generate_samples(model_dir, config, device):
  if not _as_bool(_config_value(config, 'generate_samples', True)):
    return

  model_dir = Path(model_dir)
  if not (model_dir / 'weights.pt').exists():
    print(f'[WARN] Skip sample generation because weights.pt was not found in {model_dir}.')
    return

  gc.collect()
  if torch.cuda.is_available():
    torch.cuda.empty_cache()

  sample_dir = model_dir / str(_config_value(config, 'sample_dir', 'samples'))
  sample_dir.mkdir(parents=True, exist_ok=True)
  sample_fast = _as_bool(_config_value(config, 'sample_fast', True))
  sample_device = _config_value(config, 'sample_device', device)
  manifest_path = sample_dir / 'samples_manifest.csv'

  with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['filename'] + SAMPLE_PARAM_COLUMNS)
    for index, sample_params in enumerate(_sample_param_sets(config), start=1):
      if len(sample_params) != len(SAMPLE_PARAM_COLUMNS):
        raise ValueError(f'Expected {len(SAMPLE_PARAM_COLUMNS)} sample blast parameters, got {len(sample_params)}.')
      output_path = sample_dir / f'sample_{index:03d}.csv'
      audio, sample_rate = predict(
          physical_params=sample_params,
          model_dir=str(model_dir),
          device=sample_device,
          fast_sampling=sample_fast)
      write_blast_csv(str(output_path), audio, sample_rate)
      writer.writerow([output_path.name] + sample_params)
      print(f'[INFO] Generated sample: {output_path}')


def _full_generation_eval(learner, run, config):
  if learner is None:
    return
  evaluator = learner.gen_evaluator
  if evaluator is None:
    dataset = getattr(getattr(learner, 'val_dataset', None), 'dataset', None)
    records = getattr(dataset, 'records', None)
    stats = getattr(dataset, 'stats', None)
    if not records or stats is None:
      print('[WARN] Skip full generation eval because validation records/stats are unavailable.')
      return
    evaluator = BlastGenerationEvaluator(records, stats, learner.params)

  output_dir = Path(learner.model_dir) / 'gen_eval'
  output_csv = output_dir / 'full_metrics.csv'
  device = next(learner.model.parameters()).device
  metrics = evaluator.evaluate_model(
      learner._raw_model(),
      device=device,
      subset_size=None,
      samples_per_condition=int(_config_value(
          config,
          'full_gen_eval_samples_per_condition',
          getattr(learner.params, 'full_gen_eval_samples_per_condition', 3))),
      fast_sampling=_as_bool(_config_value(
          config,
          'gen_eval_fast_sampling',
          getattr(learner.params, 'gen_eval_fast_sampling', True))),
      seed=int(_config_value(config, 'gen_eval_seed', getattr(learner.params, 'gen_eval_seed', 20260425))),
      output_csv=str(output_csv))
  if not metrics:
    return

  log_metrics = {
      'val/full_gen_score_mean': metrics['gen_score_mean'],
      'val/full_gen_score_best': metrics['gen_score_best'],
      'val/full_gen_invalid_metric_count': metrics.get('gen_invalid_metric_count_mean', 0.0),
  }
  for name, value in metrics.items():
    if name.startswith('gen_score') or name.startswith('gen_invalid_metric_count'):
      continue
    log_metrics[f'val/full_{name}'] = value
  run.log(log_metrics)

  artifact = wandb.Artifact(f'{run.id}-full-generation-eval', type='generation-eval')
  artifact.add_file(str(output_csv))
  run.log_artifact(artifact)


def _run_params(config):
  run_params = AttrDict(base_params)
  run_params.override(SAFETY_DEFAULTS)
  overrides = {
      name: config[name]
      for name in PARAM_OVERRIDE_NAMES
      if name in config and config[name] is not None
  }
  augment_level = overrides.pop('blast_augment_level', None)
  apply_blast_augment_level(run_params, augment_level)
  run_params.override(overrides)
  run_params.use_wandb = False
  return run_params


def main():
  with wandb.init() as run:
    config = dict(run.config)
    run_params = _run_params(config)
    for name in SAFETY_DEFAULTS:
      if name not in config:
        run.config.update({name: run_params[name]}, allow_val_change=True)
    run.config.update({
        'blast_augment': run_params.blast_augment,
        'blast_augment_level': run_params.blast_augment_level,
        'blast_peak_jitter': run_params.blast_peak_jitter,
        'blast_time_shift': run_params.blast_time_shift,
        'blast_gain_min': run_params.blast_gain_min,
        'blast_gain_max': run_params.blast_gain_max,
    }, allow_val_change=True)

    model_root = _resolve_path(_config_value(config, 'model_dir', 'sweep_runs/blast_mamba'))
    data_dirs = _as_data_dirs(_config_value(config, 'data_dirs'))
    params_csv = _resolve_path(_config_value(
        config,
        'params_csv',
        base_params.default_blast_params_csv))

    train_args = Namespace(
        model_dir=str(Path(model_root) / run.id),
        data_dirs=data_dirs,
        params_csv=params_csv,
        data_format=_config_value(config, 'data_format', 'blast_csv'),
        max_steps=_config_value(config, 'max_steps'),
        fp16=_as_bool(_config_value(config, 'fp16', False)),
        device=_config_value(config, 'device', 'cuda'))
    learner = train(train_args, run_params, wandb_run=run)
    _full_generation_eval(learner, run, config)
    _generate_samples(train_args.model_dir, config, train_args.device)


if __name__ == '__main__':
  main()
