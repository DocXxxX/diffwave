import os

from argparse import ArgumentParser, Namespace
from pathlib import Path

from diffwave.blast_eval import BlastGenerationEvaluator
from diffwave.learner import train
from diffwave.params import AttrDict, params as base_params


def _project_root():
  return Path(__file__).resolve().parents[2]


def _default_path(relative_path):
  return str(_project_root() / relative_path)


def _str_to_bool(value):
  if isinstance(value, bool):
    return value
  value = value.lower()
  if value in ['true', '1', 'yes', 'y']:
    return True
  if value in ['false', '0', 'no', 'n']:
    return False
  raise ValueError(f'Invalid boolean value: {value}')


def _sweep_config(max_steps, use_mamba=True, smoke=False):
  if smoke:
    batch_values = [2]
    residual_channel_values = [8]
    residual_layer_values = [2]
    mamba_state_values = [8]
    mamba_expand_values = [1]
  else:
    batch_values = [8, 16, 32]
    residual_channel_values = [32, 64]
    residual_layer_values = [12, 20, 30]
    mamba_state_values = [8, 16, 32]
    mamba_expand_values = [1, 2]

  return {
      'method': 'bayes',
      'metric': {
          'name': 'val/full_gen_score_mean',
          'goal': 'minimize',
      },
      'early_terminate': {
          'type': 'hyperband',
          'max_iter': 10,
          's': 2,
          'eta': 3,
      },
      'parameters': {
          'learning_rate': {
              'distribution': 'log_uniform_values',
              'min': 1e-5,
              'max': 5e-4,
          },
          'batch_size': {
              'values': batch_values,
          },
          'residual_channels': {
              'values': residual_channel_values,
          },
          'residual_layers': {
              'values': residual_layer_values,
          },
          'mamba_d_state': {
              'values': mamba_state_values,
          },
          'mamba_expand': {
              'values': mamba_expand_values,
          },
          'use_mamba': {
              'value': use_mamba,
          },
          'loss_type': {
              'values': ['l1', 'mse'],
          },
          'audio_len': {
              'value': base_params.audio_len,
          },
          'audio_channels': {
              'value': base_params.audio_channels,
          },
          'condition_dim': {
              'value': base_params.condition_dim,
          },
          'blast_norm_mode': {
              'value': 'robust_log_scale',
          },
          'blast_condition_mode': {
              'value': base_params.blast_condition_mode,
          },
          'blast_split_mode': {
              'value': base_params.blast_split_mode,
          },
          'predict_amplitude_scale': {
              'value': True,
          },
          'lambda_scale': {
              'values': [0.01, 0.05, 0.1],
          },
          'lambda_mr_stft': {
              'values': [0.0, 0.005, 0.01],
          },
          'lambda_band_energy': {
              'values': [0.0, 0.01, 0.03],
          },
          'lambda_envelope': {
              'values': [0.0, 0.01, 0.03],
          },
          'lambda_peak_rms': {
              'values': [0.0, 0.01, 0.03],
          },
          'lambda_cumulative_energy': {
              'values': [0.0, 0.01, 0.03],
          },
          'gen_eval_interval': {
              'value': 2000,
          },
          'gen_eval_subset_size': {
              'value': 6,
          },
          'full_gen_eval_samples_per_condition': {
              'value': 3,
          },
          'gen_eval_fast_sampling': {
              'value': True,
          },
          'max_steps': {
              'value': max_steps,
          },
      },
  }


def main(args):
  import wandb

  data_dirs = args.data_dirs or [_default_path(base_params.default_blast_data_dir)]
  params_csv = args.params_csv or _default_path(base_params.default_blast_params_csv)
  sweep_id = args.sweep_id or wandb.sweep(_sweep_config(args.max_steps, args.use_mamba, args.smoke), project=args.project)

  def train_one():
    with wandb.init(project=args.project) as run:
      run_params = AttrDict(base_params)
      run_params.override(dict(run.config))
      run_params.use_wandb = False
      run_model_dir = os.path.join(args.model_dir, run.id)
      train_args = Namespace(
          model_dir=run_model_dir,
          data_dirs=data_dirs,
          params_csv=params_csv,
          data_format='blast_csv',
          max_steps=run.config.max_steps,
          fp16=args.fp16,
          device=args.device)
      learner = train(train_args, run_params, wandb_run=run)
      if learner is None:
        return
      evaluator = learner.gen_evaluator
      if evaluator is None:
        dataset = getattr(getattr(learner, 'val_dataset', None), 'dataset', None)
        records = getattr(dataset, 'records', None)
        stats = getattr(dataset, 'stats', None)
        if not records or stats is None:
          return
        evaluator = BlastGenerationEvaluator(records, stats, learner.params)
      metrics = evaluator.evaluate_model(
          learner._raw_model(),
          device=next(learner.model.parameters()).device,
          subset_size=None,
          samples_per_condition=getattr(run_params, 'full_gen_eval_samples_per_condition', 3),
          fast_sampling=getattr(run_params, 'gen_eval_fast_sampling', True),
          seed=getattr(run_params, 'gen_eval_seed', 20260425),
          output_csv=os.path.join(run_model_dir, 'full_gen_eval.csv'))
      if metrics:
        run.log({
            'val/full_gen_score_mean': metrics['gen_score_mean'],
            'val/full_gen_score_best': metrics['gen_score_best'],
            'val/full_gen_invalid_metric_count': metrics.get('gen_invalid_metric_count_mean', 0.0),
        })

  wandb.agent(sweep_id, function=train_one, count=args.count, project=args.project)


if __name__ == '__main__':
  parser = ArgumentParser(description='run a W&B sweep for blast Mamba-DiffWave')
  parser.add_argument('--project', default=base_params.wandb_project)
  parser.add_argument('--sweep_id')
  parser.add_argument('--count', default=20, type=int)
  parser.add_argument('--max_steps', default=2000, type=int)
  parser.add_argument('--model_dir', default='sweep_runs')
  parser.add_argument('--params_csv')
  parser.add_argument('data_dirs', nargs='*')
  parser.add_argument('--device', default='cuda')
  parser.add_argument('--fp16', action='store_true', default=False)
  parser.add_argument('--use_mamba', type=_str_to_bool, default=True)
  parser.add_argument('--smoke', action='store_true', default=False)
  main(parser.parse_args())
