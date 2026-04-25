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

from argparse import ArgumentParser
from pathlib import Path
from torch.cuda import device_count
from torch.multiprocessing import spawn

from diffwave.learner import train, train_distributed
from diffwave.params import BLAST_AUGMENT_LEVELS, apply_blast_augment_level, params


def _get_free_port():
  import socketserver
  with socketserver.TCPServer(('localhost', 0), None) as s:
    return s.server_address[1]


def _str_to_bool(value):
  if isinstance(value, bool):
    return value
  value = value.lower()
  if value in ['true', '1', 'yes', 'y']:
    return True
  if value in ['false', '0', 'no', 'n']:
    return False
  raise ValueError(f'Invalid boolean value: {value}')


def _project_root():
  return Path(__file__).resolve().parents[2]


def _default_path(relative_path):
  return str(_project_root() / relative_path)


def _apply_overrides(args):
  override_names = [
      'batch_size',
      'learning_rate',
      'max_grad_norm',
      'audio_len',
      'audio_channels',
      'audio_clip',
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
      'split_seed',
      'val_ratio',
      'data_format',
      'blast_augment',
      'blast_augment_level',
      'blast_peak_jitter',
      'blast_time_shift',
      'blast_gain_min',
      'blast_gain_max',
      'use_wandb',
      'wandb_project',
      'wandb_run_name',
      'sample_clamp',
  ]
  overrides = { name: getattr(args, name) for name in override_names if getattr(args, name, None) is not None }
  augment_level = overrides.pop('blast_augment_level', None)
  apply_blast_augment_level(params, augment_level)
  params.override(overrides)


def main(args):
  if not args.data_dirs and args.data_format == 'blast_csv':
    args.data_dirs = [_default_path(params.default_blast_data_dir)]
  if not args.params_csv and args.data_format == 'blast_csv':
    args.params_csv = _default_path(params.default_blast_params_csv)
  _apply_overrides(args)

  replica_count = device_count() if args.device.startswith('cuda') else 0
  if replica_count > 1:
    if params.batch_size % replica_count != 0:
      raise ValueError(f'Batch size {params.batch_size} is not evenly divisble by # GPUs {replica_count}.')
    params.batch_size = params.batch_size // replica_count
    port = _get_free_port()
    spawn(train_distributed, args=(replica_count, port, args, params), nprocs=replica_count, join=True)
  else:
    train(args, params)


if __name__ == '__main__':
  parser = ArgumentParser(description='train (or resume training) a DiffWave model')
  parser.add_argument('model_dir',
      help='directory in which to store model checkpoints and training logs')
  parser.add_argument('data_dirs', nargs='*',
      help='space separated list of directories from which to read training files')
  parser.add_argument('--params_csv',
      help='blast parameter CSV path')
  parser.add_argument('--data_format', default='blast_csv', choices=['blast_csv', 'path'],
      help='training data format')
  parser.add_argument('--blast_augment', type=_str_to_bool)
  parser.add_argument('--blast_augment_level', choices=list(BLAST_AUGMENT_LEVELS.keys()))
  parser.add_argument('--blast_peak_jitter', type=int)
  parser.add_argument('--blast_time_shift', type=int)
  parser.add_argument('--blast_gain_min', type=float)
  parser.add_argument('--blast_gain_max', type=float)
  parser.add_argument('--max_steps', default=None, type=int,
      help='maximum number of training steps')
  parser.add_argument('--fp16', action='store_true', default=False,
      help='use 16-bit floating point operations for training')
  parser.add_argument('--device', default='cuda',
      help='training device, e.g. cuda, cuda:0, or cpu')

  parser.add_argument('--batch_size', type=int)
  parser.add_argument('--learning_rate', type=float)
  parser.add_argument('--max_grad_norm', type=float)
  parser.add_argument('--audio_len', type=int)
  parser.add_argument('--audio_channels', type=int)
  parser.add_argument('--audio_clip', type=float)
  parser.add_argument('--condition_dim', type=int)
  parser.add_argument('--residual_layers', type=int)
  parser.add_argument('--residual_channels', type=int)
  parser.add_argument('--dilation_cycle_length', type=int)
  parser.add_argument('--use_mamba', type=_str_to_bool)
  parser.add_argument('--mamba_d_state', type=int)
  parser.add_argument('--mamba_expand', type=int)
  parser.add_argument('--film_hidden_dim', type=int)
  parser.add_argument('--loss_type', choices=['l1', 'mse'])
  parser.add_argument('--validation_interval', type=int)
  parser.add_argument('--validation_batches', type=int)
  parser.add_argument('--checkpoint_interval', type=int)
  parser.add_argument('--num_workers', type=int)
  parser.add_argument('--split_seed', type=int)
  parser.add_argument('--val_ratio', type=float)
  parser.add_argument('--sample_clamp', type=_str_to_bool)

  parser.add_argument('--use_wandb', action='store_true', default=None,
      help='log training metrics to W&B')
  parser.add_argument('--wandb_project')
  parser.add_argument('--wandb_run_name')
  main(parser.parse_args())
