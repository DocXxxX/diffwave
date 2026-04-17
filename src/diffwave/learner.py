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
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

try:
  from torch.utils.tensorboard import SummaryWriter
except ImportError:
  SummaryWriter = None

from diffwave.dataset import create_blast_dataloaders, from_gtzan, from_path
from diffwave.model import DiffWave
from diffwave.physics_ops import PhysicsLoss


def _nested_map(struct, map_fn):
  if isinstance(struct, tuple):
    return tuple(_nested_map(x, map_fn) for x in struct)
  if isinstance(struct, list):
    return [_nested_map(x, map_fn) for x in struct]
  if isinstance(struct, dict):
    return { k: _nested_map(v, map_fn) for k, v in struct.items() }
  return map_fn(struct)


def _device_from_args(args, replica_id=None):
  requested = getattr(args, 'device', 'cuda')
  if requested.startswith('cuda') and not torch.cuda.is_available():
    return torch.device('cpu')
  if replica_id is not None and requested.startswith('cuda'):
    return torch.device('cuda', replica_id)
  return torch.device(requested)


def _loss_fn(params):
  if getattr(params, 'use_physics_loss', False):
    return PhysicsLoss(sample_rate=params.sample_rate)
  if getattr(params, 'loss_type', 'l1') == 'mse':
    return nn.MSELoss()
  return nn.L1Loss()


class DiffWaveLearner:
  def __init__(self, model_dir, model, dataset, optimizer, params, val_dataset=None, wandb_run=None, *args, **kwargs):
    os.makedirs(model_dir, exist_ok=True)
    self.model_dir = model_dir
    self.model = model
    self.dataset = dataset
    self.val_dataset = val_dataset
    self.optimizer = optimizer
    self.params = params
    self.wandb_run = wandb_run
    self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))
    self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))
    self.step = 0
    self.is_master = True
    self.grad_norm = torch.tensor(0.0)

    beta = np.array(self.params.noise_schedule)
    noise_level = np.cumprod(1 - beta)
    self.noise_level = torch.tensor(noise_level.astype(np.float32))
    self.loss_fn = _loss_fn(params)
    self.summary_writer = None

  def state_dict(self):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      model_state = self.model.module.state_dict()
    else:
      model_state = self.model.state_dict()
    return {
        'step': self.step,
        'model': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items() },
        'optimizer': self.optimizer.state_dict(),
        'params': dict(self.params),
        'scaler': self.scaler.state_dict(),
    }

  def load_state_dict(self, state_dict):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      self.model.module.load_state_dict(state_dict['model'])
    else:
      self.model.load_state_dict(state_dict['model'])
    self.optimizer.load_state_dict(state_dict['optimizer'])
    self.scaler.load_state_dict(state_dict.get('scaler', {}))
    self.step = state_dict['step']

  def save_to_checkpoint(self, filename='weights'):
    save_basename = f'{filename}-{self.step}.pt'
    save_name = f'{self.model_dir}/{save_basename}'
    link_name = f'{self.model_dir}/{filename}.pt'
    torch.save(self.state_dict(), save_name)
    if os.name == 'nt':
      torch.save(self.state_dict(), link_name)
    else:
      if os.path.islink(link_name):
        os.unlink(link_name)
      os.symlink(save_basename, link_name)

  def restore_from_checkpoint(self, filename='weights'):
    try:
      checkpoint = torch.load(f'{self.model_dir}/{filename}.pt', map_location=next(self.model.parameters()).device)
      self.load_state_dict(checkpoint)
      return True
    except FileNotFoundError:
      return False

  def train(self, max_steps=None):
    if len(self.dataset) == 0:
      raise ValueError('Training dataset is empty.')

    checkpoint_interval = self.params.checkpoint_interval or len(self.dataset)
    validation_interval = getattr(self.params, 'validation_interval', 0)
    self.model.train()

    while True:
      epoch = self.step // len(self.dataset)
      iterator = tqdm(self.dataset, desc=f'Epoch {epoch}') if self.is_master else self.dataset
      for features in iterator:
        if max_steps is not None and self.step >= max_steps:
          if self.is_master:
            self.save_to_checkpoint()
          return
        features = self._to_device(features)
        loss = self.train_step(features)
        if torch.isnan(loss).any():
          raise RuntimeError(f'Detected NaN loss at step {self.step}.')
        if self.is_master:
          if self.step % 50 == 0:
            self._write_summary(self.step, features, loss)
          if self.val_dataset is not None and validation_interval and self.step % validation_interval == 0:
            self.validate()
          if self.step % checkpoint_interval == 0:
            self.save_to_checkpoint()
        self.step += 1

  def _to_device(self, features):
    device = next(self.model.parameters()).device
    return _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)

  def _diffusion_loss(self, features):
    audio = features['audio']
    spectrogram = features.get('spectrogram')
    physical_params = features.get('physical_params')

    batch_size = audio.shape[0]
    device = audio.device
    self.noise_level = self.noise_level.to(device)

    t = torch.randint(0, len(self.params.noise_schedule), [batch_size], device=audio.device)
    view_shape = [batch_size] + [1] * (audio.dim() - 1)
    noise_scale = self.noise_level[t].view(*view_shape)
    noise_scale_sqrt = noise_scale**0.5
    noise = torch.randn_like(audio)
    noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale)**0.5 * noise
    predicted = self.model(noisy_audio, t, spectrogram, physical_params)

    if isinstance(self.loss_fn, PhysicsLoss):
      loss, _ = self.loss_fn(predicted[:, 0], noise[:, 0], physical_params)
      return loss
    return self.loss_fn(noise, predicted)

  def train_step(self, features):
    for param in self.model.parameters():
      param.grad = None

    with self.autocast:
      loss = self._diffusion_loss(features)

    self.scaler.scale(loss).backward()
    self.scaler.unscale_(self.optimizer)
    self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm or 1e9)
    self.scaler.step(self.optimizer)
    self.scaler.update()
    return loss.detach()

  def validate(self):
    self.model.eval()
    total_loss = 0.0
    count = 0
    max_batches = getattr(self.params, 'validation_batches', 8)
    with torch.no_grad():
      for features in self.val_dataset:
        if max_batches and count >= max_batches:
          break
        loss = self._diffusion_loss(self._to_device(features))
        total_loss += float(loss.detach().cpu())
        count += 1
    self.model.train()
    if count == 0:
      return None
    val_loss = total_loss / count
    if self.summary_writer:
      self.summary_writer.add_scalar('val/loss', val_loss, self.step)
    if self.wandb_run:
      self.wandb_run.log({'val/loss': val_loss, 'step': self.step}, step=self.step)
    return val_loss

  def _write_summary(self, step, features, loss):
    if SummaryWriter is not None:
      writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
      preview = torch.clamp(features['audio'][0, 0].detach().cpu(), -1.0, 1.0)
      writer.add_audio('feature/audio_ch1', preview, step, sample_rate=self.params.sample_rate)
      writer.add_scalar('train/loss', loss, step)
      writer.add_scalar('train/grad_norm', self.grad_norm, step)
      writer.flush()
      self.summary_writer = writer

    if self.wandb_run:
      self.wandb_run.log({
          'train/loss': float(loss.detach().cpu()),
          'train/grad_norm': float(self.grad_norm.detach().cpu()),
          'step': step,
      }, step=step)


def _create_loaders(args, params, is_distributed=False):
  data_format = getattr(args, 'data_format', getattr(params, 'data_format', 'path'))
  if data_format == 'blast_csv':
    if not getattr(args, 'params_csv', None):
      raise ValueError('--params_csv is required for blast_csv training.')
    return create_blast_dataloaders(
        args.data_dirs,
        args.params_csv,
        params,
        model_dir=args.model_dir,
        is_distributed=is_distributed)
  if args.data_dirs[0] == 'gtzan':
    return from_gtzan(params, is_distributed=is_distributed), None
  return from_path(args.data_dirs, params, is_distributed=is_distributed), None


def _train_impl(replica_id, model, dataset, val_dataset, args, params, wandb_run=None):
  torch.backends.cudnn.benchmark = True
  opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

  learner = DiffWaveLearner(
      args.model_dir,
      model,
      dataset,
      opt,
      params,
      val_dataset=val_dataset,
      wandb_run=wandb_run,
      fp16=args.fp16)
  learner.is_master = (replica_id == 0)
  learner.restore_from_checkpoint()
  learner.train(max_steps=args.max_steps)


def train(args, params, wandb_run=None):
  created_run = None
  if getattr(params, 'use_wandb', False) and wandb_run is None:
    try:
      import wandb
    except ImportError as exc:
      raise ImportError('wandb is required when use_wandb=True') from exc
    created_run = wandb.init(
        project=params.wandb_project,
        name=params.wandb_run_name,
        config=dict(params))
    wandb_run = created_run

  try:
    dataset, val_dataset = _create_loaders(args, params)
    device = _device_from_args(args)
    model = DiffWave(params).to(device)
    _train_impl(0, model, dataset, val_dataset, args, params, wandb_run=wandb_run)
  finally:
    if created_run is not None:
      created_run.finish()


def train_distributed(replica_id, replica_count, port, args, params):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = str(port)
  torch.distributed.init_process_group('nccl', rank=replica_id, world_size=replica_count)

  dataset, val_dataset = _create_loaders(args, params, is_distributed=True)
  device = _device_from_args(args, replica_id)
  torch.cuda.set_device(device)
  model = DiffWave(params).to(device)
  model = DistributedDataParallel(model, device_ids=[replica_id])
  _train_impl(replica_id, model, dataset, val_dataset, args, params)
