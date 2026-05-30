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
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

try:
  from torch.utils.tensorboard import SummaryWriter
except ImportError:
  SummaryWriter = None

from diffwave.dataset import create_blast_dataloaders, from_gtzan, from_path
from diffwave.model import DiffWave
from diffwave.physics_ops import PhysicsLoss
from diffwave.blast_eval import BlastGenerationEvaluator, validate_signal_config


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


class _NoOpScaler:
  def scale(self, loss):
    return loss

  def unscale_(self, optimizer):
    return None

  def step(self, optimizer):
    optimizer.step()

  def update(self):
    return None

  def state_dict(self):
    return {}

  def load_state_dict(self, state_dict):
    return None


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
    if next(model.parameters()).device.type == 'cpu' and hasattr(torch.backends, 'mkldnn'):
      torch.backends.mkldnn.enabled = False
    use_amp = bool(kwargs.get('fp16', False)) and next(model.parameters()).device.type == 'cuda'
    self.autocast = torch.cuda.amp.autocast(enabled=True) if use_amp else nullcontext()
    self.scaler = torch.cuda.amp.GradScaler(enabled=True) if use_amp else _NoOpScaler()
    self.step = 0
    self.is_master = True
    self.grad_norm = torch.tensor(0.0)

    beta = np.array(self.params.noise_schedule)
    noise_level = np.cumprod(1 - beta)
    self.noise_level = torch.tensor(noise_level.astype(np.float32))
    self.loss_fn = _loss_fn(params)
    self.summary_writer = None
    self.gen_evaluator = self._create_gen_evaluator()

  def _create_gen_evaluator(self):
    interval = getattr(self.params, 'gen_eval_interval', 0)
    if not interval or self.val_dataset is None:
      return None
    dataset = getattr(self.val_dataset, 'dataset', None)
    records = getattr(dataset, 'records', None)
    stats = getattr(dataset, 'stats', None)
    if not records or stats is None:
      return None
    return BlastGenerationEvaluator(records, stats, self.params)

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
      try:
        self.load_state_dict(checkpoint)
      except RuntimeError as exc:
        if 'scale_predictor' not in str(exc):
          raise
        model_state = checkpoint['model']
        missing, unexpected = self.model.load_state_dict(model_state, strict=False)
        other_missing = [key for key in missing if not key.startswith('scale_predictor.')]
        if other_missing or unexpected:
          raise
        try:
          self.optimizer.load_state_dict(checkpoint['optimizer'])
        except ValueError:
          print('[WARN] Optimizer state is incompatible with the current model; using a fresh optimizer.')
        self.scaler.load_state_dict(checkpoint.get('scaler', {}))
        self.step = checkpoint['step']
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
          return self
        features = self._to_device(features)
        loss, loss_details = self.train_step(features)
        if torch.isnan(loss).any():
          raise RuntimeError(f'Detected NaN loss at step {self.step}.')
        if self.is_master:
          if self.step % 50 == 0:
            self._write_summary(self.step, features, loss, loss_details)
          if self.val_dataset is not None and validation_interval and self.step % validation_interval == 0:
            self.validate()
          gen_eval_interval = getattr(self.params, 'gen_eval_interval', 0)
          if self.gen_evaluator is not None and gen_eval_interval and self.step > 0 and self.step % gen_eval_interval == 0:
            self.evaluate_generation()
          if self.step % checkpoint_interval == 0:
            self.save_to_checkpoint()
        self.step += 1

  def _to_device(self, features):
    device = next(self.model.parameters()).device
    return _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)

  def _raw_model(self):
    return self.model.module if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module) else self.model

  def _stft_loss(self, predicted, target):
    _, fft_sizes = validate_signal_config(
        int(self.params.sample_rate),
        int(predicted.shape[-1]),
        fft_sizes=getattr(self.params, 'gen_eval_stft_fft_sizes', None))
    flat_pred = predicted.reshape(-1, predicted.shape[-1])
    flat_target = target.reshape(-1, target.shape[-1])
    losses = []
    for n_fft in fft_sizes:
      hop = n_fft // 4
      window = torch.hann_window(n_fft, device=predicted.device, dtype=predicted.dtype)
      pred_spec = torch.stft(
          flat_pred,
          n_fft=n_fft,
          hop_length=hop,
          win_length=n_fft,
          window=window,
          return_complex=True)
      target_spec = torch.stft(
          flat_target,
          n_fft=n_fft,
          hop_length=hop,
          win_length=n_fft,
          window=window,
          return_complex=True)
      losses.append(torch.mean(torch.abs(torch.log(torch.abs(pred_spec) + 1e-7) - torch.log(torch.abs(target_spec) + 1e-7))))
    return torch.stack(losses).mean()

  def _band_energy_loss(self, predicted, target):
    bands, _ = validate_signal_config(
        int(self.params.sample_rate),
        int(predicted.shape[-1]),
        bands=getattr(self.params, 'gen_eval_freq_bands', None))
    spectrum_pred = torch.fft.rfft(predicted, dim=-1)
    spectrum_target = torch.fft.rfft(target, dim=-1)
    power_pred = torch.square(torch.abs(spectrum_pred))
    power_target = torch.square(torch.abs(spectrum_target))
    freqs = torch.fft.rfftfreq(predicted.shape[-1], d=1.0 / float(self.params.sample_rate)).to(predicted.device)
    pred_bands = []
    target_bands = []
    for index, (low, high) in enumerate(zip(bands[:-1], bands[1:])):
      if index == len(bands) - 2:
        mask = (freqs >= low) & (freqs <= high)
      else:
        mask = (freqs >= low) & (freqs < high)
      if torch.any(mask):
        pred_bands.append(power_pred[..., mask].sum(dim=-1))
        target_bands.append(power_target[..., mask].sum(dim=-1))
      else:
        pred_bands.append(torch.zeros_like(power_pred[..., 0]))
        target_bands.append(torch.zeros_like(power_target[..., 0]))
    pred_energy = torch.stack(pred_bands, dim=-1)
    target_energy = torch.stack(target_bands, dim=-1)
    pred_energy = pred_energy / (pred_energy.sum(dim=-1, keepdim=True) + 1e-8)
    target_energy = target_energy / (target_energy.sum(dim=-1, keepdim=True) + 1e-8)
    return torch.mean(0.5 * torch.sum(torch.abs(pred_energy - target_energy), dim=-1))

  def _envelope_loss(self, predicted, target):
    kernel = max(8, int(self.params.sample_rate // 200))
    kernel = min(kernel, predicted.shape[-1])
    pred_env = F.avg_pool1d(torch.abs(predicted), kernel_size=kernel, stride=kernel, ceil_mode=True)
    target_env = F.avg_pool1d(torch.abs(target), kernel_size=kernel, stride=kernel, ceil_mode=True)
    pred_centered = pred_env - pred_env.mean(dim=-1, keepdim=True)
    target_centered = target_env - target_env.mean(dim=-1, keepdim=True)
    corr = torch.sum(pred_centered * target_centered, dim=-1) / (
        torch.linalg.norm(pred_centered, dim=-1) * torch.linalg.norm(target_centered, dim=-1) + 1e-8)
    return torch.mean(1.0 - torch.clamp(corr, min=0.0, max=1.0))

  def _peak_rms_loss(self, predicted, target):
    pred_rms = torch.sqrt(torch.mean(torch.square(predicted), dim=-1) + 1e-8)
    target_rms = torch.sqrt(torch.mean(torch.square(target), dim=-1) + 1e-8)
    pred_peak = torch.max(torch.abs(predicted), dim=-1).values
    target_peak = torch.max(torch.abs(target), dim=-1).values
    pred_ptp = torch.max(predicted, dim=-1).values - torch.min(predicted, dim=-1).values
    target_ptp = torch.max(target, dim=-1).values - torch.min(target, dim=-1).values
    return (
        torch.mean(torch.abs(torch.log((pred_rms + 1e-8) / (target_rms + 1e-8)))) +
        torch.mean(torch.abs(torch.log((pred_peak + 1e-8) / (target_peak + 1e-8)))) +
        torch.mean(torch.abs(torch.log((pred_ptp + 1e-8) / (target_ptp + 1e-8)))))

  def _aux_mask(self, t, noise_scale):
    timestep_max = float(getattr(self.params, 'aux_loss_timestep_max_ratio', 0.6)) * (len(self.params.noise_schedule) - 1)
    snr = noise_scale.flatten() / (1.0 - noise_scale.flatten() + 1e-8)
    min_snr = float(getattr(self.params, 'aux_loss_min_snr', 0.1))
    return (t.float() <= timestep_max) & (snr >= min_snr)

  def _warmup_factor(self):
    warmup_steps = int(getattr(self.params, 'aux_loss_warmup_steps', 0) or 0)
    if warmup_steps <= 0:
      return 1.0
    return min(1.0, float(self.step) / float(warmup_steps))

  def _diffusion_loss(self, features, return_details=False):
    audio = features['audio']
    spectrogram = features.get('spectrogram')
    physical_params = features.get('physical_params')
    scale_target = features.get('scale_target')

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
    else:
      loss = self.loss_fn(noise, predicted)
    losses = {'loss/diffusion': loss.detach()}

    warmup = self._warmup_factor()
    aux_mask = self._aux_mask(t, noise_scale)
    if warmup > 0 and torch.any(aux_mask):
      x0_hat = (noisy_audio - (1.0 - noise_scale)**0.5 * predicted) / (noise_scale_sqrt + 1e-8)
      x0_hat = x0_hat[aux_mask]
      clean_audio = audio[aux_mask]
      aux_specs = [
          ('loss/mr_stft', 'lambda_mr_stft', self._stft_loss),
          ('loss/band_energy', 'lambda_band_energy', self._band_energy_loss),
          ('loss/envelope', 'lambda_envelope', self._envelope_loss),
          ('loss/peak_rms', 'lambda_peak_rms', self._peak_rms_loss),
      ]
      for loss_name, weight_name, fn in aux_specs:
        weight = float(getattr(self.params, weight_name, 0.0) or 0.0) * warmup
        if weight <= 0:
          continue
        aux_loss = fn(x0_hat, clean_audio)
        loss = loss + weight * aux_loss
        losses[loss_name] = aux_loss.detach()
        losses[f'{loss_name}_weighted'] = (weight * aux_loss).detach()

    scale_weight = float(getattr(self.params, 'lambda_scale', 0.0) or 0.0)
    raw_model = self._raw_model()
    if scale_weight > 0 and scale_target is not None and hasattr(raw_model, 'predict_scale'):
      predicted_scale = raw_model.predict_scale(physical_params)
      if predicted_scale is not None:
        scale_loss = F.smooth_l1_loss(predicted_scale, scale_target)
        loss = loss + scale_weight * scale_loss
        losses['loss/scale'] = scale_loss.detach()
        losses['loss/scale_weighted'] = (scale_weight * scale_loss).detach()

    losses['loss/total'] = loss.detach()
    if return_details:
      return loss, losses
    return loss

  def train_step(self, features):
    for param in self.model.parameters():
      param.grad = None

    with self.autocast:
      loss, loss_details = self._diffusion_loss(features, return_details=True)

    self.scaler.scale(loss).backward()
    self.scaler.unscale_(self.optimizer)
    self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm or 1e9)
    self.scaler.step(self.optimizer)
    self.scaler.update()
    return loss.detach(), loss_details

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

  def evaluate_generation(self):
    if self.gen_evaluator is None:
      return None
    device = next(self.model.parameters()).device
    metrics = self.gen_evaluator.evaluate_model(
        self._raw_model(),
        device=device,
        subset_size=getattr(self.params, 'gen_eval_subset_size', 6),
        samples_per_condition=getattr(self.params, 'gen_eval_samples_per_condition', 1),
        fast_sampling=getattr(self.params, 'gen_eval_fast_sampling', True),
        seed=getattr(self.params, 'gen_eval_seed', 20260425))
    if not metrics:
      return None
    log_metrics = {
        'val/fast_gen_score': metrics['gen_score_mean'],
        # W&B Hyperband watches the sweep metric; final full eval overwrites this proxy.
        'val/full_gen_score_mean': metrics['gen_score_mean'],
        'val/fast_gen_invalid_metric_count': metrics.get('gen_invalid_metric_count_mean', 0.0),
        'step': self.step,
    }
    for name, value in metrics.items():
      if name.endswith('_mean') and name != 'gen_score_mean':
        log_metrics[f'val/fast_{name[:-5]}'] = value
    if self.summary_writer:
      self.summary_writer.add_scalar('val/fast_gen_score', metrics['gen_score_mean'], self.step)
    if self.wandb_run:
      self.wandb_run.log(log_metrics, step=self.step)
    return metrics

  def _write_summary(self, step, features, loss, loss_details=None):
    if SummaryWriter is not None:
      writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
      preview = torch.clamp(features['audio'][0, 0].detach().cpu(), -1.0, 1.0)
      writer.add_audio('feature/audio_ch1', preview, step, sample_rate=self.params.sample_rate)
      writer.add_scalar('train/loss', loss, step)
      writer.add_scalar('train/grad_norm', self.grad_norm, step)
      if loss_details:
        for name, value in loss_details.items():
          writer.add_scalar(name, float(value.detach().cpu()), step)
      writer.flush()
      self.summary_writer = writer

    if self.wandb_run:
      metrics = {
          'train/loss': float(loss.detach().cpu()),
          'train/grad_norm': float(self.grad_norm.detach().cpu()),
          'step': step,
      }
      if loss_details:
        metrics.update({name: float(value.detach().cpu()) for name, value in loss_details.items()})
      self.wandb_run.log(metrics, step=step)


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
  return learner


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
    return _train_impl(0, model, dataset, val_dataset, args, params, wandb_run=wandb_run)
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
