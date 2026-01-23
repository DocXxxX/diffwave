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

import numpy as np
import os
import torch
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from diffwave.dataset import from_path, from_gtzan, from_blast_data
from diffwave.model import DiffWave
from diffwave.params import AttrDict
from diffwave.physics_ops import PhysicsLoss


def _nested_map(struct, map_fn):
  if isinstance(struct, tuple):
    return tuple(_nested_map(x, map_fn) for x in struct)
  if isinstance(struct, list):
    return [_nested_map(x, map_fn) for x in struct]
  if isinstance(struct, dict):
    return { k: _nested_map(v, map_fn) for k, v in struct.items() }
  return map_fn(struct)


class DiffWaveLearner:
  def __init__(self, model_dir, model, dataset, optimizer, params, *args, **kwargs):
    os.makedirs(model_dir, exist_ok=True)
    self.model_dir = model_dir
    self.model = model
    self.dataset = dataset
    self.optimizer = optimizer
    self.params = params
    self.autocast = torch.cuda.amp.autocast(enabled=kwargs.get('fp16', False))
    self.scaler = torch.cuda.amp.GradScaler(enabled=kwargs.get('fp16', False))
    self.step = 0
    self.is_master = True

    beta = np.array(self.params.noise_schedule)
    noise_level = np.cumprod(1 - beta)
    self.noise_level = torch.tensor(noise_level.astype(np.float32))
    
    # Use PhysicsLoss if configured (for blast vibration)
    # Default to L1Loss if not specified or not applicable/configured
    if getattr(params, 'use_physics_condition', False):
         self.loss_fn = PhysicsLoss(sample_rate=params.sample_rate)
    else:
         self.loss_fn = nn.L1Loss()
         
    self.summary_writer = None

  def state_dict(self):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      model_state = self.model.module.state_dict()
    else:
      model_state = self.model.state_dict()
    return {
        'step': self.step,
        'model': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items() },
        'optimizer': { k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items() },
        'params': dict(self.params),
        'scaler': self.scaler.state_dict(),
    }

  def load_state_dict(self, state_dict):
    if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
      self.model.module.load_state_dict(state_dict['model'])
    else:
      self.model.load_state_dict(state_dict['model'])
    self.optimizer.load_state_dict(state_dict['optimizer'])
    self.scaler.load_state_dict(state_dict['scaler'])
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
      checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
      self.load_state_dict(checkpoint)
      return True
    except FileNotFoundError:
      return False

  def train(self, max_steps=None):
    device = next(self.model.parameters()).device
    while True:
      for features in tqdm(self.dataset, desc=f'Epoch {self.step // len(self.dataset)}') if self.is_master else self.dataset:
        if max_steps is not None and self.step >= max_steps:
          return
        features = _nested_map(features, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
        loss = self.train_step(features)
        if torch.isnan(loss).any():
          raise RuntimeError(f'Detected NaN loss at step {self.step}.')
        if self.is_master:
          if self.step % 50 == 0:
            self._write_summary(self.step, features, loss)
          if self.step % len(self.dataset) == 0:
            self.save_to_checkpoint()
        self.step += 1

  def train_step(self, features):
    for param in self.model.parameters():
      param.grad = None

    audio = features['audio']
    spectrogram = features.get('spectrogram')
    physical_params = features.get('physical_params')

    N, T = audio.shape
    device = audio.device
    self.noise_level = self.noise_level.to(device)

    with self.autocast:
      t = torch.randint(0, len(self.params.noise_schedule), [N], device=audio.device)
      noise_scale = self.noise_level[t].unsqueeze(1)
      noise_scale_sqrt = noise_scale**0.5
      noise = torch.randn_like(audio)
      noisy_audio = noise_scale_sqrt * audio + (1.0 - noise_scale)**0.5 * noise

      predicted = self.model(noisy_audio, t, spectrogram, physical_params)
      
      if isinstance(self.loss_fn, PhysicsLoss):
           loss, sub_losses = self.loss_fn(predicted.squeeze(1), noise, physical_params)
           # Store sub_losses for summary if needed?
           # For now just use total loss
      else:
           loss = self.loss_fn(noise, predicted.squeeze(1))

    self.scaler.scale(loss).backward()
    self.scaler.unscale_(self.optimizer)
    self.grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.params.max_grad_norm or 1e9)
    self.scaler.step(self.optimizer)
    self.scaler.update()
    return loss

  def _write_summary(self, step, features, loss):
    writer = self.summary_writer or SummaryWriter(self.model_dir, purge_step=step)
    writer.add_audio('feature/audio', features['audio'][0], step, sample_rate=self.params.sample_rate)
    
    if not self.params.unconditional:
        if features.get('spectrogram') is not None:
             writer.add_image('feature/spectrogram', torch.flip(features['spectrogram'][:1], [1]), step)
    
    writer.add_scalar('train/loss', loss, step)
    writer.add_scalar('train/grad_norm', self.grad_norm, step)
    writer.flush()
    self.summary_writer = writer


def _train_impl(replica_id, model, dataset, args, params):
  torch.backends.cudnn.benchmark = True
  opt = torch.optim.Adam(model.parameters(), lr=params.learning_rate)

  learner = DiffWaveLearner(args.model_dir, model, dataset, opt, params, fp16=args.fp16)
  learner.is_master = (replica_id == 0)
  learner.restore_from_checkpoint()
  learner.train(max_steps=args.max_steps)


def train(args, params):
  # Check if using blast data via arguments or presence of params file
  # Note: Modified to prefer blast data loader if params.csv is present in args (needs args support)
  # BUT: The original code passed args.data_dirs (list).
  # If the user provides a params CSV, we should use from_blast_data.
  # Let's check if the second arg is a csv or if a specific flag is set.
  # Since we don't control args parsing here (it's in __main__.py), we make an assumption:
  # If we see a file named '监测参数表.csv' in the parent dir of data_dirs[0] or passed explicitly?
  # Simplified: We assume usage of from_blast_data if params.use_physics_condition is True
  # and we need to find the params csv.
  # For now, let's look for a known params file or assume custom argument handling in main.
  
  # For this specific task, let's hardcode/heuristically detect for the user:
  # If data_dirs contains a csv file?
  
  # Assuming args has a 'params_csv' attribute if we modified the arg parser.
  # But we haven't modified __main__.py yet.
  # Let's assume standard usage for now unless specific Blast dataset is requested.
  
  # However, to support the user request "run modifications", I should ensure it works.
  # I will assume that IF 'use_physics_condition' is True, we use from_blast_data.
  # And we need the params file.
  
  # Let's check if a params file is provided in args (we might need to add it to main).
  # Or we hardcode the path as per the requirement since this is a specific project.
  # The requirement said "监测参数表.csv".
  
  if getattr(params, 'use_physics_condition', False):
      # Try to find params csv
      params_csv = '监测参数表.csv' # Default expected name
      # Check if it exists relative to data_dirs[0] or current dir
      if not os.path.exists(params_csv):
           # check data_dirs[0] parent
           potential = os.path.join(os.path.dirname(args.data_dirs[0]), params_csv)
           if os.path.exists(potential):
               params_csv = potential
      
      if os.path.exists(params_csv):
           print(f"Using Blast Vibration Dataset with params: {params_csv}")
           dataset = from_blast_data(args.data_dirs, params_csv, params)
      else:
           # Fallback or error?
           print(f"Warning: use_physics_condition is True but {params_csv} not found. Using standard loader.")
           if args.data_dirs[0] == 'gtzan':
                dataset = from_gtzan(params)
           else:
                dataset = from_path(args.data_dirs, params)
  elif args.data_dirs[0] == 'gtzan':
    dataset = from_gtzan(params)
  else:
    dataset = from_path(args.data_dirs, params)
    
  model = DiffWave(params).cuda()
  _train_impl(0, model, dataset, args, params)


def train_distributed(replica_id, replica_count, port, args, params):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = str(port)
  torch.distributed.init_process_group('nccl', rank=replica_id, world_size=replica_count)
  
  # Similar logic for dataset selection
  if getattr(params, 'use_physics_condition', False):
      params_csv = '监测参数表.csv'
      if not os.path.exists(params_csv):
           potential = os.path.join(os.path.dirname(args.data_dirs[0]), params_csv)
           if os.path.exists(potential):
               params_csv = potential
      
      if os.path.exists(params_csv):
           dataset = from_blast_data(args.data_dirs, params_csv, params, is_distributed=True)
      else:
           if args.data_dirs[0] == 'gtzan':
                dataset = from_gtzan(params, is_distributed=True)
           else:
                dataset = from_path(args.data_dirs, params, is_distributed=True)
  elif args.data_dirs[0] == 'gtzan':
    dataset = from_gtzan(params, is_distributed=True)
  else:
    dataset = from_path(args.data_dirs, params, is_distributed=True)
    
  device = torch.device('cuda', replica_id)
  torch.cuda.set_device(device)
  model = DiffWave(params).to(device)
  model = DistributedDataParallel(model, device_ids=[replica_id])
  _train_impl(replica_id, model, dataset, args, params)
