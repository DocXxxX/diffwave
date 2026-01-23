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
import soundfile as sf

from argparse import ArgumentParser

from diffwave.params import AttrDict, params as base_params
from diffwave.model import DiffWave


models = {}

def predict(spectrogram=None, physical_params=None, model_dir=None, params=None, device=torch.device('cuda'), fast_sampling=False):
  # Lazy load model.
  if not model_dir in models:
    if os.path.exists(f'{model_dir}/weights.pt'):
      checkpoint = torch.load(f'{model_dir}/weights.pt')
    else:
      checkpoint = torch.load(model_dir)
    model = DiffWave(AttrDict(base_params)).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    models[model_dir] = model

  model = models[model_dir]
  model.params.override(params)
  
  # Handle physical_params input
  if physical_params is not None:
       if isinstance(physical_params, list) or isinstance(physical_params, tuple):
            physical_params = torch.tensor(physical_params).float()
       if isinstance(physical_params, np.ndarray):
            physical_params = torch.from_numpy(physical_params).float()
            
       # Ensure batch dim
       if physical_params.dim() == 1:
            physical_params = physical_params.unsqueeze(0)
       
       physical_params = physical_params.to(device)
       batch_size = physical_params.shape[0]
  else:
       # Default batch size 1 if no params provided and unconditional (or if spectrogram provided)
       batch_size = 1
       if spectrogram is not None:
             batch_size = spectrogram.shape[0] if spectrogram.dim() > 2 else 1

  with torch.no_grad():
    # Change in notation from the DiffWave paper for fast sampling.
    # DiffWave paper -> Implementation below
    # --------------------------------------
    # alpha -> talpha
    # beta -> training_noise_schedule
    # gamma -> alpha
    # eta -> beta
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


    if not model.params.unconditional:
      if spectrogram is not None:
          if len(spectrogram.shape) == 2:# Expand rank 2 tensors by adding a batch dimension.
            spectrogram = spectrogram.unsqueeze(0)
          spectrogram = spectrogram.to(device)
          audio = torch.randn(spectrogram.shape[0], model.params.hop_samples * spectrogram.shape[-1], device=device)
      else:
          # Use audio_len for physical params generation
          audio = torch.randn(batch_size, model.params.audio_len, device=device)
    else:
      audio = torch.randn(1, params.audio_len, device=device)
      
    noise_scale = torch.from_numpy(alpha_cum**0.5).float().unsqueeze(1).to(device)

    for n in range(len(alpha) - 1, -1, -1):
      c1 = 1 / alpha[n]**0.5
      c2 = beta[n] / (1 - alpha_cum[n])**0.5
      
      # Pass physical_params to model
      pred = model(audio, torch.tensor([T[n]], device=audio.device), spectrogram, physical_params).squeeze(1)
      
      audio = c1 * (audio - c2 * pred)
      if n > 0:
        noise = torch.randn_like(audio)
        sigma = ((1.0 - alpha_cum[n-1]) / (1.0 - alpha_cum[n]) * beta[n])**0.5
        audio += sigma * noise
      audio = torch.clamp(audio, -1.0, 1.0)
  return audio, model.params.sample_rate


def main(args):
  if args.spectrogram_path:
    spectrogram = torch.from_numpy(np.load(args.spectrogram_path))
  else:
    spectrogram = None
  
  physical_params = None
  if args.physical_params:
       # Parse comma separated string "2.0,60.0,0.0,100,25"
       try:
            params_list = [float(x) for x in args.physical_params.split(',')]
            physical_params = torch.tensor(params_list)
            print(f"Generating with physical params: {physical_params}")
       except ValueError:
            print("Error parsing physical params. Expected format: '2.0,60.0,0.0,100,25'")

  audio, sr = predict(spectrogram, physical_params=physical_params, model_dir=args.model_dir, fast_sampling=args.fast, params=base_params)
  sf.write(args.output, audio.cpu().numpy().squeeze(), sr)


if __name__ == '__main__':
  parser = ArgumentParser(description='runs inference on a spectrogram file generated by diffwave.preprocess')
  parser.add_argument('model_dir',
      help='directory containing a trained model (or full path to weights.pt file)')
  parser.add_argument('--spectrogram_path', '-s',
      help='path to a spectrogram file generated by diffwave.preprocess')
  parser.add_argument('--physical_params', '-p',
      help='comma separated list of physical parameters: Q_max, Distance_R, Elev_Diff, Hole_Num, Delay_Int (e.g. "2.0,60.0,0.0,100,25")')
  parser.add_argument('--output', '-o', default='output.wav',
      help='output file name')
  parser.add_argument('--fast', '-f', action='store_true',
      help='fast sampling procedure')
  main(parser.parse_args())
