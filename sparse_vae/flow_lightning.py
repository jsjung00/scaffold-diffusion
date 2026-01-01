# Adopted from Trellis code 

from math import sqrt 
from typing import Callable, Iterator, Tuple, Optional, Type, Union, List, ClassVar

import lightning as L 
import numpy as np 
from easydict import EasyDict as edict
import torch 
import torch.nn.functional as F 
import torchvision.utils
from torch import nn 
from diffusion_backbone import SparseStructureDiffusionModel
from vae_lightning import SparseStructureVAE
from flow_sampler import FlowEulerSampler
import itertools
import sys 
from pathlib import Path 
sys.path.append(Path(__file__).parent.parent)
from models.mdlm_ema import ExponentialMovingAverage

from torch.nn.parameter import Parameter 
import hydra 


class SparseFlow(L.LightningModule):
    '''
    Flow matching on latent structures
    '''
    def __init__(
        self,
        config,
        t_schedule = {
            'name':'logitNormal',
            'args': {
                'mean': 1.0, # 0.0
                'std': 1.0
            }
        },
        sigma_min: float = 1e-5
    ):
        super().__init__()
        self.save_hyperparameters()
        self.config = config 
        self.t_schedule = t_schedule 
        self.sigma_min = sigma_min 
        backbone_config = self.config.model.backbone 
        self.denoiser_module = SparseStructureDiffusionModel(backbone_config)

        # VAE encoder
        vae_path = self.config.latent.vae_ckpt
        self.vae = SparseStructureVAE.load_from_checkpoint(vae_path)
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

        if self.config.training.ema > 0:
            self.ema = ExponentialMovingAverage(
                self.denoiser_module.parameters(),
                decay=self.config.training.ema
            )
        else:
            self.ema = None  

    def setup(self, stage, dtype=None):
        if dtype is None:
            self.denoiser_module.set_dtype(self.dtype)
            self.vae.to(self.dtype)
            self.vae.encoder.set_dtype(self.dtype)
            self.vae.decoder.set_dtype(self.dtype)
        else:
            self.denoiser_module.set_dtype(dtype)
            self.vae.to(dtype)
            self.vae.encoder.set_dtype(dtype)
            self.vae.decoder.set_dtype(dtype)

    
    def diffuse(self, x_0: torch.Tensor, t: torch.Tensor, noise=None):
        '''
        Diffuse data for a given number of diffusion steps. 
        Sample from q(x_t | x_0)
        
        Returns:
            x_t, noisy version of x_0 under timestep t
        
        x_0: [NxCx...] tensor of noiseless inputs
        t: [N] Tensor of diffusion steps in [0,1]
        noise: if given, use this noise
        '''
        if noise is None:
            noise = torch.randn_like(x_0)
        assert noise.shape == x_0.shape, "noise must have same shape as x_0"

        t = t.view(-1, *[1 for _ in range(len(x_0.shape)-1)])
        x_t = (1-t) * x_0 + (self.sigma_min + (1 - self.sigma_min) * t) * noise

        return x_t 
    
    def reverse_diffuse(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        '''
        Get original image from noisy version under timestep t
        '''
        assert noise.shape == x_t.shape, "noise must be same shape as x"
        t = t.view(-1, *[1 for _ in range(len(x_t.shape)-1)])
        x_0 = (x_t - (self.sigma_min + (1-self.sigma_min)*t)*noise) / (1-t)
        return x_0
    
    def get_v(self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor):
        '''
        Compute the velocity of the diffusion process at timestep t
        '''
        return (1-self.sigma_min)*noise - x_0 
    
    def get_sampler(self):
        return FlowEulerSampler(self.sigma_min)
    
    def sample_t(self, batch_size:int):
        '''
        Get the sampler for the diffusion process
        '''
        if self.t_schedule['name'] == 'uniform':
            t = torch.rand(batch_size)
        elif self.t_schedule['name'] == 'logitNormal':
            mean = self.t_schedule['args']['mean']
            std = self.t_schedule['args']['std']
            t = torch.sigmoid(torch.randn(batch_size)*std + mean)
        else:
            raise ValueError(f"unknown t_schedule: {self.t_schedule['name']}")

        return t       
    
    def compute_losses(
        self,
        x_0: torch.Tensor, 
        cond=None,
        prefix=''
    ):
        '''
        Compute training losses for a single timestep 
        '''
        noise = torch.randn_like(x_0)
        t = self.sample_t(x_0.shape[0]).to(x_0.device).float()
        x_t = self.diffuse(x_0, t, noise=noise)

        pred = self.denoiser_module(x_t, t*1000)
        assert pred.shape == noise.shape == x_0.shape 
        target = self.get_v(x_0, noise, t)
        terms = edict()
        prefix_str = f"{prefix}/" if prefix else ""
        terms[f'{prefix_str}mse'] = F.mse_loss(pred, target)
        terms[f'{prefix_str}loss'] = terms[f'{prefix_str}mse']

        # log loss with time bins
        mse_per_instance = np.array([
            F.mse_loss(pred[i], target[i]).item() for i in range(x_0.shape[0])
        ])
        time_bin = np.digitize(t.cpu().numpy(), np.linspace(0,1,11)) - 1
        for i in range(10):
            if (time_bin == i).sum() != 0:
                terms[f"{prefix_str}bin_{i}_mse"] = mse_per_instance[time_bin == i].mean()

        return terms

    def training_step(self, batch, batch_idx):
        batch = torch.unsqueeze(batch, dim=1) #(B,1,H,W,D)
        batch = batch.to(self.device)
        with torch.no_grad():
            z = self.vae.encode(batch, sample=False, return_raw=False) #sample=True
        loss_dict = self.compute_losses(z, prefix="train")

        self.log_dict(loss_dict, on_step=True, on_epoch=True, prog_bar=True)
        return loss_dict['train/loss']

    def validation_step(self, batch, batch_idx):
        batch = torch.unsqueeze(batch, dim=1) #(B,1,H,W,D)
        batch = batch.to(self.device)
        with torch.no_grad():
            z = self.vae.encode(batch, sample=False, return_raw=False) #sample=True
        loss_dict = self.compute_losses(z, prefix="val")
        self.log_dict(loss_dict, on_step=False, on_epoch=True, prog_bar=False)
        return loss_dict['val/loss']

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=self.config.optim.lr, #0.0001
            weight_decay= self.config.optim.weight_decay, #0
        )

        scheduler = hydra.utils.instantiate(
            self.config.lr_scheduler, optimizer=optimizer
        )
        scheduler_dict = {
            "scheduler": scheduler,
            "interval": "step",
            "monitor": "val/loss",
            "name": "trainer/lr",
        }
        return [optimizer], [scheduler_dict]

    @torch.no_grad()
    def generate(self, batch_size:int, num_sampling_steps: int):
        # use ema for generation 
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)
            self.ema.store(self.denoiser_module.parameters())
            self.ema.copy_to(self.denoiser_module.parameters())
        self.denoiser_module.eval()
        self.vae.eval()

        sampler = self.get_sampler()

        res = self.config.model.backbone.resolution 
        out_channels = self.config.model.backbone.out_channels
        noise = torch.randn(batch_size, out_channels, res, res, res, device=self.device, dtype=self.dtype)

        res = sampler.sample(
                self.denoiser_module,
                noise=noise,
                steps=num_sampling_steps, verbose=True,
        )
        return res
    
     ### <<EMA CODE >> 
    def on_load_checkpoint(self, checkpoint):
        if self.ema:
            self.ema.load_state_dict(checkpoint['ema'])
    
    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            checkpoint['ema'] = self.ema.state_dict()
    
    def on_train_start(self):
        if self.ema:    
            self.ema.move_shadow_params_to_device(self.device)

    def on_train_epoch_start(self):
        self.denoiser_module.train()
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(self.denoiser_module.parameters())
    
    def on_validation_epoch_start(self):
        if self.ema:
            self.ema.store(self.denoiser_module.parameters())
            self.ema.copy_to(self.denoiser_module.parameters())
        self.denoiser_module.eval()
        self.vae.eval()
    
    def on_validation_epoch_end(self):
        if self.ema:
            self.ema.restore(self.denoiser_module.parameters())



    
    

    

