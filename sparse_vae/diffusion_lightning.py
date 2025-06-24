#Code from https://github.com/Michedev/DDPMs-Pytorch/blob/master/model/ddpm.py

from math import sqrt 
from typing import Callable, Iterator, Tuple, Optional, Type, Union, List, ClassVar

import lightning as L 
import torch 
import torchvision.utils
from torch import nn 
from diffusion_backbone import SparseStructureDiffusionModel
from vae_lightning import SparseStructureVAE

from diffusion_distributions import sigma_x_t, mu_x_t, mu_hat_xt_x0, sigma_hat_xt_x0, x0_to_xt 
#from diffusion_scheduler.abs_var_scheduler import Scheduler 
from diffusion_scheduler import LinearScheduler
from torch.nn.parameter import Parameter 
import hydra 

class GaussianDDPM(L.LightningModule):
    '''
    DDPM is (vlb=False) and improved DDPM paper 
    '''
    def __init__(self, config):
        """
        :param denoiser_module: The nn which computes the denoise step i.e. q(x_{t-1} | x_t, t)
        :param T: the amount of noising steps
        :param variance_scheduler: the variance scheduler cited in DDPM paper. See folder variance_scheduler for practical implementation
        :param lambda_variational: the coefficient in from of variational loss
        :param width: image width
        :param height: image height
        :param input_channels: image input channels
        :param logging_freq: frequency of logging loss function during training
        :param vlb: true to include the variational lower bound into the loss function
        :param init_step_vlb: the step at which the variational lower bound is included into the loss function
        """
        super().__init__()
        self.save_hyperparameters()
        self.config = config 
        self.input_channels = config.model.input_channels
        #TODO: define denoiser module 
        self.T = self.config.model.noise_steps 
        backbone_config = self.config.model.backbone 
        self.denoiser_module = SparseStructureDiffusionModel(backbone_config)

        self.scheduler_config = self.config.model.scheduler


        self.var_scheduler = LinearScheduler(T = self.scheduler_config.T, beta_start = self.scheduler_config.beta_start,\
            beta_end = self.scheduler_config.beta_end)
    
            
        self.lambda_variational = self.config.model.lambda_variational
        self.register_buffer('alphas_hat', self.var_scheduler.get_alpha_hat())
        self.register_buffer('alphas', self.var_scheduler.get_alphas())
        self.register_buffer('betas', self.var_scheduler.get_betas())
        self.register_buffer('betas_hat', self.var_scheduler.get_betas_hat())
        '''
        self.alphas_hat = self.var_scheduler.get_alpha_hat().to(self.device)
        self.alphas = self.var_scheduler.get_alphas().to(self.device)
        self.betas = self.var_scheduler.get_betas().to(self.device)
        self.betas_hat = self.var_scheduler.get_betas_hat().to(self.device)
        '''
        self.mse = nn.MSELoss()
        self.vlb = self.config.model.vlb 
        assert self.vlb == False 

        self.init_step_vlb = self.config.model.init_step_vlb
        self.iteration = 0 
        self.init_step_vlb = max(1, self.init_step_vlb)

        # VAE encoder
        vae_path = self.config.latent.vae_ckpt
        self.vae_model = SparseStructureVAE.load_from_checkpoint(vae_path)
        self.vae_model.eval()
        for p in self.vae_model.parameters():
            p.requires_grad = False

    def setup(self, stage, dtype=None):
        if dtype is None:
            self.denoiser_module.set_dtype(self.dtype)
            self.vae_model.to(self.dtype)
            self.vae_model.encoder.set_dtype(self.dtype)
            self.vae_model.decoder.set_dtype(self.dtype)
        else:
            self.denoiser_module.set_dtype(dtype)
            self.vae_model.to(dtype)
            self.vae_model.encoder.set_dtype(dtype)
            self.vae_model.decoder.set_dtype(dtype)



    def forward(self, x: torch.FloatTensor, t: torch.Tensor):
        """
        Forward pass of the DDPM model.

        Args:
            x: Input image tensor.
            t: Time step tensor.

        Returns:
            Tuple of predicted noise tensor and predicted variance tensor.
        """
        return self.denoiser_module(x,t)
    
    def training_step(self, batch, batch_idx):  
        """
        Training step of the DDPM model. Encode on the fly.

        Args:
            batch: original voxel map (B, D, D, D) 
            batch_idx: Batch index.

        Returns:
            Dictionary containing the loss.
        """
        batch = torch.unsqueeze(batch, dim=1) #(B,1,H,W,D)
        batch = batch.to(self.device)
        with torch.no_grad():
            X = self.vae_model.encode(batch) #get dense latent (B,C,D,D,D)
        
        t = torch.randint(0, self.T - 1, (X.shape[0],), device=X.device)

        # TODO: could normalize latent....
        alpha_hat = self.alphas_hat[t].reshape(-1, 1, 1, 1, 1) 
        eps = torch.randn_like(X)

        x_t = x0_to_xt(X, alpha_hat, eps)
        
        # TODO: change backbone to allow for variational loss. right now v is None 
        pred_eps, v = self(x_t, t)

        loss = 0.0
        noise_loss = self.mse(eps, pred_eps)
        loss = loss + noise_loss 

        use_vlb = (self.iteration >= self.init_step_vlb) and self.vlb 
        if use_vlb:
            loss_vlb = self.lambda_variational * self.variational_loss(x_t, X, pred_eps, v, t).mean(dim=0).sum()
            loss = loss + loss_vlb 
        
        self.iteration += 1

        log_dict = {
            'train/loss': loss, 
            'train/noise_loss': noise_loss 
        } 
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True)
        return loss 
    
    def validation_step(self, batch, batch_idx):
        batch = torch.unsqueeze(batch, dim=1) #(B,1,H,W,D)
        batch = batch.to(self.device)
        with torch.no_grad():
            X = self.vae_model.encode(batch) #get dense latent (B,C,D,D,D)

        t = torch.randint(0, self.T - 1, (X.shape[0],), device=X.device)

        alpha_hat = self.alphas_hat[t].reshape(-1, 1, 1, 1, 1)

        eps = torch.randn_like(X)

        x_t = x0_to_xt(X, alpha_hat, eps)
        
        pred_eps, v = self(x_t, t)

        loss = eps_loss = self.mse(eps, pred_eps)

        if self.iteration >= self.init_step_vlb and self.vlb:
            loss_vlb = self.lambda_variational * self.variational_loss(x_t, X, pred_eps, v, t).mean(dim=0).sum()
            loss = loss + loss_vlb 
        
        log_dict = {
            'val/loss': loss, 
            'val/noise_loss': eps_loss 
        } 
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True)

        #return dict(loss=loss, noise_loss=eps_loss, vlb_loss=loss_vlb if self.vlb else None)
        return loss


    def variational_loss(self, x_t, x_0, model_noise, v, t):
        """
        Compute variational loss for time step t
        
        Parameters:
            - x_t (torch.Tensor): the image at step t obtained with closed form formula from x_0
            - x_0 (torch.Tensor): the input image
            - model_noise (torch.Tensor): the unet predicted noise
            - v (torch.Tensor): the unet predicted coefficients for the variance
            - t (torch.Tensor): the time step
        
        Returns:
            - vlb (torch.Tensor): the pixel-wise variational loss, with shape [batch_size, channels, width, height]
        """
        vlb = 0.0 
        t_eq_0 = (t == 0).reshape(-1, 1, 1, 1, 1)

        if torch.any(t_eq_0):
            p = torch.distributions.Normal(mu_x_t(x_t, t, model_noise, self.alphas_hat,\
                                            self.betas, self.alphas),\
                                            sigma_x_t(v,t, self.betas_hat, self.betas))
            
            vlb += -p.log_prob(x_0) * t_eq_0.to(self.device)
        
        t_eq_last = (t == (self.T - 1)).reshape(-1, 1, 1, 1, 1)

        # compute variational loss for t=T-1
        if torch.any(t_eq_last):
            p = torch.distributions.Normal(0,1)
            q = torch.distributions.Normal(sqrt(self.alphas_hat[t]) * x_0, (1-self.alphas_hat[t]))

            vlb += torch.distributions.kl_divergence(q,p) * t_eq_last.to(self.device)
        
        # compute variational loss for other time steps
        mu_hat = mu_hat_xt_x0(x_t, x_0, t, self.alphas_hat, self.alphas, self.betas)
        sigma_hat = sigma_hat_xt_x0(t, self.betas_hat)
        q = torch.distributions.Normal(mu_hat, sigma_hat)

        mu = mu_x_t(x_t, t, model_noise, self.alphas_hat, self.betas, self.alphas).detach()
        sigma = sigma_x_t(v, t, self.betas_hat, self.betas)
        p = torch.distributions.Normal(mu, sigma)
        vlb += torch.distributions.kl_divergence(q, p) * (~t_eq_last).to(self.device) * (~t_eq_0).to(self.device)

        return vlb 
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=0.0001, #self.config.optim.lr,
            weight_decay=0.0 #self.config.optim.weight_decay,
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

    #TODO: add diffferent sampling schedulers 
    def generate(self, T=None, batch_size=1, get_intermediate_steps=False):
        # move transformer backbone to bf16 because of fast_attn NOTE: assumes that we use fast_attn bf16
        #self.denoiser_module = self.denoiser_module.to(dtype=torch.bfloat16)
        #self.denoiser_module.set_dtype(dtype=self.dtype)
        self.denoiser_module.set_dtype(dtype=torch.bfloat16)

        T = T or self.T 
        if get_intermediate_steps:
            steps = []

        res = self.config.model.backbone.resolution 
        out_channels = self.config.model.backbone.out_channels

        X_noise = torch.randn(batch_size, out_channels, res, res, res, device=self.device, dtype=self.dtype)
        #X_noise_bf16 = X_noise.to(torch.bfloat16)
        beta_sqrt = torch.sqrt(self.betas)

        for t in range(T-1, -1, -1):
            X_noise = X_noise.to(torch.bfloat16)
            if get_intermediate_steps:
                steps.append(X_noise)
            t_tens = torch.full((batch_size,), t, dtype=torch.long, device=self.device)
            
            #eps, v = self.denoiser_module(X_noise, t_tens)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16): 
                eps, v = self.denoiser_module(X_noise, t_tens)

            if torch.any(torch.isnan(eps)):
                print(f"At time {t} we have hit nan. Why?")
                breakpoint()
                
            sigma = beta_sqrt[t_tens].reshape(-1,1,1,1,1)
            z = torch.randn_like(X_noise)
            if t == 0:
                z.fill_(0)
            
            alpha_t = self.alphas[t_tens].reshape(-1,1,1,1,1)
            alpha_hat_t = self.alphas_hat[t_tens].reshape(-1,1,1,1,1)
            X_noise = 1 / (torch.sqrt(alpha_t)) * \
                      (X_noise - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * eps) + sigma * z  # denoise step 
            if torch.any(torch.isnan(X_noise)):
                print(f"At time {t} X_noise we have hit nan. Why?")
                breakpoint()
        
        if get_intermediate_steps:
            steps.append(X_noise)
            return steps 

        return X_noise 











        





