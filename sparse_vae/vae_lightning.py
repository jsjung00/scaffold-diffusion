'''
Adapted from https://github.com/microsoft/TRELLIS/blob/main/trellis/trainers/vae/sparse_structure_vae.py
'''


import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision.ops import sigmoid_focal_loss 
import lightning as L
from typing import Dict, Any, Tuple 
from easydict import EasyDict 
import itertools 
import hydra
import spconv.pytorch as spconv 
from spconv.pytorch import SparseModule, SparseSequential 
from models.mdlm_ema import ExponentialMovingAverage

#from sparse_vae.sparse_vae import SparseVAE, SparseEncoder
from sparse_vae.minkowski_vae import SparseEncoder, SparseDecoder  

from sparse_vae.vae import SparseStructureEncoder, SparseStructureDecoder

class DenseToSparseConverter:
    def __init__(self):
        pass 

    def __call__(self, dense_voxels):
        '''
        dense_voxels: (torch.Tensor) of shape (B, X,Y,Z) or (B,1,X,Y,Z) with values 1 as occupied and 0 as not 
        '''
        if dense_voxels.dim() == 4:
            dense_voxels = dense_voxels.unsqueeze(1)
        
        batch_size, channels, H, W, D= dense_voxels.shape 
        device = dense_voxels.device 

        occupied_mask = dense_voxels.squeeze(1)

        indices = torch.nonzero(occupied_mask, as_tuple=False)

        features = torch.ones((indices.shape[0], channels), dtype=dense_voxels.dtype,\
            device=device)

        sparse_tensor = spconv.SparseConvTensor(
            features=features, 
            indices=indices.int(),
            spatial_shape=[D,H,W],
            batch_size=batch_size
        )
        return sparse_tensor
            

class SparseStructureVAE(L.LightningModule):
    def __init__(
        self, 
        config,
        sparse_layers=False  
    ):
        super().__init__()
        self.save_hyperparameters()
        self.sparse_layers = sparse_layers
        self.to_sparse = DenseToSparseConverter()
        '''
        
        self.sparse_vae = SparseVAE(in_channels=config.model.encoder.in_channels,\
                                                latent_channels=config.model.encoder.latent_channels,\
                                                num_res_blocks=config.model.encoder.num_res_blocks,\
                                                num_res_blocks_middle=config.model.encoder.num_res_blocks_middle,\
                                                channels=config.model.encoder.channels,\
                                                spatial_shape=[config.data.voxel_side_len]*3)
        self.encoder = self.sparse_vae.encoder 
        self.decoder = self.sparse_vae.decoder 
        '''
        '''
        self.sparse_encoder = SparseEncoder(in_channels=config.model.encoder.in_channels,\
                                                latent_channels=config.model.encoder.latent_channels,\
                                                num_res_blocks=config.model.encoder.num_res_blocks,\
                                                num_res_blocks_middle=config.model.encoder.num_res_blocks_middle,\
                                                channels=config.model.encoder.channels)
        '''
        self.sparse_encoder = SparseEncoder(in_channels=config.model.encoder.in_channels,\
                                                latent_channels=config.model.encoder.latent_channels,\
                                                num_res_blocks=config.model.encoder.num_res_blocks,\
                                                num_res_blocks_middle=config.model.encoder.num_res_blocks_middle,\
                                                channels=config.model.encoder.channels)  
    
        self.sparse_decoder = SparseDecoder(latent_channels=config.model.decoder.latent_channels,\
                                                num_res_blocks=config.model.decoder.num_res_blocks,\
                                                num_res_blocks_middle=config.model.decoder.num_res_blocks_middle,\
                                                channels=config.model.decoder.channels)

        self.encoder = SparseStructureEncoder(in_channels=config.model.encoder.in_channels,\
                                                latent_channels=config.model.encoder.latent_channels,\
                                                num_res_blocks=config.model.encoder.num_res_blocks,\
                                                num_res_blocks_middle=config.model.encoder.num_res_blocks_middle,\
                                                channels=config.model.encoder.channels)  
        self.decoder = SparseStructureDecoder(out_channels=config.model.decoder.out_channels,\
                                                latent_channels=config.model.decoder.latent_channels,\
                                                num_res_blocks=config.model.decoder.num_res_blocks,\
                                                num_res_blocks_middle=config.model.decoder.num_res_blocks_middle,\
                                                channels=config.model.decoder.channels)
       
        self.config = config 
        self.lambda_kl = self.config.model.lambda_kl
        if 'weight_air' in self.config.model:
            self.weight_air = self.config.model.weight_air   
        else:
            self.weight_air = True 
        if 'air_weight' in self.config.model:
            self.air_weight = self.config.model.air_weight 
        else:
            self.air_weight = self.config.air_weight

        if self.config.training.ema > 0:
            self.ema = ExponentialMovingAverage(
                itertools.chain(self.encoder.parameters(),
                                self.decoder.parameters()),
                decay=self.config.training.ema
            )
        else:
            self.ema = None 

    def set_dtype(self, dtype):
        self.dtype = dtype
        self.to(dtype)
    
    def setup(self, stage=None):
        pass 
        #self.encoder.set_dtype(self.dtype)
        #self.decoder.set_dtype(self.dtype)

    def compute_loss(self, batch, stage):
        '''
        batch: [N x 1 x H x W x D] tensor of binary sparse structure.

        stage: train / val / test 
        '''
        loss_dict = EasyDict(loss = 0.0)

        batch = batch.to(self.dtype)
        if self.sparse_layers:
            breakpoint()
            s_batch = self.to_sparse(batch) 
            z, mean, logvar, ref_tensor = self.sparse_encoder(s_batch, sample_posterior=(stage=='train'), return_raw=True)
            out_cls, targets = self.sparse_decoder(z, ref_tensor, s_batch.indices)
            
            # TODO: refactor 
            crit = nn.BCEWithLogitsLoss()
            num_layers, BCE = len(out_cls), 0
            for out_cl, target in zip(out_cls, targets):
                curr_loss = crit(out_cl.features.squeeze(), target.to(out_cl.features.dtype))
                BCE += curr_loss / num_layers 
            
            KLD = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
            loss_dict['loss'] = BCE + self.lambda_kl * KLD 

            # get logits of final structure
            logits = torch.zeros_like(batch.squeeze())
            final_indices = out_cls[-1].indices 
            final_features = out_cls[-1].features.squeeze() 
            
            batch_idx, z_idx, y_idx, x_idx = final_indices[:, 0], final_indices[:,1], final_indices[:,2], final_indices[:,3]
            logits[batch_idx, z_idx, y_idx, x_idx] = final_features
        else:
            z, mean, logvar = self.encoder(batch, sample_posterior=(stage=='train'), return_raw=True)
            logits = self.decoder(z)

        # calculate accuracy
        air_mask = (batch == 0).to(self.dtype)
        non_air_mask = (batch > 0).to(self.dtype)

        same_value = ((F.sigmoid(logits) >= 0.5) == batch) 
        air_acc = (air_mask * same_value).sum() / torch.clamp(air_mask.sum(), min=1)
        nonair_acc = (non_air_mask * same_value).sum() / torch.clamp(non_air_mask.sum(), min=1)
        loss_dict['air_acc'] = air_acc
        loss_dict['nonair_acc'] = nonair_acc 

        if self.sparse_layers:
            return loss_dict
        
        
        #x_recon, mean, logvar = self.sparse_vae(sparse_x)
        #logits = x_recon.dense() 
        
        #z, mean, logvar = self.encoder(batch, sample_posterior=True, return_raw=True)
        #z, mean, logvar = self.encoder(sparse_x, sample_posterior=True, return_raw=True)
        #logits = self.decoder(z)
        if self.config.model.loss_type == 'bce':
            recon_loss = F.binary_cross_entropy_with_logits(logits, batch.float(), reduction='none')             
            loss_dict['recon_loss'] = recon_loss.mean()

            non_air_nlls = non_air_mask * recon_loss 
            air_nlls = air_mask * recon_loss 

            non_air_nll = non_air_nlls.sum() / torch.clamp(non_air_mask.sum(), min=1)
            air_nll = air_nlls.sum() / torch.clamp(air_mask.sum(), min=1)

        
            if self.weight_air:
                weighted_nll = self.air_weight * air_nll + (1- self.air_weight) * non_air_nll 
            else:
                weighted_nll = loss_dict['recon_loss'] 

            loss_dict['air_recon_loss']= air_nll
            loss_dict['nonair_recon_loss'] = non_air_nll 
            loss_dict['weighted_recon_loss'] = weighted_nll
        
        elif self.config.model.loss_type == 'l1':
            recon_loss = F.l1_loss(F.sigmoid(logits), batch, reduction='none')
            loss_dict['recon_loss'] = recon_loss.mean()
        elif self.config.model.loss_type == 'dice':
            logits = F.sigmoid(logits)
            recon_loss = 1 - (2 * (logits * batch).sum() + 1 ) / (logits.sum()+ batch.sum()+ 1)
            loss_dict['recon_loss'] = recon_loss 
            weighted_nll = loss_dict['recon_loss']
        elif self.config.model.loss_type == 'dice_focal':
            prob = F.sigmoid(logits)
            dice_loss = 1 - (2 * (prob * batch).sum() + 1 ) / (prob.sum()+ batch.sum()+ 1)
            focal_loss = sigmoid_focal_loss(logits, batch, alpha=0.25, gamma=2, reduction='mean')
            loss_dict['recon_loss'] = 0.5*dice_loss + 0.5*focal_loss 
            weighted_nll = loss_dict['recon_loss']
        else:
            raise ValueError(f"Invalid loss type")

        
        loss_dict['kl_loss'] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
        loss_dict['loss'] = weighted_nll + self.lambda_kl  * loss_dict['kl_loss'] 

        return loss_dict 

    def training_step(self, batch, batch_idx):
        '''
        batch: sparse binary occupancy map. (B, H, W, D)
        '''
        batch = torch.unsqueeze(batch, dim=1) #(B,1,H,W,D)
        batch = batch.to(self.dtype) 

        loss_dict = self.compute_loss(batch, stage='train')
        loss = loss_dict['loss']

        # log losses
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_recon_loss', loss_dict['recon_loss'], on_step=True, on_epoch=True)
        if 'weighted_recon_loss' in loss_dict:
            self.log('train_weight_recon_loss', loss_dict['weighted_recon_loss'], on_step=True, on_epoch=True)
        if 'air_recon_loss' in loss_dict:
            self.log('train_air_recon_loss', loss_dict['air_recon_loss'], on_step=True, on_epoch=True)
        if 'nonair_recon_loss' in loss_dict:
            self.log('train_nonair_recon_loss', loss_dict['nonair_recon_loss'], on_step=True, on_epoch=True)
        self.log('train_air_acc', loss_dict['air_acc'], on_step=True, on_epoch=True)
        self.log('train_nonair_acc', loss_dict['nonair_acc'], on_step=True, on_epoch=True)
        
        self.log('train_kl_loss', loss_dict['kl_loss'], on_step=True, on_epoch=True)
        return loss  
    
    def validation_step(self, batch, batch_idx):
        '''
        batch: sparse binary occupancy map. (B, H, W, D)
        '''
        batch = torch.unsqueeze(batch, dim=1) #(B,1,H,W,D)
        batch = batch.to(self.dtype) 
        
        loss_dict = self.compute_loss(batch, stage='val')
        loss = loss_dict['loss']

        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        self.log('val_recon_loss', loss_dict['recon_loss'], on_step=False, on_epoch=True)
        if 'air_recon_loss' in loss_dict:
            self.log('val_air_recon_loss', loss_dict['air_recon_loss'], on_step=False, on_epoch=True)
        if 'nonair_recon_loss' in loss_dict:
            self.log('val_nonair_recon_loss', loss_dict['nonair_recon_loss'], on_step=False, on_epoch=True)
        if 'weighted_recon_loss' in loss_dict:
            self.log('val_weight_recon_loss', loss_dict['weighted_recon_loss'], on_step=False, on_epoch=True)
        self.log('val_kl_loss', loss_dict['kl_loss'], on_step=False, on_epoch=True)
        self.log('val_air_acc', loss_dict['air_acc'], on_step=True, on_epoch=True)
        self.log('val_nonair_acc', loss_dict['nonair_acc'], on_step=True, on_epoch=True)
        
        return loss 

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            itertools.chain(self.encoder.parameters(), self.decoder.parameters()),
            lr=self.config.optim.lr,
            #betas=(self.config.optim.beta1, self.config.optim.beta2),
            #eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay,
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

    def encode(self, batch, sample=False, return_raw=False):
        '''
        batch: (B,1,H,W,D)
        '''
        if len(batch.shape) < 5:
            batch = torch.unsqueeze(batch, dim=1)
        
        batch = batch.to(self.dtype)
        return self.encoder(batch, sample_posterior=sample, return_raw=return_raw)
    

    def decode(self, z):
        logits = self.decoder(z)
        probs = F.sigmoid(logits)
        recon = (probs >= 0.5)
        return recon 
    
    def reconstruct(self, batch, sample=False, perturb=False):
        # use ema 
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)
            self.ema.store(itertools.chain(
                self.encoder.parameters(),
                self.decoder.parameters()
            ))
            self.ema.copy_to(itertools.chain(
                self.encoder.parameters(),
                self.decoder.parameters()
            ))
        self.encoder.eval()
        self.decoder.eval()

        batch = torch.unsqueeze(batch, dim=1) #(B,1,H,W,D)
        batch = batch.to(self.dtype) 
        if perturb:
            z, z_perturb, mu, logvar = self.encoder(batch, sample_posterior=sample, return_raw=True, perturb=True)
            logits, logits_perturb = self.decoder(z), self.decoder(z_perturb)
            probs, probs_perturb = F.sigmoid(logits), F.sigmoid(logits_perturb)
            same_value, same_value_perturb = ((probs >= 0.5) == batch), ((probs_perturb >= 0.5) == batch)  
            recon, recon_perturb = (probs >= 0.5), (probs_perturb >= 0.5)
            return recon.squeeze(), recon_perturb.squeeze(), same_value, same_value_perturb
        else:
            z, mu, logvar = self.encoder(batch, sample_posterior=sample, return_raw=True, perturb=False)

            logits = self.decoder(z)
            probs = F.sigmoid(logits)
            same_value = ((probs >= 0.5) == batch) 
            recon = (probs >= 0.5)

            return recon.squeeze(), same_value 

    
    ### <<EMA CODE >> 
    def on_load_checkpoint(self, checkpoint):
        if self.ema:
            # hack because I trained models before adding ema logic 
            if 'ema' not in checkpoint:
                self.ema = None 
                return 
            
            self.ema.load_state_dict(checkpoint['ema'])
    
    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            checkpoint['ema'] = self.ema.state_dict()
    
    def on_train_start(self):
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)
    
    def on_train_epoch_start(self):
        self.encoder.train()
        self.decoder.train()
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(itertools.chain(
                self.encoder.parameters(),
                self.decoder.parameters()
            ))
    def on_validation_epoch_start(self):
        if self.ema:
            self.ema.store(itertools.chain(
                self.encoder.parameters(),
                self.decoder.parameters()
            ))
            self.ema.copy_to(itertools.chain(
                self.encoder.parameters(),
                self.decoder.parameters()
            ))
        self.encoder.eval()
        self.decoder.eval()
    
    def on_validation_epoch_end(self):
        if self.ema:
            self.ema.restore(
                itertools.chain(self.encoder.parameters(),
                                self.decoder.parameters()
            ))

    @torch.no_grad()
    def generate(self, batch):
        # Load EMA
        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)
            self.ema.store(itertools.chain(
                self.encoder.parameters(),
                self.decoder.parameters()
            ))
            self.ema.copy_to(itertools.chain(
                self.encoder.parameters(),
                self.decoder.parameters()
            ))
        self.encoder.eval()
        self.decoder.eval()


        batch = torch.unsqueeze(batch, dim=1) #(B,1,H,W,D)
        batch = batch.to(self.dtype) 
        
       
        example_latent = self.encoder(batch, sample_posterior=False, return_raw=False)
        prior_sample = torch.randn(example_latent.shape).to(batch.device)
        logits = self.decoder(prior_sample)

        probs = F.sigmoid(logits)
        recon = (probs >= 0.5)

        return recon.squeeze() 




    


