import torch 
import torch.nn.functional as F 
import lightning as L
from typing import Dict, Any, Tuple 
from easydict import EasyDict 
import itertools 
import hydra 

from sparse_vae.vae import SparseStructureEncoder, SparseStructureDecoder

class SparseStructureVAE(L.LightningModule):
    def __init__(
        self, 
        config 
    ):
        super().__init__()
        self.save_hyperparameters()
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

    def set_dtype(self, dtype):
        self.dtype = dtype
        self.to(dtype)
    
    def setup(self, stage=None):
        self.encoder.set_dtype(self.dtype)
        self.decoder.set_dtype(self.dtype)

    def compute_loss(self, batch):
        '''
        batch: [N x 1 x H x W x D] tensor of binary sparse structure.
        '''
        batch.to(self.dtype)
        z, mean, logvar = self.encoder(batch, sample_posterior=True, return_raw=True)
        logits = self.decoder(z)
        loss_dict = EasyDict(loss = 0.0)
        if self.config.model.loss_type == 'bce':
            recon_loss = F.binary_cross_entropy_with_logits(logits, batch, reduction='none')             
            loss_dict['recon_loss'] = recon_loss.mean()
        elif self.config.model.loss_type == 'l1':
            recon_loss = F.l1_loss(F.sigmoid(logits), batch, reduction='none')
            loss_dict['recon_loss'] = recon_loss.mean()
        elif self.config.model.loss_type == 'dice':
            #TODO: handle no reduction
            logits = F.sigmoid(logits)
            recon_loss = 1 - (2 * (logits * batch).sum() + 1 ) / (logits.sum()+ batch.sum()+ 1)
            loss_dict['recon_loss'] = recon_loss 
        else:
            raise ValueError(f"Invalid loss type")

        non_air_mask = (batch > 0).to(self.dtype)
        non_air_nlls = non_air_mask * recon_loss 
        
        air_mask = (batch == 0).to(self.dtype)
        air_nlls = air_mask * recon_loss 

        non_air_nll = non_air_nlls.sum() / torch.clamp(non_air_mask.sum(), min=1)
        air_nll = air_nlls.sum() / torch.clamp(air_mask.sum(), min=1)

        # calculate accuracy
        same_value = ((F.sigmoid(logits) >= 0.5) == batch) 
        air_acc = (air_mask * same_value).sum() / torch.clamp(air_mask.sum(), min=1)
        nonair_acc = (non_air_mask * same_value).sum() / torch.clamp(non_air_mask.sum(), min=1)

        weighted_nll = self.config.air_weight * air_nll + (1- self.config.air_weight) * non_air_nll 
        
        loss_dict['air_acc'] = air_acc
        loss_dict['nonair_acc'] = nonair_acc 
        loss_dict['air_recon_loss']= air_nll
        loss_dict['nonair_recon_loss'] = non_air_nll 
        loss_dict['weighted_recon_loss'] = weighted_nll
        loss_dict['kl_loss'] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
        loss_dict['loss'] = weighted_nll + self.lambda_kl  * loss_dict['kl_loss'] 

        return loss_dict 

    def training_step(self, batch, batch_idx):
        '''
        batch: sparse binary occupancy map. (B, H, W, D)
        '''
        batch = torch.unsqueeze(batch, dim=1) #(B,1,H,W,D)
        batch = batch.to(self.dtype) 

        loss_dict = self.compute_loss(batch)
        loss = loss_dict['loss']

        # log losses
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_recon_loss', loss_dict['recon_loss'], on_step=True, on_epoch=True)
        self.log('train_weight_recon_loss', loss_dict['weighted_recon_loss'], on_step=True, on_epoch=True)
        self.log('train_air_recon_loss', loss_dict['air_recon_loss'], on_step=True, on_epoch=True)
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
        
        loss_dict = self.compute_loss(batch)
        loss = loss_dict['loss']

        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_recon_loss', loss_dict['recon_loss'], on_step=False, on_epoch=True)
        self.log('val_air_recon_loss', loss_dict['air_recon_loss'], on_step=False, on_epoch=True)
        self.log('val_nonair_recon_loss', loss_dict['nonair_recon_loss'], on_step=False, on_epoch=True)
        self.log('val_weight_recon_loss', loss_dict['weighted_recon_loss'], on_step=False, on_epoch=True)
        self.log('val_kl_loss', loss_dict['kl_loss'], on_step=False, on_epoch=True)
        self.log('val_air_acc', loss_dict['air_acc'], on_step=False, on_epoch=True)
        self.log('val_nonair_acc', loss_dict['nonair_acc'], on_step=False, on_epoch=True)
        
        return loss 

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            itertools.chain(self.encoder.parameters(), self.decoder.parameters()),
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
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

    def encode(self, batch, sample=False):
        '''
        batch: (B,1,H,W,D)
        '''
        if len(batch.shape) < 5:
            batch = torch.unsqueeze(batch, dim=1)
        
        batch = batch.to(self.dtype)
        z = self.encoder(batch, sample_posterior=sample, return_raw=False)
        return z

    def decode(self, z):
        logits = self.decoder(z)
        probs = F.sigmoid(logits)
        recon = (probs >= 0.5)
        return recon 
    
    def reconstruct(self, batch, sample=False):
        batch = torch.unsqueeze(batch, dim=1) #(B,1,H,W,D)
        batch = batch.to(self.dtype) 
        z = self.encoder(batch, sample_posterior=sample, return_raw=False)
        
        logits = self.decoder(z)
        probs = F.sigmoid(logits)
        same_value = ((probs >= 0.5) == batch) 
        recon = (probs >= 0.5)

        return recon, same_value 


    


