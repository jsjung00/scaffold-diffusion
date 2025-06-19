import torch 
import torch.nn.functional as F 
import pytorch_lightning as pl 
from typing import Dict, Any, Tuple 
from easydict import EasyDict 


class SparseStructureVAE(pl.LightningModule):
    def __init__(
        self, 
        encoder,
        decoder, 
        config 
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['encoder', 'decoder'])
        self.encoder = encoder 
        self.decoder = decoder 
        self.config = config 
        self.lambda_kl = self.config.model.lambda_kl

    def compute_loss(self, batch):
        '''
        batch: [N x 1 x H x W x D] tensor of binary sparse structure.
        '''
        # TODO: may need to unpack batch, convert to float datatype
        z, mean, logvar = self.encoder(batch, sample_posterior=True, return_raw=True)
        logits = self.decoder(z)
        loss_dict = EasyDict(loss = 0.0)
        if self.config.model.loss_type == 'bce':
            loss_dict['bce'] = F.binary_cross_entropy_with_logits(logits, batch, reduction='mean') 
            loss_dict['loss'] = loss_dict['loss'] + loss_dict['bce']
        elif self.config.model.loss_type == 'l1':
            loss_dict['l1'] = F.l1_loss(F.sigmoid(logits), batch, reduction='mean')
            loss_dict['loss'] = loss_dict['loss'] + loss_dict['l1']
        elif self.config.model.loss_type == 'dice':
            logits = F.sigmoid(logits)
            loss_dict['dice'] = 1 - (2 * (logits * batch).sum() + 1 ) / (logits.sum()+ batch.sum()+ 1)
            loss_dict['loss'] = loss_dict['loss'] + loss_dict['dice']
        else:
            raise ValueError(f"Invalid loss type")
        
        loss_dict['kl'] = 0.5 * torch.mean(mean.pow(2) + logvar.exp() - logvar - 1)
        loss_dict['loss'] = loss_dict['loss'] + self.lambda_kl  * loss_dict['kl'] 

        return loss_dict 

    def training_step(self, batch, batch_idx):
        '''
        batch: sparse binary occupancy map. (B, H, W, D)
        '''
        batch = torch.unsqueeze(batch, dim=1) #(B,1,H,W,D)
        batch = batch.float() 

        loss_dict = self.compute_loss(batch)
        loss = loss_dict['loss']

        # log losses
        self.log('train_log', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_recon_loss', loss_dict['recon_loss'], on_step=True, on_epoch=True)
        self.log('train_kl_loss', loss_dict['kl_loss'], on_step=True, on_epoch=True)
        return loss  
    
    def validation_step(self, batch, batch_idx):
        '''
        batch: sparse binary occupancy map. (B, H, W, D)
        '''
        batch = torch.unsqueeze(batch, dim=1) #(B,1,H,W,D)
        batch = batch.float() 
        
        loss_dict = self.compute_loss(batch)
        loss = loss_dict['loss']

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_recon_loss', loss_dict['recon_loss'], on_step=False, on_epoch=True)
        self.log('val_kl_loss', loss_dict['kl_loss'], on_step=False, on_epoch=True)
        
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
    


