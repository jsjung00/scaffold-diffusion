import torch 
from torch import nn 
from torch.nn import functional as F 
import numpy as np 
import math 
from .loss import lovasz_softmax 
from .model import C_Encoder, C_Decoder
from .vector_quantizer import VectorQuantizer 

import lightning as L 
import hydra 


class VQVAE(L.LightningModule):
    def __init__(self, config, multi_criterion):
        super().__init__()
        self.save_hyperparameters()
        self.config = config 
        self.config.model = config.model 

        init_size = self.config.model.init_size 
        embedding_dim = int(self.config.model.num_classes)

        self.VQ = VectorQuantizer(num_embeddings= int(self.config.model.num_classes) * int(self.config.model.vq_size), embedding_dim=embedding_dim )

        self.encoder = C_Encoder(config.model, nclasses=self.config.model.num_classes, init_size=init_size, l_size=config.model.l_size, attention=config.model.l_attention)
        self.quant_conv = nn.Conv3d(self.config.model.num_classes, self.config.model.num_classes, kernel_size=1, stride=1)

        self.decoder = C_Decoder(config.model, nclasses=self.config.model.num_classes, init_size=init_size, l_size=config.model.l_size, attention=config.model.l_attention)
        self.post_quant_conv = nn.Conv3d(self.config.model.num_classes, self.config.model.num_classes, kernel_size=1, stride=1)

        self.multi_criterion = multi_criterion
    
    def device(self):
        return self.encoder.device 

    def encode(self, x):
        latent = self.encoder(x)
        latent = self.quant_conv(latent)
        return latent 
    
    def vector_quantize(self, latent):
        quantized_latent, vq_loss, quantized_latent_ind, latents_shape = self.VQ(latent)
        return quantized_latent, vq_loss, quantized_latent_ind, latents_shape
    
    def codebook(self, quantized_latent_ind, latents_shape):
        quantized_latent = self.VQ.codebook_to_embedding(quantized_latent_ind.view(-1, 1), latents_shape)
        return quantized_latent

    def decode(self, quantized_latent):
        quantized_latent = self.post_quant_conv(quantized_latent)
        recons = self.decoder(quantized_latent)
        return recons 

    def forward(self, x):
        latent = self.encode(x)
        quantized_latent, vq_loss, _, _ = self.vector_quantize(latent)
        recons = self.decode(quantized_latent)

        recons_loss = self.multi_criterion(recons, x.long())
        loss = recons_loss + vq_loss 
        return loss 
    
    def sample(self, x):
        latent = self.encode(x)
        quantized_latent, _, _, _ = self.vector_quantize(latent)
        recons = self.decode(quantized_latent)
        recons = recons.argmax(1)
        return recons 
    
    def training_step(self, batch):
        loss = self(batch)
        self.log("train/loss", loss)
        return loss 

    def validation_step(self, batch):
        loss = self(batch)
        self.log("val/loss", loss)
        return loss 
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
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
