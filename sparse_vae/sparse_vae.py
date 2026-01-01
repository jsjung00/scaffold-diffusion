'''Adapted from Trellis'''
from typing import * 
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import sys 
from pathlib import Path 
sys.path.append(str(Path(__file__).parent))

from modules.norm import GroupNorm32, ChannelLayerNorm32
from modules.spatial import pixel_shuffle_3d 
from modules.utils import zero_module, convert_module_to_f16, convert_module_to_f32

import spconv.pytorch as spconv
from spconv.pytorch import SparseModule, SparseSequential
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import math 

class SparseRelu(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        m = nn.ReLU()
        relu_features = m(x.features)
        return x.replace_feature(relu_features)

class SparseNorm(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.norm = nn.BatchNorm1d(num_features)
    
    def forward(self, x):
        normalized_features = self.norm(x.features)
        return x.replace_feature(normalized_features)

class SparseResBlock3d(SparseModule):
    def __init__(self, channels, out_channels=None):
        super().__init__()
        self.channels = channels 
        self.out_channels = out_channels or channels 

        self.norm1 = SparseNorm(channels) #nn.BatchNorm1d(channels)
        self.norm2 = SparseNorm(self.out_channels) #nn.BatchNorm1d(self.out_channels)

        self.conv1 = spconv.SubMConv3d(
            channels, self.out_channels,
            kernel_size=3, padding=1, bias=False 
        )
        self.conv2 = spconv.SubMConv3d(
            self.out_channels, self.out_channels,
            kernel_size=3, padding=1, bias=False 
        )
        if channels != self.out_channels:
            self.skip_connection = spconv.SubMConv3d(
                channels, self.out_channels,
                kernel_size=1, bias=False 
            )
        else:
            self.skip_connection = None 
    
    def forward(self, x):
        identity = x 

        x = self.norm1(x)
        x = x.replace_feature(F.silu(x.features))
        x = self.conv1(x)

        x = self.norm2(x)
        x = x.replace_feature(F.silu(x.features))
        x = self.conv2(x)

        if self.skip_connection is not None:
            identity = self.skip_connection(identity)
        
        x = x.replace_feature(x.features + identity.features)
        return x 

class SparseEncoder(nn.Module):
    def __init__(
        self,
        in_channels:int,
        latent_channels: int,
        channels: List[int],
        num_res_blocks_middle: int=2, 
        num_res_blocks=2
    ):
        super().__init__()

        # encoder layers 
        encoder_layers = []

        encoder_layers.append(
            spconv.SparseConv3d(in_channels, channels[0], 3, padding=1)
        )

        # encoder blocks 
        for i, (ch_in, ch_out) in enumerate(zip(channels[:-1], channels[1:])):
            for _ in range(num_res_blocks):
                encoder_layers.append(SparseResBlock3d(ch_in))
            
            encoder_layers.append(
                spconv.SparseConv3d(ch_in, ch_out, kernel_size=2, stride=2)
            )
        for _ in range(num_res_blocks):
            encoder_layers.append(SparseResBlock3d(channels[-1]))

        # middle res blocks 
        for _ in range(num_res_blocks_middle):
            encoder_layers.append(SparseResBlock3d(channels[-1]))
        
        # output layer
        encoder_layers.extend([
            #SparseNorm(channels[-1]), #nn.BatchNorm1d(channels[-1]),
            SparseRelu(), #nn.ReLU()
            spconv.SubMConv3d(channels[-1], latent_channels*2, 3, padding=1)
        ])

        self.encoder = SparseSequential(*encoder_layers)
    
    def forward(self, x, sample_posterior=False, return_raw: bool = False):
        h = self.encoder(x)

        h_dense = h.dense()

        mean, logvar = h_dense.chunk(2, dim=1)

        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean 

        if return_raw:
            return z, mean, logvar 

        return z 


    
class SparseVAE(nn.Module):
    def __init__(
        self,
        in_channels:int,
        latent_channels: int,
        channels: List[int],
        num_res_blocks_middle: int=2, 
        spatial_shape=[64,64,64],
        num_res_blocks=2
    ):
        super().__init__()
        self.spatial_shape = spatial_shape

        # encoder layers 
        encoder_layers = []

        encoder_layers.append(
            spconv.SparseConv3d(in_channels, channels[0], 3, padding=1)
        )

        # encoder blocks 
        for i, (ch_in, ch_out) in enumerate(zip(channels[:-1], channels[1:])):
            for _ in range(num_res_blocks):
                encoder_layers.append(SparseResBlock3d(ch_in))
            
            encoder_layers.append(
                spconv.SparseConv3d(ch_in, ch_out, kernel_size=2, stride=2)
            )
        for _ in range(num_res_blocks):
            encoder_layers.append(SparseResBlock3d(channels[-1]))

        # middle res blocks 
        for _ in range(num_res_blocks_middle):
            encoder_layers.append(SparseResBlock3d(channels[-1]))
        
        # output layer
        encoder_layers.extend([
            nn.BatchNorm1d(channels[-1]),
            nn.ReLU(),
            #SparseNorm(channels[-1]), #nn.BatchNorm1d(channels[-1]),
            #SparseRelu(), #nn.ReLU()
            spconv.SubMConv3d(channels[-1], latent_channels*2, 3, padding=1)
        ])

        self.encoder = SparseSequential(*encoder_layers)

        # Decoder 
        decoder_layers = []
        # input layer 
        decoder_layers.append(
            spconv.SparseConv3d(latent_channels, channels[-1], 3, padding=1)
        )

        # middle res blocks
        for _ in range(num_res_blocks_middle):
            decoder_layers.append(SparseResBlock3d(channels[-1], channels[-1]))

        # res blocks 
        reversed_channels = list(reversed(channels))
        for i, (ch_in, ch_out) in enumerate(zip(reversed_channels[:-1], reversed_channels[1:])):
            for _ in range(num_res_blocks):
                decoder_layers.append(SparseResBlock3d(ch_in))
            
            # upsample
            decoder_layers.append(
                spconv.SparseConvTranspose3d(ch_in, ch_out, kernel_size=2, stride=2)
            )
        
        # final res blocks
        for _ in range(num_res_blocks):
            decoder_layers.append(SparseResBlock3d(channels[0]))
        
        # output layer 
        decoder_layers.extend([
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(),
            spconv.SubMConv3d(channels[0], in_channels, 3, padding=1)
        ])
        self.decoder = SparseSequential(*decoder_layers)

    def global_pool(self, sparse_x):
        dense_x = sparse_x.dense() #(B,C,X,Y,Z)
        spatial_dims = list(range(2, dense_x.ndim))
        pooled = torch.mean(dense_x, dim=spatial_dims)
        return pooled 

    def encode(self, x, sample_posterior=False, return_raw: bool = False):
        sparse_h = self.encoder(x)
        #h_pooled = spconv.SparseGlobalAvgPool()(sparse_h) 
        h_pooled = self.global_pool(sparse_h)
        mean, logvar = h_pooled.chunk(2, dim=1) #(B,latent_channels), (B,latent_channels)

        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean 

        if return_raw:
            return z, mean, logvar, sparse_h  

        return z, sparse_h  

    def decode(self, z, ref_sparse_tensor):
        '''
        z: (dense torch.Tensor) Shape (B,C). Latent feature for each batch
        ref_sparse_tensor: (sparse Tensor) The last sparse tensor in the encoder used to define spatial shape of first input
            to decoder. 

        Decoder takes as first input a sparse tensor whose shape matches shape of last tensor in encoder and 
            contains all tokens, where each token has feature corresponding to batch latent vector in z  
        '''
        batch_size = z.shape[0]
        base_shape = ref_sparse_tensor.spatial_shape 

        indices = []
        features = [] 
        
        for b in range(batch_size):
            for i in range(base_shape[0]):
                for j in range(base_shape[1]):
                    for k in range(base_shape[2]):
                        features.append(z[b])
                        indices.append([b, i, j, k])
        
        indices = torch.tensor(indices, dtype=torch.int32).cuda()
        features = torch.stack(features).cuda() # (num_active, latent_dim)

        sparse_z = spconv.SparseConvTensor(
            features=features, 
            indices=indices,
            spatial_shape=base_shape,
            batch_size=batch_size
        )

        return self.decoder(sparse_z)
    
    def forward(self, sparse_x, sample_posterior=False):
        z, mean, logvar, last_sparse_encode = self.encode(sparse_x, sample_posterior, return_raw=True)
        x_recon = self.decode(z, last_sparse_encode)
        return x_recon, mean, logvar 
 