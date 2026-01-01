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
        #TODO: zero out the conv2 weights if we get bad performance
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

class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
    
    def forward(self, x):
        return spconv.SparseConvTensor(
            features=self.linear(x.features),
            indices=x.indices, 
            spatial_shape=x.spatial_shape, 
            batch_size=x.batch_size 
        )

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
            spconv.SubMConv3d(in_channels, channels[0], 3, padding=1)
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
            nn.BatchNorm1d(channels[-1]), #SparseNorm(channels[-1]), #,
            nn.ReLU(), #SparseRelu(), #
            spconv.SubMConv3d(channels[-1], latent_channels*2, 3, padding=1)
        ])

        self.encoder = SparseSequential(*encoder_layers)

        self.linear_mean = nn.Linear(latent_channels, latent_channels)
        self.linear_logvar = nn.Linear(latent_channels, latent_channels)
    
    def global_pool(self, sparse_x):
        dense_x = sparse_x.dense() #(B,C,X,Y,Z)
        spatial_dims = list(range(2, dense_x.ndim))
        pooled = torch.mean(dense_x, dim=spatial_dims)
        return pooled 
    
    def forward(self, x, sample_posterior=False, return_raw: bool = False):
        sparse_h = self.encoder(x)

        h_pooled = self.global_pool(sparse_h)
        mean, logvar = h_pooled.chunk(2, dim=1) #dense torch (B,latent_channels), (B,latent_channels)

        mean = self.linear_mean(mean)
        logvar = self.linear_logvar(logvar)

        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean 

        if return_raw:
            return z, mean, logvar, sparse_h  

        return z, sparse_h 

class SparseDecoder(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        channels: List[int],
        num_res_blocks_middle: int=2, 
        num_res_blocks=2
    ):
        super().__init__()

        self.channels = channels 
        input_layers = []
        input_layers.append(
            spconv.SubMConv3d(latent_channels, channels[0], 3, padding=1)
        )
        for _ in range(num_res_blocks_middle):
            input_layers.append(SparseResBlock3d(channels[0], channels[0]))

        self.input_layer = SparseSequential(*input_layers)

        self.keep_layers = nn.ModuleList()
        self.blocks = nn.ModuleList()
        # middle blocks 
        for i, (ch_in, ch_out) in enumerate(zip(channels[:-1], channels[1:])):
            block_layers = []

            # upsample
            block_layers.append(
                spconv.SparseConvTranspose3d(ch_in, ch_out, kernel_size=2, stride=2)
            )
            
            for _ in range(num_res_blocks):
                block_layers.append(SparseResBlock3d(ch_out))
            
           
            block = SparseSequential(*block_layers)
            self.blocks.append(block) 

            # keep at this resolution
            self.keep_layers.append(spconv.SubMConv3d(ch_out, 1, kernel_size=1, bias=True))
        
        # final block layer 
        '''
        final_block_layers = nn.ModuleList() 
        for _ in range(num_res_blocks):
            final_block_layers.append(SparseResBlock3d(channels[-1]))
        
        final_block_layers.extend([
            nn.BatchNorm1d(channels[-1]),
            nn.ReLU(),
        ])
        self.final_block = SparseSequential(*final_block_layers)
        
        # final keep mask 
        self.final_keep = spconv.SubMConv3d(channels[-1], 1, kernel_size=1, bias=True)
        '''
        

    def forward(self, z_glob, sparse_ref, target_coords=None):
        '''
        z_glob: (dense torch.Tensor) Shape (B,C). Latent feature for each batch
        ref_sparse_tensor: (sparse Tensor) The last sparse tensor in the encoder used to define spatial shape of first input
            to decoder. 
        target_coords: (dense torch.Tensor) Shape (N,4) where N is number of active voxels

        Decoder takes as first input a sparse tensor whose shape matches shape of last tensor in encoder and 
            contains all tokens, where each token has feature corresponding to batch latent vector in z  
        '''
        batch_size = z_glob.shape[0]
        base_shape = sparse_ref.spatial_shape 

        indices_list = []
        features_list = []

        for batch_idx in range(batch_size):
            # One voxel at origin for this batch
            indices_list.append([batch_idx, 0, 0, 0])
            features_list.append(z_glob[batch_idx])

        # Stack them
        indices = torch.tensor(indices_list, dtype=torch.int32, device=z_glob.device)  # (N, 4)
        features = torch.stack(features_list, dim=0)  # (N, latent_dim)

        sparse_z = spconv.SparseConvTensor(
            features=features, 
            indices=indices,
            spatial_shape=base_shape,
            batch_size=batch_size
        )

        # initial conv
        x = self.input_layer(sparse_z)

        keep_outputs = []
        targets = []

        for i, (block, keep_conv) in enumerate(zip(self.blocks, self.keep_layers)):
            # upsample 
            x = block(x)

            keep_out = keep_conv(x)
            keep_outputs.append(keep_out)
            keep_mask = (keep_out.features > 0).squeeze()

            # add target locations to keep while training
            if target_coords is not None:
                target = self.get_target_mask(x, target_coords, i)
                targets.append(target)

                if self.training and i < len(self.blocks)-1:
                    # last layer does not require keep 
                    keep_mask = keep_mask | target 
            
            # prune 
            if keep_mask.sum() > 0:
                x = self.prune_sparse_tensor(x, keep_mask)
        
        #final_out = self.final_block(x)
        #keep_out = self.final_keep(final_out)
        #keep_outputs.append(keep_out)
        #target = self.get_target(final_out, target_coords)
        #targets.append(target)

        return keep_outputs, targets
    
    def prune_sparse_tensor(self, x, keep_mask):
        '''
        keep_mask: 1D boolean tensor
        '''
        assert keep_mask.ndim == 1 and len(keep_mask) == len(x.features)

        return spconv.SparseConvTensor(
            features=x.features[keep_mask],
            indices=x.indices[keep_mask],
            spatial_shape=x.spatial_shape, 
            batch_size=x.batch_size 
        )
    
    def get_target_mask(self, x, target_coords, block_idx, exact_matching=True, tolerance=1):
        '''
        x: spconv.SparseConvTensor
        target_coords: torch.Tensor. Shape (N,4)
        block_idx: int. Represents how many up sampling operations have happened
        '''
        
        target = torch.zeros(x.features.shape[0], dtype=torch.bool, device=x.features.device)

        pred_coords = x.indices
       
        assert target_coords.shape[1] == 4, "shape should be (N, 4)"
        # calculate stride
        stride = 2** (len(self.channels) - 1 - (block_idx + 1))

        strided_target_coords = target_coords.clone()
        strided_target_coords[:, 1:] = strided_target_coords[:, 1:] // stride 

        if exact_matching:
            target_set = set()
            for coord in strided_target_coords:
                target_set.add(tuple(coord.cpu().numpy()))
            
            for i, pred_coord in enumerate(pred_coords):
                if tuple(pred_coord.cpu().numpy()) in target_set:
                    target[i] = True 
        else:
            # for each predicted coord, check if any target is within tolerance 
            for i, pred_coord in enumerate(pred_coords):
                batch_mask = strided_target_coords[:, 0] == pred_coord[0]
                batch_targets = strided_target_coords[batch_mask]

                if len(batch_targets) > 0:
                    # compute minimum (to any batch target point) of max coordinate difference
                    distances = torch.abs(batch_targets[:, 1:] - pred_coord[1:].unsqueeze(0))
                    min_distance = distances.max(dim=1)[0].min()

                    if min_distance <= tolerance:
                        target[i] = True 
        return target 
        