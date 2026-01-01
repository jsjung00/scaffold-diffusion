'''
Hard-coded generator that generates voxel maps 
'''
import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from torch import Tensor
import os 
import dataloader 
from voxelcnn.datasets import MinecraftTokenizer

HOME_DIR = os.path.dirname(os.path.abspath(__file__))


class RealOccupancyGenerator:
    def __init__(self, config, stored_maps_file=None):
        self.config = config 
        tokenizer = MinecraftTokenizer(config, self.config.data.air_not_air)

        train_ds, valid_ds = dataloader.get_dataloaders(
            config, tokenizer)
        self.train_ds = train_ds 
        self.valid_ds = valid_ds 

        if stored_maps_file is None:
            self.create_maps_file(os.path.join(HOME_DIR, '32real_occupancy_maps.pth'))
        
        self.maps = torch.load(os.path.join(HOME_DIR, '32real_occupancy_maps.pth'))  
         

    def create_maps_file(self, file_path):
        binary_occupancy_maps = []
        for batch in self.train_ds:
            occupancy_map = (batch > 0).long()
            binary_occupancy_maps.append(occupancy_map)
        
        for batch in self.valid_ds:
            occupancy_map = (batch > 0).long() 
            binary_occupancy_maps.append(occupancy_map)
        
        occupancy_tensor = torch.concat(binary_occupancy_maps, dim=0) #(N, X, Y, Z)
        print(f"Total number of houses: {occupancy_tensor.size(0)}")
        torch.save(occupancy_tensor, file_path)
        print(f"Saved tensor to {file_path}")

    def get_random_batch(self, batch_size):
        rand_idxs = torch.randint(low=0, high=len(self.maps), size=(batch_size,))
        return self.maps[rand_idxs]

@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
    breakpoint()
    OccGen = RealOccupancyGenerator(config, stored_maps_file=None)
    batch = OccGen.get_random_batch(batch_size=16)
    pass 



if __name__ == "__main__":
    main()
