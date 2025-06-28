from real_generator import RealOccupancyGenerator
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
from voxelcnn.data_utils import voxel_to_nbt, voxel_to_plot


HOME_DIR = os.path.dirname(os.path.abspath(__file__))


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def see_occupancy_maps(config):
    output_dir = '/home/jsjung00/Desktop/Code/voxeldiffusion/output_files'
    OccGen = RealOccupancyGenerator(config, stored_maps_file=None)
    batch = OccGen.get_random_batch(batch_size=16)
    for batch_idx in range(0, 6):
        voxel_to_nbt(batch[batch_idx], f"batch{batch_idx}_generated_sample", base_dir=output_dir, gzip=True)
        voxel_to_plot(batch[batch_idx], f"batch{batch_idx}_generated_sample", base_dir=output_dir)


if __name__ == "__main__":
    see_occupancy_maps()  