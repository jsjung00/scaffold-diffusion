import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch

import dataloader
from voxelcnn.datasets import MinecraftTokenizer
from voxelcnn.data_utils import voxel_to_nbt, voxel_to_plot
from sparse_vae.vae_lightning import SparseStructureVAE
import numpy as np 

from functools import reduce
from operator import mul


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
    CHECKPOINT_PATH = '/home/jsjung00/Desktop/Code/voxeldiffusion/outputs/minecraft/2025.06.20/182243/checkpoints/sparse_vae_best.ckpt'
    tokenizer = MinecraftTokenizer(config, config.data.air_not_air) # can later play around with changing mask token
    train_ds, valid_ds = dataloader.get_dataloaders(
        config, tokenizer)

    first_batch = next(iter(train_ds))

    model = SparseStructureVAE.load_from_checkpoint(CHECKPOINT_PATH)
    model.eval()
    with torch.no_grad():
        train_acc, val_acc = [], []
        NUM_EVAL_BATCHES = 10
        i = 0
        for batch in train_ds:
            if i >= 10:
                break
            batch_hat, same = model.reconstruct(batch.cuda(), sample=False)
            total = reduce(mul, batch_hat.shape)
            acc = same.detach().cpu().sum() / total 
            train_acc.append(acc)
            i += 1 

        i = 0 
        for batch in valid_ds:
            if i >= 10:
                break 
            
            batch_hat, same = model.reconstruct(batch.cuda(), sample=False)
            #alignment = (batch_hat.detach().cpu() == batch).sum()
            total = reduce(mul, batch_hat.shape)
            #acc = same.detach().cpu().sum() / total
            val_acc.append(acc) 
            i += 1 
   
    print(f"average train acc: {np.mean(train_acc)}")
    print(f'avg val acc: {np.mean(val_acc)}')
    


if __name__ == "__main__":
    ## experiment with equally weighted
    main()