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
from sparse_vae.diffusion_lightning import GaussianDDPM
import numpy as np 

from functools import reduce
from operator import mul
from tqdm import tqdm 

@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def test_latents(config):
    FULL_VAE_PATH = '/home/jsjung00/Desktop/Code/voxeldiffusion/outputs/minecraft/2025.06.22/165728/checkpoints/sparse_vae_best.ckpt'
    FULL_DIFFUSION_PATH = '/home/jsjung00/Desktop/Code/voxeldiffusion/outputs/minecraft/2025.06.23/115747/checkpoints/latent_diffusion_best.ckpt'
  
    VAE_PATH = '/home/jsjung00/Desktop/Code/voxeldiffusion/outputs/minecraft/2025.06.20/200536/checkpoints/sparse_vae_best.ckpt'
    #VAE_PATH = '/home/jsjung00/Desktop/Code/voxeldiffusion/outputs/minecraft/2025.06.20/200536/checkpoints/sparse_vae_best.ckpt' 
    DIFFUSION_PATH = '/home/jsjung00/Desktop/Code/voxeldiffusion/outputs/minecraft/2025.06.22/183428/checkpoints/latent_diffusion_best.ckpt'
    tokenizer = MinecraftTokenizer(config, config.data.air_not_air) # can later play around with changing mask token
    train_ds, valid_ds = dataloader.get_dataloaders(
        config, tokenizer)

    vae = SparseStructureVAE.load_from_checkpoint(FULL_VAE_PATH)
    vae.eval()
    #vae.to(torch.bfloat16) # force inference at 16
    vae.setup()


    diffusion = GaussianDDPM.load_from_checkpoint(FULL_DIFFUSION_PATH)
    #diffusion.to(torch.bfloat16)
    diffusion.eval()

    # get the distribution of latents (we expect close to gaussian)
    latent_stds = []
    latent_means = [] 
    NUM_BATCHES = 1000
    i = 0
    breakpoint()
    with torch.no_grad():
        for batch in valid_ds:
            batch = batch.cuda()
            if i >= NUM_BATCHES:
                break
            
            X = vae.encode(batch)
            latent_stds.append(torch.std(X).item())
            latent_means.append(torch.mean(X).item())
        
        print(f"Encoder latent mean: {np.mean(latent_means)}.\n Encoder latent std: {np.mean(latent_stds)}\n")

        # get distribution of generated latents
        gen_latent_stds, gen_latent_means = [], []
        NUM_BATCHES = 8
        for _ in tqdm(range(NUM_BATCHES)):
            gen_z = diffusion.generate(batch_size=4)
            print(f"Mean is {torch.mean(gen_z).item()}")
            print(f"Std is {torch.std(gen_z).item()}")
            gen_latent_means.append(torch.mean(gen_z).item())
            gen_latent_stds.append(torch.std(gen_z).item())
        
        print(f"Encoder latent mean: {np.mean(gen_latent_means)}\n . Encoder latent std: {np.mean(gen_latent_stds)}")




@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def test_diffusion(config):
    FULL_VAE_PATH = '/home/jsjung00/Desktop/Code/voxeldiffusion/outputs/minecraft/2025.06.22/165728/checkpoints/sparse_vae_best.ckpt'
    FULL_DIFFUSION_PATH = '/home/jsjung00/Desktop/Code/voxeldiffusion/outputs/minecraft/2025.06.23/115747/checkpoints/latent_diffusion_best.ckpt'   
    #FULL_VAE_PATH = '/home/jsjung00/Desktop/Code/voxeldiffusion/outputs/minecraft/2025.06.22/165728/checkpoints/sparse_vae_best.ckpt'
    VAE_PATH = '/home/jsjung00/Desktop/Code/voxeldiffusion/outputs/minecraft/2025.06.20/200536/checkpoints/sparse_vae_best.ckpt'
    #VAE_PATH = '/home/jsjung00/Desktop/Code/voxeldiffusion/outputs/minecraft/2025.06.20/200536/checkpoints/sparse_vae_best.ckpt' 
    #DIFFUSION_PATH = '/home/jsjung00/Desktop/Code/voxeldiffusion/outputs/minecraft/2025.06.22/183428/checkpoints/latent_diffusion_best.ckpt'
    #FULL_DIFFUSION_PATH = '/home/jsjung00/Desktop/Code/voxeldiffusion/outputs/minecraft/2025.06.23/115747/checkpoints/latent_diffusion_best.ckpt'
    tokenizer = MinecraftTokenizer(config, config.data.air_not_air) # can later play around with changing mask token
    train_ds, valid_ds = dataloader.get_dataloaders(
        config, tokenizer)

    vae = SparseStructureVAE.load_from_checkpoint(FULL_VAE_PATH)
    #vae.to(torch.bfloat16)
    vae.eval()
    vae.setup()

    diffusion = GaussianDDPM.load_from_checkpoint(FULL_DIFFUSION_PATH)
    #diffusion.to(torch.bfloat16)
    diffusion.eval()
    #diffusion.vae_model.setup()
    breakpoint()
    #diffusion.setup("predict", dtype=torch.bfloat16)
    
    with torch.no_grad():
        #with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        generated_z = diffusion.generate(batch_size=4)
        occupancy_maps = vae.decode(generated_z)

    voxel_to_plot(occupancy_maps[0].squeeze().long(), "occupancy_generated", base_dir='/home/jsjung00/Desktop/Code/voxeldiffusion/output_files')
    voxel_to_plot(occupancy_maps[1].squeeze().long(), "occupancy_generated1", base_dir='/home/jsjung00/Desktop/Code/voxeldiffusion/output_files')
    voxel_to_plot(occupancy_maps[2].squeeze().long(), "occupancy_generated2", base_dir='/home/jsjung00/Desktop/Code/voxeldiffusion/output_files')
    voxel_to_plot(occupancy_maps[3].squeeze().long(), "occupancy_generated3", base_dir='/home/jsjung00/Desktop/Code/voxeldiffusion/output_files')




@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
    FULL32_VAE_PATH = '/home/jsjung00/Desktop/Code/voxeldiffusion/outputs/minecraft/2025.06.22/165728/checkpoints/sparse_vae_best.ckpt'
    #VAE_PATH = '/home/jsjung00/Desktop/Code/voxeldiffusion/outputs/minecraft/2025.06.20/200536/checkpoints/sparse_vae_best.ckpt'
    #CHECKPOINT_PATH = '/home/jsjung00/Desktop/Code/voxeldiffusion/outputs/minecraft/2025.06.20/182243/checkpoints/sparse_vae_best.ckpt'
    tokenizer = MinecraftTokenizer(config, config.data.air_not_air) # can later play around with changing mask token
    train_ds, valid_ds = dataloader.get_dataloaders(
        config, tokenizer)

    first_batch = next(iter(train_ds))

    model = SparseStructureVAE.load_from_checkpoint(FULL32_VAE_PATH)
    model.to(torch.bfloat16)
    model.setup("train")
    model.eval()
    breakpoint()

    with torch.no_grad():
        train_acc, val_acc = [], []
        NUM_EVAL_BATCHES = 10
        i = 0
        for batch in train_ds:
            if i >= 10:
                break
            batch_hat, same = model.reconstruct(batch.cuda().to(), sample=False)
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
    #main()
    #test_diffusion()
    test_latents()