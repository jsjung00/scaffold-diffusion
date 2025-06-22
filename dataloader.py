from voxelcnn.datasets import Craft3DDataset
from torch.utils.data import DataLoader, Dataset, RandomSampler 
import torch 
import json 
from sparse_vae.vae_lightning import SparseStructureVAE


def latent_collate_fn(batch, encoder, device='cuda'):
    '''
    batch: list of raw voxel tensors [H, W, D]
    '''
    voxels = torch.stack(batch, dim=0).to(device)
    voxels = voxels.unsqueeze(dim=1) # (B, 1, H, W, D)

    encoder.eval()
    with torch.no_grad():
        latents = encoder.encode(voxels) #(B,C, _, _, _)
    
    return latents 

class CycleDataset(Dataset):
    def __init__(self, original_dataset, cycle_length):
        self.original_dataset = original_dataset
        self.cycle_length = cycle_length
        self.original_length = len(original_dataset)
    
    def __len__(self):
        return self.cycle_length
    
    def __getitem__(self, idx):
        # Cycle through the original dataset
        original_idx = idx % self.original_length
        return self.original_dataset[original_idx]

class RepeatDataset(Dataset):
    def __init__(self, sample, length):
        self.sample = sample
        self.length = length 
    
    def __len__(self):
        return self.length 

    def __getitem__(self, index):
        return self.sample



def get_dataloaders(config, tokenizer):
    data_loaders = {}
    for subset in ("train", "val"): #TODO: figure out why "test" fails 
        dataset = Craft3DDataset(
            config.data.data_dir,
            subset,
            tokenizer=tokenizer,
            max_samples=config.data.max_samples,
            voxel_side_len=config.data.voxel_side_len,
            air_not_air=config.data.air_not_air
        )
        if config.overfit:
            dataset = RepeatDataset(dataset[0], 20000) 

        if subset == "train":
            total_samples_per_epoch = 10000
            dataset = CycleDataset(dataset, total_samples_per_epoch)
            
        # TODO: remove or fix. Problem with forking is that there is overhead with spawn
        if config.model.model_name == "latent_diffusion" and False:
            # load encoder and get encoded dataset
            encoder_path = config.latent.encoder_ckpt
            vae_model = SparseStructureVAE.load_from_checkpoint(encoder_path)
            vae_model.eval()
            data_loaders[subset] = DataLoader(
                dataset,
                batch_size=config.loader.global_batch_size,
                shuffle=subset == "train",
                num_workers=config.loader.num_workers,
                pin_memory=config.loader.pin_memory,
                collate_fn=lambda batch: latent_collate_fn(batch, vae_model, "cuda")
            )
        else:
            data_loaders[subset] = DataLoader(
                dataset,
                batch_size=config.loader.global_batch_size,
                shuffle=subset == "train",
                num_workers=config.loader.num_workers,
                pin_memory=config.loader.pin_memory,
            )
    return data_loaders['train'], data_loaders['val']