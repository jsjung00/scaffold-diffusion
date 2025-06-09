from voxelcnn.datasets import Craft3DDataset
from torch.utils.data import DataLoader, Dataset, RandomSampler 
import torch 
import json 

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
            
      
        data_loaders[subset] = DataLoader(
            dataset,
            batch_size=config.loader.global_batch_size,
            shuffle=subset == "train",
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
        )
    return data_loaders['train'], data_loaders['val']