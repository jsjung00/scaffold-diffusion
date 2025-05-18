from voxelcnn.datasets import Craft3DDataset
from torch.utils.data import DataLoader
import torch 
import json 


def build_data_loaders(config, tokenizer):

    data_loaders = {}
    for subset in ("train", "val", "test"):
        dataset = Craft3DDataset(
            config.data.data_dir,
            subset,
            tokenizer=tokenizer,
            max_samples=config.data.data_dir,
            next_steps=10,
        )
        data_loaders[subset] = DataLoader(
            dataset,
            batch_size=config.loader.global_batch_size,
            shuffle=subset == "train",
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.pin_memory,
        )
    return data_loaders