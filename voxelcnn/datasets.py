#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import tarfile
import warnings
from os import path as osp
from typing import Dict, Optional, Tuple
import sys 
from pathlib import Path 
from box import Box
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import requests
import torch
from torch.utils.data import Dataset
from voxelcnn.data_utils import voxel_to_nbt, voxel_to_plot, voxel_to_json, voxel_to_schematic

#torch.manual_seed(42)

PARENT = Path(__file__).parent
class MinecraftTokenizer:
    def __init__(self, config, air_not_air=False, block_map_file='block_id_map.json'):
        self.config = config
        self.air_not_air = air_not_air 
        
        if not self.air_not_air:
            with open(PARENT / block_map_file, 'r') as f:
                block_map = json.load(f) #block_id : name 

            pairs = sorted(block_map.items(), key= lambda kv: int(kv[0]))
            keys, vals = zip(*pairs)  
            print(f"Confirm that the first val is air: {vals[0]}")   

            self.block_id_token_map = {int(block_id): i for i, block_id in enumerate(keys)} #block_id : token_id
            new_pairs = sorted(self.block_id_token_map.items())
            block_ids, token_ids = zip(*new_pairs)
            block_ids = torch.tensor(block_ids, dtype=torch.long)
            token_ids = torch.tensor(token_ids, dtype=torch.long)
        else:
            block_ids = torch.tensor([0,1], dtype=torch.long)
            token_ids = torch.tensor([0, 1], dtype=torch.long) 

        if self.config.mask_token_id is not None:
            raise ValueError("must not supply mask token id")
            self.vocab_size = len(self.block_id_token_map)
            self.mask_token_id = self.config.mask_token_id #make air the mask token 
        #set the mask token id and the pad token id to be the two largest
        self.pad_token_id = len(block_ids)
        self.mask_token_id = len(block_ids) + 1
        if config.model.model_name == 'ar_baseline':
            self.bos_token_id = len(block_ids) + 2
            self.vocab_size = len(block_ids) + 3
        else:
            self.vocab_size = len(block_ids)+2
        

        if self.air_not_air:
            self.id2token = torch.full((self.vocab_size,), fill_value=-1, dtype=torch.long)
            self.id2token[block_ids] = token_ids
            self.token2blockid = torch.full((self.vocab_size,), fill_value=-1, dtype=torch.long)
            self.token2blockid[token_ids] = block_ids
            return 


        # create a map from block_id to token_id by simply doing self.id2token[voxel]
        max_block_id = max(self.block_id_token_map.keys())
        id2token = torch.full((max_block_id+1,), fill_value=-1, dtype=torch.long)
        id2token[block_ids] = token_ids
        self.id2token = id2token 

        # create a map from token_id to block_id. mask_id and pad_id maps to itself 
        token2blockid = torch.full((self.vocab_size,), fill_value=-1, dtype=torch.long)
        token2blockid[token_ids] = block_ids # first set all the tokens that label the minecraft voxels
        token2blockid[self.pad_token_id] = self.pad_token_id
        token2blockid[self.mask_token_id] = self.mask_token_id
        if config.model.model_name == "ar_baseline": 
            token2blockid[self.bos_token_id] = self.bos_token_id

        self.token2blockid = token2blockid
        
    
    def tokenize(self, voxel_tens):
        '''
        Takes in a voxel tensor of [B,W,L,H] that contains block_id values and converts to tokens in [0, vocab_len-1]

        Return: voxel tensor of shape [B,W,L,H] that contains token_ids in [0, vocab_len-1]
        '''
        return self.id2token[voxel_tens]

    def detokenize(self, voxel_tens):
        '''
        Take in voxel tensor of [B,W,L,H] that contains tokens in [0, vocab_len-1]

        Return: voxel tensor of shape [B,W,L,H] that contains block_ids 
        '''
        return self.token2blockid[voxel_tens]


class Craft3DDataset(Dataset):
    NUM_BLOCK_TYPES = 256
    URL = "https://craftassist.s3-us-west-2.amazonaws.com/pubr/house_data.tar.gz"

    def __init__(
        self,
        data_dir: str,
        subset: str,
        tokenizer, 
        voxel_side_len: int = 64, 
        local_size: int = 7,
        global_size: int = 21,
        history: int = 3,
        max_samples: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
        air_not_air: bool = False,
        middle_crop: bool = False,
        max_active_tokens: Optional[int] = None,
        translate: bool = True, 
        rotate: bool = True,
        max_translation: int = 4   
    ):
        """ Download and construct 3D-Craft dataset

        data_dir (str): Directory to save/load the dataset
        subset (str): 'train' | 'val' | 'test'
        voxel_side_len (int): Length of voxel map that we return
        local_size (int): Local context size. Default: 7
        global_size (int): Global context size. Default: 21
        history (int): Number of previous steps considered as inputs. Default: 3
        next_steps (int): Number of next steps considered as targets. Default: -1,
            meaning till the end
        max_samples (int, optional): Limit the maximum number of samples. Used for
            faster debugging. Default: None, meaning no limit
        logger (logging.Logger, optional): A logger. Default: None, meaning will print
            to stdout
            air_not_air (boolean): If true, then house structure has two token IDs, one air and one non-air. 
            middle_crop: (Bool) If true, crop the middle voxel_side_len**3 of the house structure and fill rest with zeros
        max_active_tokens (int, optional): Remove structure if number of non-air blocks exceeds this number
        """
        super().__init__()
        self.data_dir = data_dir
        self.subset = subset
        self.tokenizer = tokenizer
        self.voxel_side_len = voxel_side_len
        self.local_size = local_size
        self.global_size = global_size
        self.history = history
        self.max_local_distance = self.local_size // 2
        self.max_global_distance = self.global_size // 2
        self.max_samples = max_samples
        self.logger = logger
        self.air_not_air = air_not_air
        self.middle_crop = middle_crop 
        self.max_active_tokens = max_active_tokens

        self.translate = translate
        self.rotate = rotate 
        self.max_translation = max_translation if self.translate else 0  

        if self.subset not in ("train", "val", "test"):
            raise ValueError(f"Unknown subset: {self.subset}")

        if not self._has_raw_data():
            self._download()

        self._load_dataset()


    def __len__(self) -> int:
        """ Get number of valid blocks """
        ret = len(self._all_houses)
        if self.max_samples is not None:
            ret = min(ret, self.max_samples)
        return ret

    def __getitem__(
        self, index: int
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """ Get the index-th valid voxel map containing a house structure and convert to token_ids 

        Returns:
            Voxel map tensor of size (voxel_side_len, voxel_side_len, voxel_side_len)
        """
        voxel_map = self._all_houses[index]
        if self.tokenizer is not None:
            voxel_tokens = self.tokenizer.tokenize(voxel_map)
        else:
            voxel_tokens = voxel_map

        if self.rotate:
            voxel_tokens = self.apply_rotation(voxel_tokens)
        if self.translate:
            voxel_tokens = self.apply_translation(voxel_tokens)

        return voxel_tokens 

    def get_percentage_nonair(self) -> int:
        '''Get the percentage of voxels that are air'''
        num_non_air = 0 
        num_total = 0 
        for voxels in self._all_houses:
            num_non_air += torch.count_nonzero(voxels)
            num_total += voxels.numel()
        return num_non_air / num_total


    def get_num_houses(self) -> int:
        """ Get the total number of houses. Use for thorough evaluation """
        return len(self._all_houses)

    def _log(self, msg: str):
        if self.logger is None:
            print(msg)
        else:
            self.logger.info(msg)

    def _has_raw_data(self) -> bool:
        return osp.isdir(osp.join(self.data_dir, "houses"))

    def _download(self):
        os.makedirs(self.data_dir, exist_ok=True)

        tar_path = osp.join(self.data_dir, "houses.tar.gz")
        if not osp.isfile(tar_path):
            self._log(f"Downloading dataset from {Craft3DDataset.URL}")
            response = requests.get(Craft3DDataset.URL, allow_redirects=True)
            if response.status_code != 200:
                raise RuntimeError(
                    f"Failed to retrieve image from url: {Craft3DDataset.URL}. "
                    f"Status: {response.status_code}"
                )
            with open(tar_path, "wb") as f:
                f.write(response.content)

        extracted_dir = osp.join(self.data_dir, "houses")
        if not osp.isdir(extracted_dir):
            self._log(f"Extracting dataset to {extracted_dir}")
            tar = tarfile.open(tar_path, "r")
            tar.extractall(self.data_dir)

    def apply_rotation(self, voxel_cube):
        '''
        Randomly rotate 

        voxel_cube: (torch.Tensor) Shape (X,Y,Z) contains air padding
        '''
        # uniform sample from [0, 90, 180, 270]
        num_90_rotations = torch.randint(low=0, high=4, size=(1,)).item()

        if num_90_rotations == 0:
            return voxel_cube
        
        return torch.rot90(voxel_cube, k=num_90_rotations, dims=(0,1))

    def apply_translation(self, voxel_cube):
        '''
        Randomly translate 

        voxel_cube: (torch.Tensor) Shape (X,Y,Z) contains air padding
        '''
        # uniformly sample axes (x,y, z) to translate + translation amount
        translation_amount = torch.randint(low=0, high=self.max_translation+1, size=()).item()
        translate_direction = torch.randint(low=0, high=2, size=()).item()*2 - 1
        translate_dim = torch.randint(low=0, high=3, size=()).item() 

        if translation_amount == 0:
            return voxel_cube.clone()

        new_voxel_cube = torch.zeros_like(voxel_cube)
        if translate_dim == 0:
            if translate_direction == 1:
                new_voxel_cube[translation_amount:, :, :] = voxel_cube[0:-translation_amount, :, :]
            else:
                new_voxel_cube[0:-translation_amount, :, :] = voxel_cube[translation_amount:, :, :]
        elif translate_dim == 1:
            if translate_direction == 1:
                new_voxel_cube[:, translation_amount:, :] = voxel_cube[:, 0:-translation_amount, :]
            else:
                new_voxel_cube[:, 0:-translation_amount, :] = voxel_cube[:, translation_amount:, :] 
        else:
            if translate_direction == 1:
                new_voxel_cube[:, :, translation_amount:] = voxel_cube[:,:, 0:-translation_amount]
            else:
                new_voxel_cube[:,:, 0:-translation_amount] = voxel_cube[:,:, translation_amount:]
            

        return new_voxel_cube
    
    def _get_house_voxels(self, annotation: torch.Tensor):
        '''
        Given my annotation or house structure that is shape (N,4) where each block represented
            by [block_id, x,y,z],
            
            
        If house doesn't within (voxel_size-max_translation)**3 return None
            Else return voxel map of size (voxel_side_len,voxel_side_len,voxel_side_len)
        '''
        coords = annotation[:, 1:].long()
        block_ids  = annotation[:, 0].long() 
        mins, _ = coords.min(dim=0)
        maxs, _ = coords.max(dim=0)
        spans = maxs - mins + 1 
        if spans.max() > self.voxel_side_len - (self.max_translation*2):
            return None

        slack = self.voxel_side_len - spans 
        offset = slack // 2

        new_xyz = coords - mins + offset 

        voxels = torch.zeros((self.voxel_side_len,self.voxel_side_len,self.voxel_side_len), dtype=torch.long)
        voxels[new_xyz[:, 0], new_xyz[:, 1], new_xyz[:, 2]] = block_ids

        if self.air_not_air:
            voxels = (voxels > 0).long()  

        return voxels 
        
    def _load_dataset(self):
        splits_path = osp.join(self.data_dir, "splits.json")

        if not osp.isfile(splits_path):
            raise RuntimeError(f"Split file not found at: {splits_path}")

        with open(splits_path, "r") as f:
            splits = json.load(f)

        self._all_houses = []
        total_files = 0
        max_len = 0
        for filename in splits[self.subset]:
            total_files += 1

            annotation = osp.join(self.data_dir, "houses", filename, "placed.json")
            if not osp.isfile(annotation):
                warnings.warn(f"No annotation file for: {annotation}")
                continue
            annotation = self._load_annotation(annotation)
            voxel_map = self._get_house_voxels(annotation)

            valid_house = len(annotation) >= 100 and voxel_map is not None 
            if valid_house and self.max_active_tokens is not None:
                num_active_tokens = torch.numel(voxel_map[voxel_map != 0])  
                valid_house &= (num_active_tokens <= self.max_active_tokens)    

            if valid_house:
                self._all_houses.append(voxel_map)
                max_len = max(max_len, len(annotation))
        
        print(f"Original num houses: {total_files}; filtered num houses {len(self._all_houses)}")

    def _load_annotation(self, annotation_path: str) -> torch.Tensor:
        with open(annotation_path, "r") as f:
            annotation = json.load(f)
        final_house = {}
        types_and_coords = []
        last_timestamp = -1
        for i, item in enumerate(annotation):
            timestamp, annotator_id, coordinate, block_info, action = item
            assert timestamp >= last_timestamp
            last_timestamp = timestamp
            coordinate = tuple(np.asarray(coordinate).astype(np.int64).tolist())
            block_type = int(block_info[0]) & 0xFF
            #block_type = np.asarray(block_info, dtype=np.uint8)[0]
            if action == "B":
                final_house.pop(coordinate, None)
            else:
                final_house[coordinate] = i
            types_and_coords.append((block_type,) + coordinate)
        indices = sorted(final_house.values())
        types_and_coords = [types_and_coords[i] for i in indices]
        return torch.tensor(types_and_coords, dtype=torch.int64)

if __name__ == "__main__":
    work_dir = osp.join(osp.dirname(osp.abspath(__file__)), "..")
    config = Box({'mask_token_id': None})
    tokenizer = MinecraftTokenizer(config, air_not_air=False)
    dataset = Craft3DDataset(osp.join(work_dir, "data"), "train", voxel_side_len=64,\
                             tokenizer=tokenizer, air_not_air=False, max_samples=1, max_translation=0,\
                                rotate=False, translate=False, max_active_tokens=1024)
    
    valdataset = Craft3DDataset(osp.join(work_dir, "data"), "val", voxel_side_len=32,\
                             tokenizer=tokenizer, air_not_air=True, max_samples=1)

    no_token_dataset = Craft3DDataset(osp.join(work_dir, "data"), "train", voxel_side_len=64,\
                             tokenizer=None, max_active_tokens=1024, rotate=True, translate=True, max_translation=4)



    breakpoint()
    for i in range(5):
        house = dataset[i]
        #house = no_token_dataset[i]
        voxel_to_json(house, f"house_example_{i}", base_dir='/home/jsjung00/Desktop/Code/voxeldiffusion/output_files')
        #voxel_to_schematic(house, f'example_{i}')
        #voxel_to_nbt(house, f"example_{i}", base_dir='/home/jsjung00/Desktop/Code/voxeldiffusion/output_files')
        #house_blocks = tokenizer.detokenize(house)
        break 
        #voxel_to_plot(house_blocks, f"sample{i}", base_dir='/home/jsjung00/Desktop/Code/voxeldiffusion/output_files')
        # Note: need to de-tokenize and get block_ids before saving to nbt 
        #voxel_to_nbt(house_blocks, f"air_not_air_{i}", base_dir='/home/jsjung00/Desktop/Code/voxeldiffusion/output_files', gzip=True)
        
