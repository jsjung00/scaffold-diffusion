import torch 
import numpy as np 
from nbtlib import tag 
import json 
import nbtlib 
from pathlib import Path
import sys 
import os 
sys.path.append(str(Path(__file__).parent.parent))

def voxel_tensor_to_nbt(voxel_tens):
    with open("block_id_map.json") as f:
        id_to_name = json.load(f)

    voxel = voxel_tens.detach().numpy()

    sx, sy, sz = voxel.shape #Note: minecraft takes in y,z,x order    
    size_tag = tag.List[tag.Int]([sx, sy, sz])
    palette = []
    palette_index = {}
    blocks_tag = tag.List[tag.Compound]()
    
    for x in range(sx):
        for y in range(sy):
            for z in range(sz):
                bid = int(voxel[x,y,z])
                
                name = id_to_name.get(str(bid))
                if name not in palette_index:
                    palette_index[name] = len(palette)
                    palette.append(tag.Compound({"Name" : tag.String(name)}))
                
                blocks_tag.append(tag.Compound({
                "pos" : tag.List[tag.Int]([x,y,z]),
                "state": tag.Int(palette_index[name])
                }))
    
    root = nbtlib.File({
        "size": size_tag, 
        "palette": tag.List[tag.Compound](palette),
        "blocks": blocks_tag,
        "entities": tag.List[tag.Compound]([]),
        "author": tag.String(''),
        "DataVersion": tag.Int('3779'),
        "version": tag.Int(1),
        "paletteMax": tag.Int(len(palette)-1) 
    })
    
    return root 

def voxel_to_nbt(voxel_tens, file_name, gzip=False):
    nbt_file = voxel_tensor_to_nbt(voxel_tens)
    BASE_DIR = Path(__file__).parent.parent 
    out_dir = BASE_DIR / "output_files"
    os.makedirs(out_dir, exist_ok=True) 
    out_path = out_dir / f"{file_name}_gzip_{gzip}.nbt"

    nbt_file.save(str(out_path), gzipped=gzip)



if __name__ == "__main__":
    all_zeros = torch.zeros(64,64,64)
    voxel_to_nbt(all_zeros, "test_zero")
    
    #nbt_file = voxel_tensor_to_nbt(all_zeros)
    #nbt_file.save("all_air.nbt")


