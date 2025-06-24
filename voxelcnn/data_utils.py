import torch 
import numpy as np 
from nbtlib import tag 
import json 
import nbtlib 
from pathlib import Path
import sys 
import os 
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def voxel_to_plot(voxel_tens, file_name, base_dir=None):
    '''
    voxel_tens: (torch.Tensor) Voxels containing block_ids, shape (X,Y,Z)
    '''
     
    if base_dir is None:
        base_dir = Path(__file__).parent.parent / "output_files"

    if isinstance(base_dir, str):
        base_dir = Path(base_dir) 

    os.makedirs(base_dir, exist_ok=True)
    out_path = base_dir / f"{file_name}_matplot.png"

    voxels = voxel_tens.detach().cpu().numpy()
    occupancy = voxels.astype(bool)
    occupancy = np.transpose(voxels, (0, 2, 1)).astype(bool)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(occupancy)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Voxel Visualization')
    plt.savefig(out_path)
    plt.close()





def voxel_tensor_to_nbt(voxel_tens):
    '''
    voxel_tens: (torch.Tensor) Voxels containing block_ids, shape (X,Y,Z)
    '''
    base_dir = os.path.dirname(__file__)
    map_path = os.path.join(base_dir, "block_id_map.json")

    with open(map_path) as f:
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

def voxel_to_nbt(voxel_tens, file_name, base_dir=None, gzip=False):
    nbt_file = voxel_tensor_to_nbt(voxel_tens)
    if base_dir is None:
        base_dir = Path(__file__).parent.parent / "output_files"

    if isinstance(base_dir, str):
        base_dir = Path(base_dir) 

    os.makedirs(base_dir, exist_ok=True) 
    out_path = base_dir / f"{file_name}_gzip_{gzip}.nbt"

    nbt_file.save(str(out_path), gzipped=gzip)



if __name__ == "__main__":
    all_zeros = torch.zeros(64,64,64)
    voxel_to_nbt(all_zeros, "test_zero")
    
    #nbt_file = voxel_tensor_to_nbt(all_zeros)
    #nbt_file.save("all_air.nbt")


