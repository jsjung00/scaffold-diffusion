import torch 
import numpy as np 
from nbtlib import tag 
import json 
import nbtlib 
from pathlib import Path
import sys 
import os 
from pathlib import Path
import numpy as np 
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent))
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import Dash, dcc, html
import copy 
from typing import List, Optional
import shutil 
from collections import Counter 
from scipy import stats 



def generate_figure(v1: np.ndarray, v2: np.ndarray, port:int=8050):
    if isinstance(v1, torch.Tensor):
        v1 = v1.cpu().numpy()
    if isinstance(v2, torch.Tensor):
        v2 = v2.cpu().numpy()

    # get coordinates of non-zero voxels
    x1, y1, z1 = np.where(v1 > 0)
    x2, y2, z2 = np.where(v2 > 0)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=x1, y=y1, z=z1,
        mode='markers',
        marker=dict(size=3, opacity=0.5, color='blue'),
        name='Voxel A'
    ))
    fig.add_trace(go.Scatter3d(
        x=x2, y=y2, z=z2,
        mode='markers',
        marker=dict(size=3, opacity=0.5, color='red'),
        name='Voxel B'
    ))
    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    app = Dash(__name__)
    app.layout = html.Div([
        html.H3("Voxel Overlay Viewer"),
        dcc.Graph(figure=fig, style={'height': '90vh'}),
    ])

    app.run(host='0.0.0.0', port=port)




def voxel_overlay_plotly(voxel1, voxel2, title1="Voxel Map 1", title2="Voxel Map 2", 
                        opacity1=0.6, opacity2=0.6, port=8050):
    """
    Create an interactive 3D overlay of two voxel maps using Plotly
    
    Args:
        voxel1, voxel2: 3D numpy arrays or torch tensors (X, Y, Z)
        title1, title2: Labels for the voxel maps
        opacity1, opacity2: Transparency values
        port: Port for localhost server
    """
    
    # Convert to numpy if torch tensors
    if isinstance(voxel1, torch.Tensor):
        voxel1 = voxel1.cpu().numpy()
    if isinstance(voxel2, torch.Tensor):
        voxel2 = voxel2.cpu().numpy()
    
    # Get coordinates where voxels are non-zero
    def get_voxel_coords(voxel_map):
        coords = np.where(voxel_map > 0)
        return coords[0], coords[1], coords[2], voxel_map[coords]
    
    x1, y1, z1, values1 = get_voxel_coords(voxel1)
    x2, y2, z2, values2 = get_voxel_coords(voxel2)
    
    # Create traces
    trace1 = go.Scatter3d(
        x=x1, y=y1, z=z1,
        mode='markers',
        marker=dict(
            size=3,
            color=values1,
            colorscale='Reds',
            opacity=opacity1,
            colorbar=dict(title=title1, x=0.0)
        ),
        name=title1
    )
    
    trace2 = go.Scatter3d(
        x=x2, y=y2, z=z2,
        mode='markers',
        marker=dict(
            size=3,
            color=values2,
            colorscale='Blues',
            opacity=opacity2,
            colorbar=dict(title=title2, x=1.0)
        ),
        name=title2
    )
    
    # Create layout
    layout = go.Layout(
        title='Voxel Overlay Visualization',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='cube'
        ),
        showlegend=True
    )
    
    fig = go.Figure(data=[trace1, trace2], layout=layout)
    
    # Show in browser (opens localhost)
    fig.show(port=port)
    
    return fig


def generate_block_colors(num_blocks=255, seed=42):
    '''
    Generate random colors for each block 
    '''
    np.random.seed(seed)

    colors = np.random.rand(num_blocks, 4)
    colors[:, 3] = 1.0
    colors[0] = [0,0,0,0] #air is transparent

    return colors 

def voxel_to_plot(voxel_tens, file_name, base_dir=None, color=True):
    '''
    voxel_tens: (torch.Tensor) Voxels containing block_ids, shape (B, X,Y,Z) or (X,Y,Z)

    color: (bool) Display blocks in color instead of just boolean air not air
    '''
     
    if base_dir is None:
        base_dir = Path(__file__).parent.parent / "output_files"

    if isinstance(base_dir, str):
        base_dir = Path(base_dir) 

    os.makedirs(base_dir, exist_ok=True)
    out_path = base_dir / f"{file_name}_matplot.png"

    if voxel_tens.ndim == 3:
        voxel_tens = voxel_tens.unsqueeze(0)

    B = voxel_tens.shape[0]
    voxels_batch = voxel_tens.detach().cpu().numpy()
    cols = int(np.ceil(np.sqrt(B)))
    rows = int(np.ceil(B / cols))

    fig = plt.figure(figsize=(4*cols, 4*rows))
    colormap = generate_block_colors() if color else None 

    for i in range(B):
        voxels = np.transpose(voxels_batch[i], (0,2,1)).astype(np.int64 if color else bool)
        ax = fig.add_subplot(rows, cols, i+1, projection='3d')

        if color:
            facecolors = colormap[voxels] #(X,Y,Z,4) of color values
            ax.voxels(voxels, facecolors=facecolors)
        else:
            ax.voxels(voxels)
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
    map_path = os.path.join(base_dir, "block_id_map.json") #block_id_map.json / minecraft_id_mapping

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

    inner_structure = tag.Compound({
        "size": size_tag, 
        "palette": tag.List[tag.Compound](palette),
        "blocks": blocks_tag,
        "entities": tag.List[tag.Compound]([]),
        "author": tag.String(''),
        "DataVersion": tag.Int('3779'),
        "version": tag.Int(1),
        "paletteMax": tag.Int(len(palette)-1) 
    })

    root = nbtlib.File({'': inner_structure})
    
    '''
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
    '''
    
    return root 

def voxel_to_schematic(voxel_tens, file_name, base_dir='/home/jsjung00/Desktop/Code/voxeldiffusion/output_files'):
    '''
    Convert voxel tensor to .schematic file 

    voxel_tens: (torch.Tensor) Shape (X,Y,Z) and contains int block_ids in [0,255]
    '''
    from nbtschematic import SchematicFile
    sf = SchematicFile(shape=voxel_tens.shape)
    assert sf.blocks.shape == voxel_tens.shape 

    X,Y,Z = voxel_tens.shape 

    for i in range(X):
        for j in range(Y):
            for k in range(Z):
                sf.blocks[j,k,i] = voxel_tens[i,j,k]

    if not file_name.lower().endswith('.schematic'):
        file_name += '.schematic'
    
    save_path = os.path.join(base_dir, file_name)
    sf.save(save_path)
    print(f"Saved schematic file at {save_path}")
    return 


def voxel_to_json(voxel_tens, file_name, base_dir=None, sparse=True):
    '''
    Convert voxel map to a json that can be loaded by our simple viewer
        Becomes an array of {x,y,z,block} objects
    
    voxel_tens: (torch.Tensor) Single map of shape (X,Y,Z) which contains minecraft block_ids
    '''
    voxel_list = [] 
    assert len(voxel_tens.shape) == 3

    (X,Y,Z) = voxel_tens.shape 
    for i in range(X):
        for j in range(Y):
            for k in range(Z):
                if sparse:
                    if voxel_tens[i,j,k].item() > 0:
                        voxel_list.append({'x': i, 'y': j, 'z': k,\
                                            'block': voxel_tens[i,j,k].item()})
                else:
                    voxel_list.append({'x': i, 'y': j, 'z': k,\
                                            'block': voxel_tens[i,j,k].item()})
                
    if not file_name.lower().endswith('.json'):
        file_name += '.json'
    
    out_path = os.path.join(base_dir, file_name) if base_dir else file_name

    with open(out_path, 'w') as f:
        json.dump(voxel_list, f, indent=2)

    print(f"wrote to file {out_path}")

    return 

    
    
def voxel_to_nbt(voxel_tens, file_name, base_dir=None, gzip=False):
    nbt_file = voxel_tensor_to_nbt(voxel_tens)
    if base_dir is None:
        base_dir = Path(__file__).parent.parent / "output_files"

    if isinstance(base_dir, str):
        base_dir = Path(base_dir) 

    os.makedirs(base_dir, exist_ok=True) 
    out_path = base_dir / f"{file_name}_gzip_{gzip}.nbt"

    nbt_file.save(str(out_path), gzipped=gzip)

def json_to_sequence(voxel_json, file_name, base_dir=None):
    '''
    Given a json that represents a house structure, return a json that is a sequence of 
        noised -> house structures
    '''
    sequence = [] # list of lists 

    with open(voxel_json, 'r') as fp:
        house_list = json.load(fp) #house represented as list of blocks 
        
        sequence.append(copy.deepcopy(house_list))

        prev_house = house_list 
        for i in range(0, len(house_list)):
            prev_house = copy.deepcopy(prev_house)
            prev_house[i]['block'] = 255  
            sequence.append(prev_house)

    reversed_seq = list(reversed(sequence))
    breakpoint()

    
    if not file_name.endswith('.json'):
        file_name += '.json'
    
    out_path = os.path.join(base_dir, file_name) if base_dir else file_name

    with open(out_path, 'w') as f:
        json.dump(reversed_seq, f, indent=2)



def process_voxel_data(json_file_path):
    """
    Process a JSON file containing voxel data with initial frame and deltas.
    
    Args:
        json_file_path: Path to the JSON file containing voxel data
        
    Returns:
        torch.Tensor: A 32x32x32 tensor containing the final block values
    """
    # Initialize 32x32x32 tensor with zeros
    voxel_grid = np.zeros((32, 32, 32), dtype=np.int32)
    
    # Load JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Process initial frame
    if "initial_frame" in data:
        for voxel in data["initial_frame"]:
            x, y, z = voxel["x"], voxel["y"], voxel["z"]
            block = voxel["block"]
            # Check bounds to ensure we're within [0, 31]
            if 0 <= x < 32 and 0 <= y < 32 and 0 <= z < 32:
                voxel_grid[x, y, z] = block
    
    # Process deltas sequentially
    if "deltas" in data:
        for delta in data["deltas"]:
            # Process removed voxels (set to 0)
            if "removed" in delta:
                for voxel in delta["removed"]:
                    x, y, z = voxel["x"], voxel["y"], voxel["z"]
                    # Check bounds
                    if 0 <= x < 32 and 0 <= y < 32 and 0 <= z < 32:
                        voxel_grid[x, y, z] = 0
            
            # Process added voxels (set to block value)
            if "added" in delta:
                for voxel in delta["added"]:
                    x, y, z = voxel["x"], voxel["y"], voxel["z"]
                    block = voxel["block"]
                    # Check bounds
                    if 0 <= x < 32 and 0 <= y < 32 and 0 <= z < 32:
                        voxel_grid[x, y, z] = block
    
    return torch.from_numpy(voxel_grid)


def calculate_voxel_percentages(folder_path: str) -> List[float]:
    """
    Process all JSON files in a folder and calculate the percentage of voxel values
    that match a predefined list.
    
    Args:
        folder_path: Path to the folder containing JSON files
        
    Returns:
        List of percentages (as floats) for each JSON file
    """
    # Define the target values (expanding ranges like 134-136, 219-234)
    target_values = [
        8, 10, 53, 54, 63, 67, 68, 71, 97, 108, 109, 114, 119, 128, 130,
        134, 135, 136,  # Expanded 134-136
        139, 144, 146, 156, 163, 164, 166, 176, 177, 180, 203, 217,
        219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234,  # Expanded 219-234
        243, 251, 252, 255
    ]
    
    percentages = []
    
    # Get all JSON files in the folder
    folder = Path(folder_path)
    json_files = list(folder.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return percentages
    
    # Process each JSON file
    for json_file in json_files:
        try:
            # Get voxel data from the JSON file
            voxel = process_voxel_data(str(json_file))
            
            # Calculate percentage of matching values
            percentage = 100*torch.isin(voxel, torch.tensor(target_values, dtype=torch.long )).sum().item() / torch.nonzero(voxel, as_tuple=False).shape[0]
            percentages.append(percentage)
            
            print(f"Processed {json_file.name}: {percentage:.2f} match")
            
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            # You might want to decide whether to skip this file or add 0.0
            # For now, we'll skip it
            continue
    
    return percentages

def get_clean_files(folder_path: str, threshold: float) -> List[float]:
    """
    Process all JSON files in a folder and create copy folder which only contains files whose bad blocks are less than threshold ratio 
    
    Args:
        folder_path: Path to the folder containing JSON files
    """
    # Define the target values (expanding ranges like 134-136, 219-234)
    target_values = [
        8, 10, 53, 54, 63, 67, 68, 71, 97, 108, 109, 114, 119, 128, 130,
        134, 135, 136,  # Expanded 134-136
        139, 144, 146, 156, 163, 164, 166, 176, 177, 180, 203, 217,
        219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234,  # Expanded 219-234
        243, 251, 252, 255
    ]
    
    percentages = []
    
    # Get all JSON files in the folder
    folder = Path(folder_path)
    new_folder = folder.parent / f"{folder.name}_copy"
    new_folder.mkdir(parents=True, exist_ok=True)

    json_files = list(folder.glob("*.json"))
    
    print(f"No JSON files found in {folder_path}")
    if not json_files:
        return percentages
    
    # Process each JSON file
    for json_file in json_files:
        try:
            # Get voxel data from the JSON file
            voxel = process_voxel_data(str(json_file))
            
            # Calculate percentage of matching values
            percentage = torch.isin(voxel, torch.tensor(target_values, dtype=torch.long )).sum().item() / torch.nonzero(voxel, as_tuple=False).shape[0]
            
            if percentage < threshold:
                dest = new_folder / json_file.name 
                shutil.copy2(json_file, dest)
                print(f"Processed {json_file.name}: {percentage:.2f} match")
            
            
            
            
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            # You might want to decide whether to skip this file or add 0.0
            # For now, we'll skip it
            continue

def get_entropy_and_percentage(folder_path: str):
    '''
    Returns three lists, entropy list, percentage of bad blocks, num of active blocks 
    '''
    target_values = [
        8, 10, 53, 54, 63, 67, 68, 71, 97, 108, 109, 114, 119, 128, 130,
        134, 135, 136,
        139, 144, 146, 156, 163, 164, 166, 176, 177, 180, 203, 217,
        219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234,
        243, 251, 252, 255
    ]
    
    entropies = []
    percentages = [] 
    num_active_blocks = []

    # Get all JSON files in the folder
    folder = Path(folder_path)
    json_files = list(folder.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {folder_path}")
        return entropies, percentages, num_active_blocks
    
    # Process each JSON file
    for json_file in json_files:
        try:
            # Get voxel data from the JSON file
            voxel = process_voxel_data(str(json_file))
            
            # Calculate percentage of matching values
            percentage = torch.isin(voxel, torch.tensor(target_values, dtype=torch.long )).sum().item() / torch.nonzero(voxel, as_tuple=False).shape[0]
            percentages.append(percentage)

            # Calculate entropy of the voxel vocab 
            active_voxel_values = torch.flatten(voxel[torch.nonzero(voxel, as_tuple=True)]).tolist()
            total = len(active_voxel_values)
            num_active_blocks.append(total)

            entropy = 0
            c = Counter(active_voxel_values)
            for value in range(256):
                count = c[value]
                p = count / total 
                if count > 0:
                    entropy -= (p * np.log(p))
            
            entropies.append(entropy) 

        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            # You might want to decide whether to skip this file or add 0.0
            # For now, we'll skip it
            continue
        
    return entropies, percentages, num_active_blocks    


def plot_percentage_histogram(percentages: List[float], 
                             bins: int = 20,
                             title: Optional[str] = None,
                             save_path: Optional[str] = None,
                             figsize: tuple = (10, 6),
                             color: str = 'skyblue',
                             show_stats: bool = True) -> None:
    """
    Create a histogram distribution of percentage values.
    
    Args:
        percentages: List of percentage values (0-100)
        bins: Number of bins for the histogram (default: 20)
        title: Custom title for the plot (optional)
        save_path: Path to save the figure (optional)
        figsize: Figure size as (width, height) tuple (default: (10, 6))
        color: Color for the histogram bars (default: 'skyblue')
        show_stats: Whether to show statistics on the plot (default: True)
    """
    if not percentages:
        print("No data to plot")
        return
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create the histogram
    n, bins_edges, patches = ax.hist(percentages, 
                                     bins=bins, 
                                     color=color, 
                                     edgecolor='black', 
                                     alpha=0.7)
    
    # Set labels and title
    ax.set_xlabel('Percentage (%)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title('Distribution of Voxel Match Percentages', fontsize=14, fontweight='bold')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Calculate statistics
    mean_val = np.mean(percentages)
    median_val = np.median(percentages)
    std_val = np.std(percentages)
    
    # Add statistics to the plot if requested
    if show_stats:
        # Add vertical lines for mean and median
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}%')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}%')
        
        # Add text box with statistics
        stats_text = f'Count: {len(percentages)}\nMean: {mean_val:.2f}%\nMedian: {median_val:.2f}%\nStd Dev: {std_val:.2f}%'
        ax.text(0.02, 0.98, stats_text, 
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10)
        
        # Add legend
        ax.legend(loc='upper right')
    
    # Set x-axis limits to 0-100 for percentage scale
    ax.set_xlim(0, 100)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    # Show the plot
    plt.show()

def compare_tensor_distributions(tensor1, tensor2, names=['Tensor 1', 'Tensor 2'], bins=50):
    """
    Comprehensive comparison of two PyTorch tensor distributions
    """
    # Convert to numpy for easier handling with matplotlib
    if torch.is_tensor(tensor1) and torch.is_tensor(tensor2):
        arr1 = tensor1.detach().cpu().numpy().flatten()
        arr2 = tensor2.detach().cpu().numpy().flatten()
    elif isinstance(tensor1, list) and isinstance(tensor2, list):
        arr1 = np.array(tensor1)
        arr2 = np.array(tensor2)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Tensor Distribution Comparison', fontsize=16)
    
    # 1. Overlaid Histograms
    axes[0, 0].hist(arr1, bins=bins, alpha=0.7, label=names[0], color='blue', density=True)
    axes[0, 0].hist(arr2, bins=bins, alpha=0.7, label=names[1], color='red', density=True)
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Overlaid Histograms')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Box Plot Comparison
    box_data = [arr1, arr2]
    box_plot = axes[0, 1].boxplot(box_data, labels=names, patch_artist=True)
    box_plot['boxes'][0].set_facecolor('blue')
    box_plot['boxes'][1].set_facecolor('red')
    axes[0, 1].set_title('Box Plot Comparison')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Density Plots (KDE)
    from scipy.stats import gaussian_kde
    
    # Create density estimates
    kde1 = gaussian_kde(arr1)
    kde2 = gaussian_kde(arr2)
    
    # Create x range for smooth plotting
    x_min = min(arr1.min(), arr2.min())
    x_max = max(arr1.max(), arr2.max())
    x_range = np.linspace(x_min, x_max, 200)
    
    axes[1, 0].plot(x_range, kde1(x_range), label=names[0], color='blue', linewidth=2)
    axes[1, 0].plot(x_range, kde2(x_range), label=names[1], color='red', linewidth=2)
    axes[1, 0].fill_between(x_range, kde1(x_range), alpha=0.3, color='blue')
    axes[1, 0].fill_between(x_range, kde2(x_range), alpha=0.3, color='red')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Density Plots (KDE)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Q-Q Plot
    # Sort both arrays for Q-Q plot
    sorted1 = np.sort(arr1)
    sorted2 = np.sort(arr2)
    
    # Interpolate to same length for comparison
    n = min(len(sorted1), len(sorted2))
    q1 = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(sorted1)), sorted1)
    q2 = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(sorted2)), sorted2)
    
    axes[1, 1].scatter(q1, q2, alpha=0.6, s=1)
    
    # Add diagonal line for reference
    min_val = min(q1.min(), q2.min())
    max_val = max(q1.max(), q2.max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
    
    axes[1, 1].set_xlabel(f'{names[0]} Quantiles')
    axes[1, 1].set_ylabel(f'{names[1]} Quantiles')
    axes[1, 1].set_title('Q-Q Plot')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def print_distribution_stats(tensor1, tensor2, names=['Tensor 1', 'Tensor 2']):
    """
    Print comprehensive statistical comparison
    """
    arr1 = tensor1.detach().cpu().numpy().flatten()
    arr2 = tensor2.detach().cpu().numpy().flatten()
    
    print("=" * 60)
    print("DISTRIBUTION STATISTICS COMPARISON")
    print("=" * 60)
    
    stats_dict = {}
    
    for i, (arr, name) in enumerate([(arr1, names[0]), (arr2, names[1])]):
        print(f"\n{name}:")
        print(f"  Mean:     {np.mean(arr):.6f}")
        print(f"  Std:      {np.std(arr):.6f}")
        print(f"  Median:   {np.median(arr):.6f}")
        print(f"  Min:      {np.min(arr):.6f}")
        print(f"  Max:      {np.max(arr):.6f}")
        print(f"  25th %:   {np.percentile(arr, 25):.6f}")
        print(f"  75th %:   {np.percentile(arr, 75):.6f}")
        print(f"  Skewness: {stats.skew(arr):.6f}")
        print(f"  Kurtosis: {stats.kurtosis(arr):.6f}")
        print(f"  Count:    {len(arr)}")
        
        stats_dict[name] = {
            'mean': np.mean(arr),
            'std': np.std(arr),
            'median': np.median(arr),
            'min': np.min(arr),
            'max': np.max(arr)
        }
    
    # Differences
    print(f"\nDIFFERENCES ({names[1]} - {names[0]}):")
    print(f"  Mean diff:   {stats_dict[names[1]]['mean'] - stats_dict[names[0]]['mean']:.6f}")
    print(f"  Std diff:    {stats_dict[names[1]]['std'] - stats_dict[names[0]]['std']:.6f}")
    print(f"  Median diff: {stats_dict[names[1]]['median'] - stats_dict[names[0]]['median']:.6f}")
    
    # Statistical tests
    print(f"\nSTATISTICAL TESTS:")
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_p = stats.ks_2samp(arr1, arr2)
    print(f"  Kolmogorov-Smirnov test:")
    print(f"    Statistic: {ks_stat:.6f}")
    print(f"    p-value:   {ks_p:.6f}")
    print(f"    Result:    {'Different distributions' if ks_p < 0.05 else 'Similar distributions'} (α=0.05)")
    
    # Mann-Whitney U test
    mw_stat, mw_p = stats.mannwhitneyu(arr1, arr2, alternative='two-sided')
    print(f"  Mann-Whitney U test:")
    print(f"    Statistic: {mw_stat:.6f}")
    print(f"    p-value:   {mw_p:.6f}")
    print(f"    Result:    {'Different medians' if mw_p < 0.05 else 'Similar medians'} (α=0.05)")
    
    # Wasserstein distance
    wasserstein_dist = stats.wasserstein_distance(arr1, arr2)
    print(f"  Wasserstein distance: {wasserstein_dist:.6f}")




if __name__ == "__main__":
    all_ones = torch.ones(64,64,64)
    #voxel_to_nbt(all_ones, "test_ones", gzip=True)

    #json_to_sequence('/home/jsjung00/Desktop/Code/voxeldiffusion/output_files/house_example_0.json', file_name='house_0_seq.json', base_dir='/home/jsjung00/Desktop/Code/voxeldiffusion/output_files')
    
    #percentages = calculate_voxel_percentages('/home/jsjung00/Desktop/Code/voxeldiffusion/output_files/07-08-2025-12-45-05')
    #plot_percentage_histogram(percentages, save_path=os.path.join('/home/jsjung00/Desktop/Code/voxeldiffusion/output_files', 'bad_dist.png'))
    get_clean_files('/home/jsjung00/Desktop/Code/voxeldiffusion/output_files/copy_with_ar', threshold=0.03)
    
    '''
    entropies_1024, percentages_1024, num_active_blocks_1024 = get_entropy_and_percentage('/home/jsjung00/Desktop/Code/voxeldiffusion/output_files/07-08-2025-21-37-05')
    entropies_128, percentages_128, num_active_blocks_128 = get_entropy_and_percentage('/home/jsjung00/Desktop/Code/voxeldiffusion/output_files/08-08-2025-22-05-55')


    fig = compare_tensor_distributions(entropies_1024, entropies_128, 
                                     names=['1024 entropy', '128 entropy'], 
                                     bins=60)
    fig.savefig("1024vs128entropy.png")

    fig = compare_tensor_distributions(percentages_1024, percentages_128, 
                                     names=['1024 percentage', '128 percentage'], 
                                     bins=60)
    fig.savefig("1024vs128percentage.png")

    fig = compare_tensor_distributions(num_active_blocks_1024, num_active_blocks_128, 
                                     names=['1024 numblocks', '128 numblocks'], 
                                     bins=60)
    fig.savefig("1024vs128numblocks.png")
    '''
    '''
    entropies, percentages, num_active_blocks = get_entropy_and_percentage('/home/jsjung00/Desktop/Code/voxeldiffusion/output_files/08-08-2025-22-05-55')

    # correlation plot of percentage and entropy
    plt.scatter(percentages, entropies)
    plt.xlim(0, 0.03)
    plt.xlabel('Percentage')
    plt.ylabel('Entropy')
    plt.savefig('percentage_versus_entropy_128.png')
    plt.clf()
    rho, p_value = stats.spearmanr(percentages, entropies)
    print(f"Spearman ρ: {rho:.3f}, p-value: {p_value:.3e}")

    # correlation plot of percentage and num blocks 
    plt.scatter(percentages, num_active_blocks)
    plt.xlim(0, 0.1)
    plt.xlabel('Percentage')
    plt.ylabel('Num blocks')
    plt.savefig('percentage_versus_numblocks_128.png')
    rho, p_value = stats.spearmanr(percentages, num_active_blocks)
    print(f"Spearman ρ: {rho:.3f}, p-value: {p_value:.3e}")

    #nbt_file = voxel_tensor_to_nbt(all_zeros)
    #nbt_file.save("all_air.nbt")
    '''


