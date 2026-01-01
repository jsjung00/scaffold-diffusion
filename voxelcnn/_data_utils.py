import json
from typing import List, Dict, Set, Tuple, Any
from pathlib import Path 
import torch 
import os 


def load_and_delta_encode_voxels(filepath: str) -> Dict[str, Any]:
    """
    Load a JSON file containing a list of voxel frames and encode them using deltas.
    
    Args:
        filepath: Path to the JSON file containing list of lists of voxel dictionaries
        
    Returns:
        Dictionary with 'initial_frame' and 'deltas' for compact representation
    """
    # Load the JSON file
    with open(filepath, 'r') as f:
        frames = json.load(f)
    
    # Convert to delta encoding
    delta_encoded = create_delta_encoding(frames)
    
    return delta_encoded


def create_delta_encoding(frames: List[List[Dict]]) -> Dict[str, Any]:
    """
    Convert a sequence of voxel frames to delta encoding.
    
    Args:
        frames: List of frames, where each frame is a list of voxel dictionaries
        
    Returns:
        Dictionary with initial frame and deltas
    """
    if not frames:
        return {"initial_frame": [], "deltas": []}
    
    # Convert first frame to initial state
    initial_frame = frames[0].copy()
    deltas = []
    
    # Process each subsequent frame
    for i in range(1, len(frames)):
        prev_frame = frames[i - 1]
        curr_frame = frames[i]
        
        # Calculate delta between frames
        delta = calculate_frame_delta(prev_frame, curr_frame)
        deltas.append(delta)
    
    return {
        "initial_frame": initial_frame,
        "deltas": deltas
    }


def calculate_frame_delta(prev_frame: List[Dict], curr_frame: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Calculate the delta between two frames.
    
    Args:
        prev_frame: Previous frame's voxels
        curr_frame: Current frame's voxels
        
    Returns:
        Dictionary with 'added' and 'removed' voxels
    """
    # Create sets of voxel tuples for efficient comparison
    prev_voxels = voxel_list_to_set(prev_frame)
    curr_voxels = voxel_list_to_set(curr_frame)
    
    # Find added and removed voxels
    added_tuples = curr_voxels - prev_voxels
    removed_tuples = prev_voxels - curr_voxels
    
    # Convert back to list of dicts
    added = [tuple_to_voxel_dict(t) for t in added_tuples]
    removed = [tuple_to_voxel_dict(t) for t in removed_tuples]
    
    return {
        "added": added,
        "removed": removed
    }


def voxel_list_to_set(voxels: List[Dict]) -> Set[Tuple]:
    """
    Convert list of voxel dictionaries to a set of tuples for efficient comparison.
    
    Args:
        voxels: List of voxel dictionaries
        
    Returns:
        Set of tuples (x, y, z, block)
    """
    voxel_set = set()
    for v in voxels:
        # Create tuple with all voxel properties
        voxel_tuple = (v['x'], v['y'], v['z'], v.get('block', 0))
        voxel_set.add(voxel_tuple)
    return voxel_set


def tuple_to_voxel_dict(voxel_tuple: Tuple) -> Dict:
    """
    Convert voxel tuple back to dictionary format.
    
    Args:
        voxel_tuple: Tuple (x, y, z, block)
        
    Returns:
        Dictionary with voxel properties
    """
    return {
        'x': voxel_tuple[0],
        'y': voxel_tuple[1],
        'z': voxel_tuple[2],
        'block': voxel_tuple[3]
    }


def save_delta_encoded(delta_data: Dict[str, Any], output_filepath: str):
    """
    Save delta-encoded data to a JSON file.
    
    Args:
        delta_data: Delta-encoded voxel data
        output_filepath: Path to save the output JSON
    """
    with open(output_filepath, 'w') as f:
        json.dump(delta_data, f, indent=2)


def reconstruct_frame(initial_frame: List[Dict], deltas: List[Dict], frame_index: int) -> List[Dict]:
    """
    Reconstruct a specific frame from delta encoding.
    
    Args:
        initial_frame: The first frame
        deltas: List of delta changes
        frame_index: Index of frame to reconstruct (0 = initial frame)
        
    Returns:
        Reconstructed frame as list of voxel dictionaries
    """
    if frame_index == 0:
        return initial_frame.copy()
    
    # Start with initial frame
    current_voxels = voxel_list_to_set(initial_frame)
    
    # Apply deltas up to requested frame
    for i in range(min(frame_index, len(deltas))):
        delta = deltas[i]
        
        # Remove voxels
        removed_set = voxel_list_to_set(delta['removed'])
        current_voxels -= removed_set
        
        # Add voxels
        added_set = voxel_list_to_set(delta['added'])
        current_voxels |= added_set
    
    # Convert back to list of dicts
    return [tuple_to_voxel_dict(t) for t in current_voxels]


def convert_seq_to_deltas(json_seq):
    with open(json_seq, 'r') as fp:
        orig_seq = json.load(fp)
    
    delta_encoded = create_delta_encoding(orig_seq)

    p = Path(json_seq)
    stem, suffix, parent = p.stem, p.suffix, p.parent 
    new_path = parent / (stem + '_deltas' + suffix)
    
    with open(new_path, 'w') as f:
        json.dump(delta_encoded, f, indent=2)
    
    print(f"Saved delta compressed to {new_path}")
    return 

def voxel_seq_to_deltas(voxel_seq, file_name, base_dir=None):
    '''
    voxel_seq: List[torch.Tensor]
    '''
    import os 
    delta_encoded = create_delta_encoding(voxel_seq)

    new_path = os.path.join(base_dir, file_name)
    
    with open(new_path, 'w') as f:
        json.dump(delta_encoded, f, indent=2)
    
    print(f"Saved delta compressed to {new_path}")
    return 


def delta_encode_tensor(frames: torch.Tensor) -> Dict[str, Any]:
    """
    Convert a dense voxel tensor of shape (B, X, Y, Z) into
    {'initial_frame': [...], 'deltas': [...]} where each frame is
    represented sparsely.
    """
    B, X, Y, Z = frames.shape

    # --- initial frame: all non-zero voxels in frame 0 ---
    nz0 = (frames[0] != 0).nonzero(as_tuple=False)  # (N0, 3)
    initial_frame = [
        {
            "x": int(x), "y": int(y), "z": int(z),
            "block": int(frames[0, x, y, z].item())
        }
        for x, y, z in nz0
    ]

    deltas: List[Dict[str, List[Dict[str, int]]]] = []
    for i in range(1, B):
        prev = frames[i - 1]
        curr = frames[i]

        # find coords that changed
        changed = (curr != prev).nonzero(as_tuple=False)  # (Ni, 3)

        added: List[Dict[str, int]] = []
        removed: List[Dict[str, int]] = []
        for x, y, z in changed:
            pv = int(prev[x, y, z].item())
            cv = int(curr[x, y, z].item())

            if cv != 0:
                added.append({"x": int(x), "y": int(y), "z": int(z), "block": cv})
            if pv != 0:
                removed.append({"x": int(x), "y": int(y), "z": int(z), "block": pv})

        deltas.append({"added": added, "removed": removed})

    return {"initial_frame": initial_frame, "deltas": deltas}


def save_delta_encoded(
    frames: torch.Tensor,
    file_name: str,
    base_dir: str 
) -> None:
    """
    Delta-encode `frames` and write JSON to `output_path`.
    """
    out = delta_encode_tensor(frames)
    p = Path(os.path.join(base_dir, file_name))
    p.write_text(json.dumps(out, indent=2))
    print(f"Delta-encoded JSON saved to {p}")

def voxel_fill_sequence(
    tensor: torch.Tensor,
    replacement_value: int = 255
) -> List[torch.Tensor]:
    """
    Given an input tensor (of any shape) containing zeros and non‐zeros,
    returns a list of tensors where:
      - seq[0] is a clone of the original tensor
      - at each subsequent step, one additional non‐zero voxel is set to replacement_value
      - seq[-1] has all originally non‐zero voxels set to replacement_value

    Args:
        tensor:           torch.Tensor of any dtype/shape
        replacement_value: int value to write to each non‐zero location (default=255)

    Returns:
        List[torch.Tensor]: sequence of clones showing the gradual replacement
    """
    # Work on a clone so we don’t overwrite the user’s original
    t = tensor.clone()
    seq = [t.clone()]

    # Find all non-zero coordinates
    # as_tuple=False gives a M×D tensor of indices
    nz_indices = torch.nonzero(t, as_tuple=False).tolist()

    # (Optional) sort for deterministic order (e.g. lexicographic)
    nz_indices.sort()

    # For each non-zero coord, set it to replacement_value and record the snapshot
    for coord in nz_indices:
        # coord is a list of D integers
        t[tuple(coord)] = replacement_value
        seq.append(t.clone())

    return seq


# Example usage
if __name__ == "__main__":
    with open('output_files/house_example_0.json', 'r') as fp:
        list_objs = json.load(fp)
    
    example_house = torch.zeros(64,64,64)
    for obj in list_objs:
        example_house[obj['x'], obj['y'], obj['z']] = obj['block']

    seq = voxel_fill_sequence(example_house, replacement_value=255)

    save_delta_encoded(torch.stack(seq), "example_delta_seq.json")

    
    
    
    #convert_seq_to_deltas('/home/jsjung00/Desktop/Code/voxeldiffusion/output_files/house_0_seq.json')

