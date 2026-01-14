import os

import fsspec
import hydra
import lightning as L
import omegaconf
import rich.syntax
import rich.tree
import torch

import dataloader
from voxelcnn.datasets import MinecraftTokenizer
from voxelcnn.data_utils import voxel_to_nbt, voxel_to_plot, voxel_overlay_plotly, generate_figure
from voxelcnn._data_utils import save_delta_encoded
from callbacks.ema import EMA, EMAModelCheckpoint
import diffusion
from scaffold import ScaffoldDiffusion
from ar_baseline import ARBaseline
from baselines.multinomial.stage1.vqvae import VQVAE 
from baselines.multinomial.stage2.diffusion import latent_multinomial_diffusion
from baselines.multinomial.utils import get_class_freq, get_class_weights
import utils
import numpy as np 
import json 
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
os.environ['DISPLAY'] = ''  # Disable display
os.environ['MPLBACKEND'] = 'Agg'  # Force matplotlib backend
os.environ['WANDB_SILENT'] = 'true'
os.environ['WANDB_CONSOLE'] = 'off'



omegaconf.OmegaConf.register_new_resolver(
    'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
    'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
    'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
    'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(config, tokenizer, train_ds=None):
    if config.model.model_name == "scaffold_diffusion":
        return ScaffoldDiffusion.load_from_checkpoint(
            config.eval.checkpoint_path,
            tokenizer=tokenizer,
            config=config
        )
    elif config.model.model_name == "ar_baseline":
        return ARBaseline.load_from_checkpoint(
            config.eval.checkpoint_path,
            tokenizer=tokenizer, 
            config=config
        )
    elif config.model.model_name == "latent_multinomial":
        class_freq = get_class_freq(train_ds)
        class_weights = get_class_weights(class_freq)
        completion_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        print("\n NOTE: You have to be confident that your vqvae ckpt hasn't changed since training your latent model.\n")
        Dense = VQVAE.load_from_checkpoint(
            config.multinomial_vqvae.vqvae_ckpt,
            config=config, 
            multi_criterion=completion_criterion
        )
        Dense.freeze()

        return latent_multinomial_diffusion.load_from_checkpoint(
            config.multinomial_vqvae.diffusion_ckpt, 
            config=config,
            VAE_DENSE=Dense, 
            multi_criterion=completion_criterion
        )

    elif config.model.model_name == "latent_diffusion":
        return diffusion.Diffusion.load_from_checkpoint(
            config.eval.checkpoint_path,
            tokenizer=tokenizer)
    else:
        raise ValueError("model type not supported")
    

@L.pytorch.utilities.rank_zero_only
def _print_config(
        config: omegaconf.DictConfig,
        resolve: bool = True,
        save_cfg: bool = True) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
      config (DictConfig): Configuration composed by Hydra.
      resolve (bool): Whether to resolve reference fields of DictConfig.
      save_cfg (bool): Whether to save the configuration tree to a file.
    """

    style = 'dim'
    tree = rich.tree.Tree('CONFIG', style=style, guide_style=style)

    fields = config.keys()
    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, omegaconf.DictConfig):
            branch_content = omegaconf.OmegaConf.to_yaml(
                config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, 'yaml'))
    rich.print(tree)
    if save_cfg:
        with fsspec.open(
            '{}/config_tree.txt'.format(
                config.checkpointing.save_dir), 'w') as fp:
            rich.print(tree, file=fp)


def create_progressive_tensors(samples, mask_token_id, pad_mask):
    """
    Creates a list of tensors that progressively reveal the original samples.
    
    Args:
        samples: Tensor of shape (B, L)
        mask_token_id: The token ID to use for masking
        pad_mask: Tensor of shape (B,L). 1 if not pad 
    
    Returns:
        List of tensors, of variable shapes (T', L) where T' is the variable number of timesteps for that batch 
    """
    B, L = samples.shape
    result = []

    for batch_idx in range(B):
        sequence = samples[batch_idx]
        pad_sequence = pad_mask[batch_idx]
        progression_list = []     
        # First tensor: all mask tokens
        mask_tensor = torch.full_like(sequence, mask_token_id)
        progression_list.append(mask_tensor.clone())

        # Progressive reveal: each step reveals one more position
        for i in range(torch.sum(pad_sequence).item()):
            tensor = torch.full_like(sequence, mask_token_id)
            tensor[:i+1] = sequence[:i+1]
            progression_list.append(tensor)


        progression_tensor = torch.stack(progression_list) # (T', L)
        result.append(progression_tensor)
    
    return result

def generate_samples(config, logger, tokenizer, save_traj=False):
    ''' 
    Generates samples from the checkpoint model and saves as .nbt and .png files in a created folder 
    '''
    logger.info('Generating samples.')
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    save_base_folder = os.path.join(BASE_DIR, "output_files")

    new_folder_path = os.path.join(save_base_folder, datetime.now().strftime('%d-%m-%Y-%H-%M-%S'))
    print("Making folder:", os.path.abspath(new_folder_path))
    os.makedirs(new_folder_path, exist_ok=True)

    train_ds, valid_ds = dataloader.get_dataloaders(
        config, tokenizer)
    
    example_batch = next(iter(train_ds))

    model = _load_from_checkpoint(config=config,
                                  tokenizer=tokenizer,
                                  train_ds=train_ds)

    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None
    
    for batch_idx in range(config.sampling.num_sample_batches):
        if config.model.model_name == "ar_baseline":
            samples, token_pos, pad_mask = model.restore_model_and_sample()
            samples = samples.detach().cpu()

            if save_traj:
                inter_values = create_progressive_tensors(samples, tokenizer.mask_token_id, pad_mask) #List[(B,L)]
                batch_size = len(inter_values)
                for i in range(batch_size):
                    time_seq = inter_values[i] #(T, L)
                    time_len = time_seq.shape[0] 

                    time_seq_blocks = model.tokenizer.detokenize(time_seq)
                    sample_pad_mask = pad_mask[i].repeat(time_len, 1)
                    sample_token_pos = token_pos[i].repeat(time_len, 1, 1)
                    # normalize the y coordinate to make min 0 
                    active_token_pos = sample_token_pos[sample_pad_mask] #(N, 3)
                    sample_token_pos[:, :, 1] -= active_token_pos[:,1].min() 
                    time_seq_voxels = model._reshape_seq_voxels(time_seq_blocks, sample_token_pos, sample_pad_mask) #(T, X,Y,Z)
                    # convert MASK token to block_id 252
                    time_seq_voxels[time_seq_voxels == tokenizer.mask_token_id] = 252
                    # convert erroneous BOS token to block_id 252 (OR: hard code logits to not choose BOS)
                    time_seq_blocks[time_seq_blocks == tokenizer.bos_token_id] = 252
                    voxels = time_seq_voxels[-1]

                    save_delta_encoded(time_seq_voxels, f"batch{batch_idx}_seq{i}.json", base_dir=new_folder_path)
                    #voxel_to_nbt(voxels, f"batch{batch_idx}_sample_{i}", base_dir=new_folder_path, gzip=True)
                    #voxel_to_plot(voxels, f"batch{batch_idx}_sample_{i}", base_dir=new_folder_path)
            else:
                block_samples = model.tokenizer.detokenize(samples) #(B, L) containing pad tokens
                block_voxels = model._reshape_seq_voxels(block_samples, token_pos, pad_mask) #(B,X,Y,Z)
                for i in range(6):
                    voxels = block_voxels[i]
                    voxel_to_nbt(voxels, f"batch{batch_idx}_generated_sample_{i+1}", base_dir=new_folder_path, gzip=True)
                    voxel_to_plot(voxels, f"batch{batch_idx}_generated_sample_{i+1}", base_dir=new_folder_path)
        elif config.model.model_name == "latent_multinomial":
            samples = model.sample(example_batch).detach().cpu()
            if save_traj:
                raise ValueError("Not implemented yet. Shouldn't be too hard, we are just creating frames with left to right")
            else:
                block_samples = tokenizer.detokenize(samples) #(B,X,Y,Z)
                for i in range(6):
                    voxels = block_samples[i]
                    voxel_to_nbt(voxels, f"batch{batch_idx}_generated_sample_{i+1}", base_dir=new_folder_path, gzip=True)
                    voxel_to_plot(voxels, f"batch{batch_idx}_generated_sample_{i+1}", base_dir=new_folder_path)
        else:
            samples, inter_values, token_pos, pad_mask = model.restore_model_and_sample(
                num_steps=config.sampling.steps) #(B,L), List[B,L], (B,L, 3), (B, L). Pad_mask true if not pad token 
            samples = samples.detach().cpu()
            
            if save_traj:
                batch_seq = torch.stack(inter_values).detach().cpu()
                batch_size, time_len = batch_seq.shape[1], batch_seq.shape[0]
                for i in range(batch_size):
                    time_seq = batch_seq[:, i]  #(T, L)
                    time_seq_blocks = model.tokenizer.detokenize(time_seq)
                    sample_pad_mask = pad_mask[i].repeat(time_len, 1)
                    sample_token_pos = token_pos[i].repeat(time_len, 1, 1)
                    # normalize the y coordinate to make min 0 
                    active_token_pos = sample_token_pos[sample_pad_mask] #(N, 3)
                    sample_token_pos[:, :, 1] -= active_token_pos[:,1].min() 
                    time_seq_voxels = model._reshape_seq_voxels(time_seq_blocks, sample_token_pos, sample_pad_mask) #(T, X,Y,Z)
                    # convert MASK token to block_id 252
                    time_seq_voxels[time_seq_voxels == tokenizer.mask_token_id] = 252
                    voxels = time_seq_voxels[-1]

                    save_delta_encoded(time_seq_voxels, f"batch{batch_idx}_seq{i}.json", base_dir=new_folder_path)
                    voxel_to_nbt(voxels, f"batch{batch_idx}_sample_{i}", base_dir=new_folder_path, gzip=True)
                    voxel_to_plot(voxels, f"batch{batch_idx}_sample_{i}", base_dir=new_folder_path)
            else:        
                # Save just the first 8 generated structure
                block_samples = model.tokenizer.detokenize(samples) #(B, L) containing pad tokens
                block_voxels = model._reshape_seq_voxels(block_samples, token_pos, pad_mask) #(B,X,Y,Z)
                for i in range(6):
                    voxels = block_voxels[i]
                    voxel_to_nbt(voxels, f"batch{batch_idx}_generated_sample_{i+1}", base_dir=new_folder_path, gzip=True)
                    voxel_to_plot(voxels, f"batch{batch_idx}_generated_sample_{i+1}", base_dir=new_folder_path)
    return 

@torch.no_grad()
def autoencode(config, logger, tokenizer, save_traj=False):
    '''
    Calculate autoencoder voxel reconstruction accuracy
    '''
    logger.info('Generating samples.')
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    save_base_folder = os.path.join(BASE_DIR, "output_files")

    new_folder_path = os.path.join(save_base_folder, datetime.now().strftime('%d-%m-%Y-%H-%M-%S'))
    print("Making folder:", os.path.abspath(new_folder_path))
    os.makedirs(new_folder_path, exist_ok=True)

    train_ds, valid_ds = dataloader.get_dataloaders(
        config, tokenizer)

    model = _load_from_checkpoint(config=config,
                                  tokenizer=tokenizer,
                                  train_ds=train_ds)
    model.eval()
    
    batch_accs = []
    non_air_accs = []
    max_batches = 100
    i = 0 
    for batch in valid_ds:
        if i >= max_batches: break 
        batch = batch.to(model.device)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output, equal = model.reconstruct(batch)
        equal = equal.squeeze()

        nonair_equal = equal[batch == 1]
        air_equal = equal[batch == 0]
        non_air_acc = (torch.sum(nonair_equal) / nonair_equal.numel()).item()
        non_air_accs.append(non_air_acc)

        non_equal_voxels = (~equal.bool()).int()
        voxel_to_plot(non_equal_voxels[0], f"batch{i}_nonequal_{0}", base_dir=new_folder_path)
        voxel_to_plot(output[0], f"batch{i}_generated_sample_{0}", base_dir=new_folder_path)

        acc = torch.sum(equal) / equal.numel()
        batch_accs.append(acc.item())
        i += 1
    print(f"Mean accuracy {np.mean(batch_accs)} Mean nonair {np.mean(non_air_accs)}")
    

def _train(config, logger, tokenizer):
    logger.info('Starting Training.')
    wandb_logger = None
    if config.get('wandb', None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config),
            ** config.wandb)
  
    if (config.checkpointing.resume_from_ckpt
        and config.checkpointing.resume_ckpt_path is not None):
        ckpt_path = config.checkpointing.resume_ckpt_path
    else:
        ckpt_path = None

    # Lightning callbacks
    callbacks, hydra_callbacks = [], []
    if 'callbacks' in config:
        pass 
        for _, callback in config.callbacks.items():
            hydra_callbacks.append(hydra.utils.instantiate(callback))

    hydra_dir = os.getcwd()
    print(f"CWD {hydra_dir}")
    checkpoint_every_n_steps = EMAModelCheckpoint(save_top_k=-1, save_last=False,\
                                                   dirpath=hydra_dir, verbose=True,
                                                   auto_insert_metric_name=False,
                                                   every_n_train_steps=5000)

    checkpoint_monitor = EMAModelCheckpoint(monitor='val/loss', mode='min', save_top_k=1,\
                                            dirpath=hydra_dir, auto_insert_metric_name=False, verbose=True)

    callbacks.extend([checkpoint_every_n_steps, checkpoint_monitor])


    if config.training.use_ema:
        callbacks.append(EMA(decay=config.training.ema)) 

    train_ds, valid_ds = dataloader.get_dataloaders(
        config, tokenizer)

    if config.model.model_name == "scaffold_diffusion":
        model = ScaffoldDiffusion(config, tokenizer)
    elif config.model.model_name == "ar_baseline":
        model = ARBaseline(config, tokenizer)
    elif config.model.model_name == "vqvae_baseline":
        class_freq = get_class_freq(train_ds)
        class_weights = get_class_weights(class_freq)
        completion_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        model = VQVAE(config, completion_criterion)
    elif config.model.model_name == "latent_multinomial":
        class_freq = get_class_freq(train_ds)
        class_weights = get_class_weights(class_freq)
        completion_criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        Dense = VQVAE.load_from_checkpoint(
            config.multinomial_vqvae.vqvae_ckpt,
            config=config, 
            multi_criterion=completion_criterion
        )
        Dense.freeze()
        model = latent_multinomial_diffusion(config, Dense, completion_criterion)
    else:
        raise ValueError("Incorrect model type specified")


    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=hydra_callbacks, 
        strategy="auto",
        logger=wandb_logger,
        profiler=None,
        enable_checkpointing=True) 
    
    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):

    """Main entry point for training."""
    _print_config(config, resolve=True, save_cfg=True)

    logger = utils.get_logger(__name__)
    tokenizer = dataloader.get_tokenizer(config)
   
    if config.mode == 'sample_eval':
        generate_samples(config, logger, tokenizer, save_traj=True)
    elif config.mode == 'autoencode':
        autoencode(config, logger, tokenizer)
    else:
        _train(config, logger, tokenizer)


if __name__ == '__main__':
    main()
