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
from voxelcnn.data_utils import voxel_to_nbt, voxel_to_plot
import diffusion
import utils
import json 
from datetime import datetime



omegaconf.OmegaConf.register_new_resolver(
    'cwd', os.getcwd)
omegaconf.OmegaConf.register_new_resolver(
    'device_count', torch.cuda.device_count)
omegaconf.OmegaConf.register_new_resolver(
    'eval', eval)
omegaconf.OmegaConf.register_new_resolver(
    'div_up', lambda x, y: (x + y - 1) // y)


def _load_from_checkpoint(config, tokenizer):
    return diffusion.Diffusion.load_from_checkpoint(
        config.eval.checkpoint_path,
        tokenizer=tokenizer,
        config=config)


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


def generate_samples(config, logger, tokenizer, save_traj=False):
    ''' 
    #TODO: fix. Our output should be a list of voxel maps? that way we can convert to a list of files if we want... 
    #NOTE: when we de-tokenize mask_id, it will convert to block value -1. You need to decide what block value mask_id should be
    '''
    logger.info('Generating samples.')
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    save_base_folder = os.path.join(BASE_DIR, "output_files")

    new_folder_path = os.path.join(save_base_folder, datetime.now().strftime('%d-%m-%Y-%H-%M-%S'))
    print("Making folder:", os.path.abspath(new_folder_path))
    os.makedirs(new_folder_path, exist_ok=True)

    model = _load_from_checkpoint(config=config,
                                  tokenizer=tokenizer)

    if config.eval.disable_ema:
        logger.info('Disabling EMA.')
        model.ema = None
    
    for batch_idx in range(config.sampling.num_sample_batches):
        samples, inter_values = model.restore_model_and_sample(
            num_steps=config.sampling.steps)
        if save_traj:
            raise ValueError("not implemented") 
        else:        
            # Save just the first generated structure
            voxels = model.tokenizer.detokenize(samples[0].detach().cpu())
            # manually convert all [MASK] tokens to lava 
            voxels[voxels == -1] = 10
            voxel_to_nbt(voxels, f"batch{batch_idx}_generated_sample", base_dir=new_folder_path, gzip=True)
            voxel_to_plot(voxels, f"batch{batch_idx}_generated_sample", base_dir=new_folder_path)
    return 


def _train(config, logger, tokenizer):
    logger.info('Starting Training.')
    wandb_logger = None
    if config.get('wandb', None) is not None:
        wandb_logger = L.pytorch.loggers.WandbLogger(
            config=omegaconf.OmegaConf.to_object(config),
            ** config.wandb)

    if (config.checkpointing.resume_from_ckpt
        and config.checkpointing.resume_ckpt_path is not None
        and utils.fsspec_exists(
            config.checkpointing.resume_ckpt_path)):
        ckpt_path = config.checkpointing.resume_ckpt_path
    else:
        ckpt_path = None

    # Lightning callbacks
    callbacks = []
    if 'callbacks' in config:
        for _, callback in config.callbacks.items():
            callbacks.append(hydra.utils.instantiate(callback))

    train_ds, valid_ds = dataloader.get_dataloaders(
        config, tokenizer)

    first_batch = next(iter(train_ds))
    breakpoint()
    voxel_to_plot(first_batch[0], "overfit_first_sample", base_dir='/home/jsjung00/Desktop/Code/voxeldiffusion/output_files')

    #model = diffusion.Diffusion(
    #    config, tokenizer)

    #TODO: finish the VAE model


    trainer = hydra.utils.instantiate(
        config.trainer,
        default_root_dir=os.getcwd(),
        callbacks=callbacks,
        strategy=hydra.utils.instantiate(config.strategy),
        logger=wandb_logger)
    trainer.fit(model, train_ds, valid_ds, ckpt_path=ckpt_path)


@hydra.main(version_base=None, config_path='configs',
            config_name='config')
def main(config):
    """Main entry point for training."""
    #L.seed_everything(config.seed)
    _print_config(config, resolve=True, save_cfg=True)

    logger = utils.get_logger(__name__)
    tokenizer = MinecraftTokenizer(config, config.data.air_not_air) # can later play around with changing mask token 

    if config.mode == 'sample_eval':
        generate_samples(config, logger, tokenizer)
    else:
        _train(config, logger, tokenizer)


if __name__ == '__main__':
    main()
