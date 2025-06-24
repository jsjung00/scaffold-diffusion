import itertools
import math
import os
import typing
from dataclasses import dataclass

import hydra.utils
import lightning as L
import numpy as np
import torch
import torch.nn.functional as F
import torchmetrics
from torch import Tensor

from data_utils import voxel_to_plot

import dataloader
import models
import models.dit3d_window
import noise_schedule
import utils
import os 

LOG2 = math.log(2)


def _sample_categorical(categorical_probs):
    gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
    return x.view(*x.shape, *((1,) * (len(reference.shape) - len(x.shape))))


@dataclass
class Loss:
    loss: torch.FloatTensor
    nlls: torch.FloatTensor
    token_mask: torch.FloatTensor
    non_air_nll: torch.FloatTensor
    air_nll : torch.FloatTensor 


class NLL(torchmetrics.aggregation.MeanMetric):
    pass


class BPD(NLL):
    def compute(self) -> Tensor:
        """Computes the bits per dimension.

        Returns:
          bpd
        """
        return self.mean_value / self.weight / LOG2


class Perplexity(NLL):
    def compute(self) -> Tensor:
        """Computes the Perplexity.

        Returns:
         Perplexity
        """
        return torch.exp(self.mean_value / self.weight)


class Diffusion(L.LightningModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.val_sample_freq = config.sampling.val_sample_freq

        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.sampler = self.config.sampling.predictor
        self.gen_ppl_eval_model_name_or_path = (
            self.config.eval.gen_ppl_eval_model_name_or_path
        )
        self.antithetic_sampling = self.config.training.antithetic_sampling
        self.importance_sampling = self.config.training.importance_sampling
        self.change_of_variables = self.config.training.change_of_variables
        self.mask_index = self.tokenizer.mask_token_id
        self.parameterization = self.config.parameterization
        if self.config.backbone == "unet":
            self.backbone = models.unet.UNet3D(
                in_channels=1, out_channels=self.vocab_size, f_maps=self.config.model.f_maps, num_levels=self.config.model.num_levels
            )
        elif self.config.backbone == "dit_3d":
            self.backbone = models.dit3d_window.DiT_B_4(vocab_dim=self.vocab_size, input_size=self.config.data.voxel_side_len)
        else:
            raise ValueError(f"Unknown backbone: {self.config.backbone}")

        self.T = self.config.T
        self.subs_masking = self.config.subs_masking

        self.softplus = torch.nn.Softplus()
        # metrics are automatically reset at end of epoch
        metrics = torchmetrics.MetricCollection(
            {
                "nll": NLL(),
                "bpd": BPD(),
                "ppl": Perplexity(),
            }
        )
        metrics.set_dtype(torch.float64)
        self.train_metrics = metrics.clone(prefix="train/")
        self.valid_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        self.noise = noise_schedule.get_noise(self.config, dtype=self.dtype)
        if self.config.training.ema > 0:
            self.ema = models.ema.ExponentialMovingAverage(
                itertools.chain(self.backbone.parameters(), self.noise.parameters()),
                decay=self.config.training.ema,
            )
        else:
            self.ema = None

        self.lr = self.config.optim.lr
        self.sampling_eps = self.config.training.sampling_eps
        self.time_conditioning = self.config.time_conditioning
        self.neg_infinity = -1000000.0
        self.fast_forward_epochs = None
        self.fast_forward_batches = None
        self._validate_configuration()

    def _validate_configuration(self):
        assert not (self.change_of_variables and self.importance_sampling)
        if self.parameterization == "sedd":
            assert not self.importance_sampling
            assert not self.change_of_variables
        if self.parameterization == "d3pm":
            assert self.T > 0
        if self.T > 0:
            assert self.parameterization in {"d3pm", "subs"}
        if self.subs_masking:
            assert self.parameterization == "d3pm"

    def on_load_checkpoint(self, checkpoint):
        if self.ema:
            self.ema.load_state_dict(checkpoint["ema"])
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py#L41
        self.fast_forward_epochs = checkpoint["loops"]["fit_loop"]["epoch_progress"][
            "current"
        ]["completed"]
        self.fast_forward_batches = checkpoint["loops"]["fit_loop"][
            "epoch_loop.batch_progress"
        ]["current"]["completed"]

    def on_save_checkpoint(self, checkpoint):
        if self.ema:
            checkpoint["ema"] = self.ema.state_dict()
        # Copied from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/tasks/seq.py
        # ['epoch_loop.batch_progress']['total']['completed'] is 1 iteration
        # behind, so we're using the optimizer's progress.
        checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]["total"][
            "completed"
        ] = (
            checkpoint["loops"]["fit_loop"][
                "epoch_loop.automatic_optimization.optim_progress"
            ]["optimizer"]["step"]["total"]["completed"]
            * self.trainer.accumulate_grad_batches
        )
        checkpoint["loops"]["fit_loop"]["epoch_loop.batch_progress"]["current"][
            "completed"
        ] = (
            checkpoint["loops"]["fit_loop"][
                "epoch_loop.automatic_optimization.optim_progress"
            ]["optimizer"]["step"]["current"]["completed"]
            * self.trainer.accumulate_grad_batches
        )
        # _batches_that_stepped tracks the number of global steps, not the number
        # of local steps, so we don't multiply with self.trainer.accumulate_grad_batches here.
        checkpoint["loops"]["fit_loop"]["epoch_loop.state_dict"][
            "_batches_that_stepped"
        ] = checkpoint["loops"]["fit_loop"][
            "epoch_loop.automatic_optimization.optim_progress"
        ][
            "optimizer"
        ][
            "step"
        ][
            "total"
        ][
            "completed"
        ]
        if "sampler" not in checkpoint.keys():
            checkpoint["sampler"] = {} 
        if hasattr(self.trainer.train_dataloader.sampler, "state_dict"):
            sampler_state_dict = self.trainer.train_dataloader.sampler.state_dict()
            checkpoint["sampler"]["random_state"] = sampler_state_dict.get(
                "random_state", None
            )
        else:
            checkpoint["sampler"]["random_state"] = None

    def on_train_start(self):
        print(f"Trainer has val dataloader: {self.trainer.val_dataloaders is not None}")
        if self.trainer.val_dataloaders is not None:
            print(f"Val dataloader length: {len(self.trainer.val_dataloaders)}")

        if self.ema:
            self.ema.move_shadow_params_to_device(self.device)

        return
        # Adapted from:
        # https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py
        distributed = (
            self.trainer._accelerator_connector.use_distributed_sampler
            and self.trainer._accelerator_connector.is_distributed
        )
        if distributed:
            sampler_cls = dataloader.FaultTolerantDistributedSampler
        else:
            sampler_cls = dataloader.RandomFaultTolerantSampler
        updated_dls = []
        for dl in self.trainer.fit_loop._combined_loader.flattened:
            if hasattr(dl.sampler, "shuffle"):
                dl_sampler = sampler_cls(dl.dataset, shuffle=dl.sampler.shuffle)
            else:
                dl_sampler = sampler_cls(dl.dataset)
            if (
                distributed
                and self.fast_forward_epochs is not None
                and self.fast_forward_batches is not None
            ):
                dl_sampler.load_state_dict(
                    {
                        "epoch": self.fast_forward_epochs,
                        "counter": (
                            self.fast_forward_batches * self.config.loader.batch_size
                        ),
                    }
                )
            updated_dls.append(
                torch.utils.data.DataLoader(
                    dl.dataset,
                    batch_size=self.config.loader.batch_size,
                    num_workers=self.config.loader.num_workers,
                    pin_memory=self.config.loader.pin_memory,
                    sampler=dl_sampler,
                    shuffle=False,
                    persistent_workers=True,
                )
            )
        self.trainer.fit_loop._combined_loader.flattened = updated_dls

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.ema:
            self.ema.update(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )

    def _subs_parameterization(self, logits, xt):
        # log prob at the mask index = - infinity
        logits[..., self.mask_index] += self.neg_infinity

        # Normalize the logits such that x.exp() is
        # a probability distribution over vocab_size.
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        # Apply updates directly in the logits matrix.
        # For the logits of the unmasked tokens, set all values
        # to -infinity except for the indices corresponding to
        # the unmasked tokens.
        unmasked_indices = xt != self.mask_index
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits

    def _d3pm_parameterization(self, logits):
        if self.subs_masking:
            logits[:, :, self.mask_index] += self.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        return logits

    def _sedd_parameterization(self, logits, xt, sigma):
        esigm1_log = (
            torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1)
            .log()
            .to(logits.dtype)
        )
        # logits shape
        # (batch_size, diffusion_model_input_length, vocab_size)
        logits = logits - esigm1_log[:, None, None] - np.log(logits.shape[-1] - 1)
        # The below scatter operation sets the log score
        # for the input word to 0.
        logits = torch.scatter(
            logits, -1, xt[..., None], torch.zeros_like(logits[..., :1])
        )
        return logits

    def _process_sigma(self, sigma):
        if sigma is None:
            assert self.parameterization == "ar"
            return sigma
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    def forward(self, x, sigma):
        """Returns log score.
            x: torch.tensor (B,X,Y,Z)
        
        """
        sigma = self._process_sigma(sigma)
        with torch.cuda.amp.autocast(dtype=torch.float32):
            logits = self.backbone(x, sigma)  # (B,X,Y,Z,Vocab)

        if self.parameterization == "subs":
            return self._subs_parameterization(logits=logits, xt=x)
        elif self.parameterization == "sedd":
            raise ValueError("Did not fix shape issue")
            return self._sedd_parameterization(logits=logits, xt=x, sigma=sigma)
        elif self.parameterization == "d3pm":
            raise ValueError("Did not fix shape issue")
            return self._d3pm_parameterization(logits=logits)

        raise ValueError(
            "Need some parameterization: otherwise the model outputs aren't logits"
        )
        return logits

    def _d3pm_loss(self, model_output, xt, x0, t):
        dt = 1 / self.T

        if torch.is_tensor(t):
            t = t[:, None]
            assert t.ndim == 2
            t = t.clamp(0.0, 1.0 - 1e-4)
        alpha_t = 1 - t + torch.zeros_like(xt)
        alpha_s = 1 - (t - dt) + torch.zeros_like(xt)

        log_x_theta_at_x0 = torch.gather(model_output, -1, x0[:, :, None]).squeeze(-1)
        log_x_theta_at_m = model_output[:, :, self.mask_index]
        x_theta_at_m = log_x_theta_at_m.exp()

        term_1_coef = dt / t
        term_1_log_nr = torch.log(alpha_t * x_theta_at_m / t + 1)
        term_1_log_dr = log_x_theta_at_x0

        term_2_coef = 1 - dt / t
        term_2_log_nr = term_1_log_nr
        term_2_log_dr = torch.log(alpha_s * x_theta_at_m / (t - dt) + 1)

        L_vb_masked = term_1_coef * (term_1_log_nr - term_1_log_dr) + term_2_coef * (
            term_2_log_nr - term_2_log_dr
        )

        L_vb = L_vb_masked * (xt == self.mask_index)

        return self.T * L_vb

    def _compute_loss(self, batch, prefix):
        attention_mask = None  # TODO: can change to add attention mask
        losses = self._loss(batch, attention_mask)
        loss = losses.loss

        if prefix == "train":
            pass 
            #self.train_metrics.update(losses.nlls, losses.token_mask)
            #metrics = self.train_metrics
        elif prefix == "val":
            pass 
            #self.valid_metrics.update(losses.nlls, losses.token_mask)
            #metrics = self.valid_metrics
        elif prefix == "test":
            pass 
            #self.test_metrics.update(losses.nlls, losses.token_mask)
            #metrics = self.test_metrics
        else:
            raise ValueError(f"Invalid prefix: {prefix}")

        #self.log_dict(metrics, on_step=False, on_epoch=True, sync_dist=True) 
        self.log(f"{prefix}/nll_non_air", 
                losses.non_air_nll, 
                on_step=False,     
                on_epoch=True,      
                sync_dist=True)
        self.log(f"{prefix}/nll_air", 
        losses.air_nll, 
        on_step=False,     
        on_epoch=True,      
        sync_dist=True)    
        
        self.log(
            name=f"{prefix}/loss",
            value=loss.item(),
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )      
        
        return loss

    def on_train_epoch_start(self):
        self.backbone.train()
        self.noise.train()

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, prefix="train")
        return loss

    def on_validation_epoch_start(self):
        if self.ema:
            self.ema.store(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
            self.ema.copy_to(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
        self.backbone.eval()
        self.noise.eval()
        #assert self.valid_metrics.nll.mean_value == 0
        #assert self.valid_metrics.nll.weight == 0

    def validation_step(self, batch, batch_idx):
        print(f"Validation step {batch_idx}")  
        val_loss = self._compute_loss(batch, prefix="val")
        self.log("val_batch_loss", val_loss, on_step=True, on_epoch=False)
        return val_loss 

    def on_validation_epoch_end(self):
        if (self.current_epoch + 1) % self.val_sample_freq != 0:
            return  

        for _ in range(self.config.sampling.num_sample_batches):
            samples, _ = self._sample()
            samples = samples.detach().cpu()
            block_samples = self.tokenizer.detokenize(samples)
            for i in range(self.config.sampling.num_sample_log):
                sample = block_samples[i]
                voxel_to_plot(sample, f"sample_{i+1}_epoch{self.current_epoch}", base_dir=os.getcwd())

        if self.ema:
            self.ema.restore(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
        return
       

    def configure_optimizers(self):
        # TODO(yair): Lightning currently giving this warning when using `fp16`:
        #  "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
        #  Not clear if this is a problem or not.
        #  See: https://github.com/Lightning-AI/pytorch-lightning/issues/5558
        optimizer = torch.optim.AdamW(
            itertools.chain(self.backbone.parameters(), self.noise.parameters()),
            lr=self.config.optim.lr,
            betas=(self.config.optim.beta1, self.config.optim.beta2),
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay,
        )

        scheduler = hydra.utils.instantiate(
            self.config.lr_scheduler, optimizer=optimizer
        )
        scheduler_dict = {
            "scheduler": scheduler,
            "interval": "step",
            "monitor": "val/loss",
            "name": "trainer/lr",
        }
        return [optimizer], [scheduler_dict]

    def q_xt(self, x, move_chance):
        """Computes the noisy sample xt.

        Args:
          x: int torch.Tensor with shape (batch_size,
              diffusion_model_input_length), input.
          move_chance: float torch.Tensor with shape (batch_size, 1).
        """
        move_indices = torch.rand(*x.shape, device=x.device) < move_chance

        xt = torch.where(move_indices, self.mask_index, x)

        return xt

    def _sample_prior(self, *batch_dims):
        return self.mask_index * torch.ones(*batch_dims, dtype=torch.int64)

    def _ddpm_caching_update(self, x, t, dt, p_x0=None):
        assert self.config.noise.type == "loglinear"
        sigma_t, _ = self.noise(t)
        if t.ndim > 1:
            t = t.squeeze(-1)
        assert t.ndim == 1
        move_chance_t = t[:, None, None, None, None]
        move_chance_s = (t - dt)[:, None, None, None, None]
        assert move_chance_t.ndim == 5, move_chance_t.shape
        if p_x0 is None:
            p_x0 = self.forward(x, sigma_t).exp()

        assert move_chance_t.ndim == p_x0.ndim
        q_xs = p_x0 * (move_chance_t - move_chance_s)
        q_xs[:, :, :, :, self.mask_index] = move_chance_s[:, :, :, :, 0]
        _x = _sample_categorical(q_xs)

        copy_flag = (x != self.mask_index).to(x.dtype)
        return p_x0, copy_flag * x + (1 - copy_flag) * _x

    def _ddpm_update(self, x, t, dt):
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
            sigma_s = sigma_s.squeeze(-1)
        assert sigma_t.ndim == 1, sigma_t.shape
        assert sigma_s.ndim == 1, sigma_s.shape
        move_chance_t = 1 - torch.exp(-sigma_t)
        move_chance_s = 1 - torch.exp(-sigma_s)
        move_chance_t = move_chance_t[:, None, None]
        move_chance_s = move_chance_s[:, None, None]
        unet_conditioning = sigma_t
        log_p_x0 = self.forward(x, unet_conditioning)
        assert move_chance_t.ndim == log_p_x0.ndim
        # Technically, this isn't q_xs since there's a division
        # term that is missing. This division term doesn't affect
        # the samples.
        q_xs = log_p_x0.exp() * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]
        _x = _sample_categorical(q_xs)

        copy_flag = (x != self.mask_index).to(x.dtype)
        return copy_flag * x + (1 - copy_flag) * _x

    @torch.no_grad()
    def _sample(self, num_steps=None, eps=1e-5):
        """Generate samples from the model."""
        inter_values = []

        batch_size_per_gpu = self.config.loader.eval_batch_size
        # Lightning auto-casting is not working in this method for some reason
        if num_steps is None:
            num_steps = self.config.sampling.steps

        # generate fully masked (B, D, D, D)
        x = self._sample_prior(batch_size_per_gpu, self.config.data.voxel_side_len, self.config.data.voxel_side_len, self.config.data.voxel_side_len).to(
            self.device
        )
        timesteps = torch.linspace(1, eps, num_steps + 1, device=self.device)
        dt = (1 - eps) / num_steps
        p_x0_cache = None

        for i in range(num_steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=self.device)
            if self.sampler == "ddpm":
                x = self._ddpm_update(x, t, dt)
            elif self.sampler == "ddpm_cache":
                p_x0_cache, x_next = self._ddpm_caching_update(
                    x, t, dt, p_x0=p_x0_cache
                )
                if not torch.allclose(x_next, x) or self.time_conditioning:
                    # Disable caching
                    p_x0_cache = None
                x = x_next
            else:
                x = self._analytic_update(x, t, dt)
            inter_values.append(x)

        if self.config.sampling.noise_removal:
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=self.device)
            if self.sampler == "analytic":
                x = self._denoiser_update(x, t)
            else:
                unet_conditioning = self.noise(t)[0]
                x = self.forward(x, unet_conditioning).argmax(dim=-1)

        inter_values.append(x)
        return x, inter_values

    def restore_model_and_sample(self, num_steps, eps=1e-5):
        """Generate samples from the model."""
        # Lightning auto-casting is not working in this method for some reason
        if self.ema:
            self.ema.store(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
            self.ema.copy_to(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
        self.backbone.eval()
        self.noise.eval()
        samples, inter_values = self._sample(num_steps=num_steps, eps=eps)
        if self.ema:
            self.ema.restore(
                itertools.chain(self.backbone.parameters(), self.noise.parameters())
            )
        self.backbone.train()
        self.noise.train()
        return samples, inter_values

    def get_score(self, x, sigma):
        model_output = self.forward(x, sigma)
        if self.parameterization == "subs":
            # score(x, t) = p_t(y) / p_t(x)
            # => log score(x, t) = log p_t(y) - log p_t(x)

            # case 1: x = masked
            #   (i) y = unmasked
            #     log score(x, t) = log p_\theta(x)|_y + log k
            #     where k = exp(- sigma) / (1 - exp(- sigma))
            #   (ii) y = masked
            #     log score(x, t) = 0

            # case 2: x = unmasked
            #   (i) y != masked, y != x
            #     log score(x_i, t) = - inf
            #   (ii) y = x
            #     log score(x_i, t) = 0
            #   (iii) y = masked token
            #     log score(x_i, t) = - log k
            #     where k = exp(- sigma) / (1 - exp(- sigma))

            log_k = -torch.log(torch.expm1(sigma)).squeeze(-1)
            assert log_k.ndim == 1

            masked_score = model_output + log_k[:, None, None]
            masked_score[:, :, self.mask_index] = 0

            unmasked_score = self.neg_infinity * torch.ones_like(model_output)
            unmasked_score = torch.scatter(
                unmasked_score,
                -1,
                x[..., None],
                torch.zeros_like(unmasked_score[..., :1]),
            )
            unmasked_score[:, :, self.mask_index] = -(
                log_k[:, None] * torch.ones_like(x)
            )

            masked_indices = (x == self.mask_index).to(model_output.dtype)[:, :, None]
            model_output = masked_score * masked_indices + unmasked_score * (
                1 - masked_indices
            )
        return model_output.exp()

    def _staggered_score(self, score, dsigma):
        score = score.clone()
        extra_const = (1 - dsigma.exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., self.mask_index] += extra_const
        return score

    def _analytic_update(self, x, t, step_size):
        curr_sigma, _ = self.noise(t)
        next_sigma, _ = self.noise(t - step_size)
        dsigma = curr_sigma - next_sigma
        score = self.get_score(x, curr_sigma)
        stag_score = self._staggered_score(score, dsigma)
        probs = stag_score * self._transp_transition(x, dsigma)
        return _sample_categorical(probs)

    def _denoiser_update(self, x, t):
        sigma, _ = self.noise(t)
        score = self.get_score(x, sigma)
        stag_score = self._staggered_score(score, sigma)
        probs = stag_score * self._transp_transition(x, sigma)
        probs[..., self.mask_index] = 0
        samples = _sample_categorical(probs)
        return samples

    def _transp_transition(self, i, sigma):
        sigma = _unsqueeze(sigma, reference=i[..., None])
        edge = torch.exp(-sigma) * F.one_hot(i, num_classes=self.vocab_size)
        edge += torch.where(i == self.mask_index, 1 - torch.exp(-sigma).squeeze(-1), 0)[
            ..., None
        ]
        return edge

    def _sample_t(self, n, device):
        _eps_t = torch.rand(n, device=device)
        if self.antithetic_sampling:
            offset = torch.arange(n, device=device) / n
            _eps_t = (_eps_t / n + offset) % 1
        t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
        if self.importance_sampling:
            return self.noise.importance_sampling_transformation(t)
        return t

    def _reconstruction_loss(self, x0):
        t0 = torch.zeros(x0.shape[0], dtype=self.dtype, device=self.device)
        assert self.config.noise.type == "loglinear"
        # The above assert is for d3pm parameterization
        unet_conditioning = self.noise(t0)[0][:, None]
        model_output_t0 = self.forward(x0, unet_conditioning)
        return -torch.gather(
            input=model_output_t0, dim=-1, index=x0[:, :, None]
        ).squeeze(-1)

    def _forward_pass_diffusion(self, x0):
        t = self._sample_t(x0.shape[0], x0.device)
        if self.T > 0:
            t = (t * self.T).to(torch.int)
            t = t / self.T
            # t \in {1/T, 2/T, ..., 1}
            t += 1 / self.T

        if self.change_of_variables:
            unet_conditioning = t[:, None]
            f_T = torch.log1p(-torch.exp(-self.noise.sigma_max))
            f_0 = torch.log1p(-torch.exp(-self.noise.sigma_min))
            move_chance = torch.exp(f_0 + t * (f_T - f_0))
            move_chance = move_chance[:, None]
        else:
            sigma, dsigma = self.noise(t)
            unet_conditioning = sigma[:, None]
            move_chance = 1 - torch.exp(-sigma)

        move_chance = move_chance.view(move_chance.shape[0], *([1] * (x0.dim() - 1)))

        xt = self.q_xt(x0, move_chance)
        model_output = self.forward(xt, unet_conditioning)  # (B,X,Y,Z, vocab_len)
        utils.print_nans(model_output, "model_output")

        if self.parameterization == "sedd":
            return dsigma[:, None] * self._score_entropy(
                model_output, sigma[:, None], xt, x0
            )

        if self.T > 0:
            raise ValueError("Have not fixed code for this yet. Set T == 0")
            diffusion_loss = self._d3pm_loss(
                model_output=model_output, xt=xt, x0=x0, t=t
            )
            if self.parameterization == "d3pm":
                reconstruction_loss = self._reconstruction_loss(x0)
            elif self.parameterization == "subs":
                reconstruction_loss = 0
            return reconstruction_loss + diffusion_loss

        # SUBS parameterization, continuous time.
        log_p_theta = torch.gather(
            input=model_output, dim=-1, index=x0.unsqueeze(-1)
        ).squeeze(-1)  # (B,X,Y,Z) full of logit of correct token id

        if self.change_of_variables or self.importance_sampling:
            raise ValueError("Not verified yet")
            return log_p_theta * torch.log1p(-torch.exp(-self.noise.sigma_min))

        assert len((dsigma / torch.expm1(sigma))[:, None, None, None].shape) == len(
            log_p_theta.shape
        )

        return -log_p_theta * (dsigma / torch.expm1(sigma))[:, None, None, None]

    def _loss(self, x0, attention_mask):
        '''
        x0: (torch.tensor) Voxel map of shape (B,X,Y,Z)

        #TODO: need to refactor this so it can also handle original all tokens not just air/not air 
        '''
        loss = self._forward_pass_diffusion(x0)

        if attention_mask is not None:
            raise ValueError("Did not handle this yet")
            nlls = loss * attention_mask
            count = attention_mask.sum()

            batch_nll = nlls.sum()
            token_nll = batch_nll / count
        else:
            nlls = loss
            token_nll = torch.mean(nlls)

        # air is token_id 0; compute loss on non-air 
        non_air_mask = (x0 > 0).float()
        non_air_nlls = non_air_mask * loss 
        
        air_mask = (x0 == 0).float()
        air_nlls = air_mask * loss 

        non_air_nll = non_air_nlls.sum() / torch.clamp(non_air_mask.sum(), min=1)
        air_nll = air_nlls.sum() / torch.clamp(air_mask.sum(), min=1)

        weighted_nll = self.config.air_weight * air_nll + (1- self.config.air_weight) * non_air_nll 


        return Loss(loss=weighted_nll, nlls=nlls, token_mask=attention_mask, non_air_nll=non_air_nll, air_nll = air_nll)

    def _score_entropy(self, log_score, sigma, xt, x0):
        """Computes the SEDD loss.

        Args:
          log_score: float torch.Tensor with shape (batch_size,
              diffusion_model_input_length, vocab_size),
              log score, output of the denoising network.
          xt: int torch.Tensor with shape (batch_size,
              diffusion_model_input_length), input.
          x0: int torch.Tensor with shape (batch_size,
              diffusion_model_input_length), input.
          sigma: float torch.Tensor with shape (batch_size, 1).

        Returns:
          loss with shape (batch_size, diffusion_model_input_length)
        """
        masked_indices = xt == self.mask_index

        expsig_minus_1 = torch.expm1(sigma).expand_as(xt)
        q_ratio = 1 / expsig_minus_1[masked_indices]

        words_that_were_masked = x0[masked_indices]

        neg_term = q_ratio * torch.gather(
            log_score[masked_indices], -1, words_that_were_masked[..., None]
        ).squeeze(-1)
        score = log_score[masked_indices].exp()
        if self.mask_index == self.vocab_size - 1:
            pos_term = score[:, :-1].sum(dim=-1)
        else:
            pos_term = score[:, : self.mask_index].sum(dim=-1) + score[
                :, self.mask_index + 1 :
            ].sum(dim=-1)
        const = q_ratio * (q_ratio.log() - 1)

        entropy = torch.zeros(*xt.shape, device=xt.device)
        entropy[masked_indices] += pos_term - neg_term + const
        return entropy
