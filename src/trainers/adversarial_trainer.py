"""AdversarialTrainer: Configurable dual-optimizer training loop with callback integration.

This module implements a flexible and lightweight `AdversarialTrainer` class
for training adversarial models such as GANs or diffusion–discriminator hybrids.
It extends the base `Trainer` pattern to handle *two optimizers* and *two model
components* (e.g., generator and discriminator), while preserving the same
callback and logging structure.

The design emphasizes:
    • Simplicity — minimal boilerplate, research-friendly implementation.
    • Transparency — callbacks receive lifecycle events at key stages.
    • Flexibility — supports gradient accumulation, checkpoint resume,
      and arbitrary generator/discriminator configurations.

Typical usage example
---------------------
>>> trainer = AdversarialTrainer(
...     model=gan_model,
...     optim_g=optimizer_g,
...     optim_d=optimizer_d,
...     train_config=cfg.trainer.train_config,
...     data_wrapper=wrapper,
...     output_dir=cfg.trainer.output_dir,
...     scheduler_g=scheduler_g,
...     scheduler_d=scheduler_d,
...     callbacks=callbacks,
... )
>>> trainer.train(train_loader, validation_loader)

The model must implement the following:
    • `train_step(batch_dict, optim_g, optim_d)` ->
          returns a dictionary containing:
          {
              "loss_g": ...,
              "loss_d": ...,
              "loss": ...   # optional combined scalar
          }

Callbacks can extend training behavior by overriding any of the following hooks:
    - on_train_begin(self, trainer)
    - on_epoch_begin(self, trainer)
    - on_batch_end(self, trainer)
    - on_validation_end(self, trainer)
    - on_epoch_end(self, trainer)
    - on_train_end(self, trainer)
    - (optional) resume(self, trainer)

All logging goes through the standard `logging` module, which Hydra automatically
redirects into timestamped log directories (e.g. `outputs/YYYY-MM-DD/HH-MM-SS/`).
"""

import os
import torch
import logging
from torch.utils.data import DataLoader
from typing import Optional, List
from tqdm import tqdm

log = logging.getLogger(__name__)


class AdversarialTrainer:
    """Encapsulates a complete training and validation loop for adversarial models.

    The `AdversarialTrainer` coordinates the joint optimization of a generator
    and discriminator (or other dual-model systems), handling forward/backward
    passes, scheduler stepping, and callback signaling. It mirrors the standard
    `Trainer` API for compatibility and code reuse.

    Parameters
    ----------
    model : torch.nn.Module
        Adversarial model implementing `train_step(batch_dict, optim_g, optim_d)`.

    optim_g : torch.optim.Optimizer
        Optimizer for the generator (or first submodule).

    optim_d : torch.optim.Optimizer
        Optimizer for the discriminator (or second submodule).

    train_config : omegaconf.DictConfig
        Training configuration with at least:
            - `n_epoch`: number of epochs
            - `batch_size`: batch size
            - `device`: e.g. "cuda" or "cpu"
            - `clip_grad_norm`: optional gradient clipping threshold
            - `eval_patience`: validation frequency in epochs
            - `start_epoch`: starting epoch index
            - (optional) `accumulate_grad_batches`: gradient accumulation steps

    data_wrapper : callable
        Maps raw dataset batches to standardized input dicts:
        { "image", "segmentation", "mask", "label" }.

    output_dir : str
        Directory for logs, checkpoints, and results.

    scheduler_g : torch.optim.lr_scheduler._LRScheduler, optional
        Learning rate scheduler for the generator.

    scheduler_d : torch.optim.lr_scheduler._LRScheduler, optional
        Learning rate scheduler for the discriminator.

    ema : torch.nn.Module, optional
        Exponential moving average tracker for the generator.

    callbacks : list of Callback, optional
        List of callback instances for checkpointing, logging, etc.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optim_g: torch.optim.Optimizer,
        optim_d: torch.optim.Optimizer,
        train_config,
        data_wrapper,
        output_dir: str,
        scheduler_g: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scheduler_d: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        ema: Optional[torch.nn.Module] = None,
        callbacks: Optional[List] = None,
    ):
        """Initialize trainer state and runtime configuration."""
        self._init_config(train_config)

        self.model = model.to(self.device)
        self.opt_gen = optim_g
        self.opt_disc = optim_d
        self.scheduler_g = scheduler_g
        self.scheduler_d = scheduler_d
        self.ema = ema
        self.output_dir = output_dir

        self.data_wrapper = data_wrapper
        self.callbacks = callbacks if callbacks is not None else []

        self._stop_training = False

    def _init_config(self, train_config):
        """Extract relevant attributes from the Hydra configuration object."""
        self.n_epoch = train_config.n_epoch
        self.batch_size = train_config.batch_size
        self.device = train_config.device
        self.clip_grad_norm = train_config.clip_grad_norm
        self.eval_patience = train_config.eval_patience
        self.current_epoch = train_config.start_epoch
        self.accumulate_grad_batches = getattr(
            train_config, "accumulate_grad_batches", 1
        )

        # Adversarial control parameters (default safe values)
        self.disc_steps_per_gen = getattr(train_config, "disc_steps_per_gen", 1)
        self.gen_steps_per_disc = getattr(train_config, "gen_steps_per_disc", 1)
        self.ema_update_interval = getattr(train_config, "ema_update_interval", 1)
    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    def train(
        self, train_loader: DataLoader, validation_loader: Optional[DataLoader] = None
    ):
        """Execute the main adversarial training loop with configurable patience."""
        if self.ema is not None:
            # Register only the generator for EMA tracking
            self.ema.register(self.model.generator)

        self.model.train()

        # Allow callbacks to restore checkpointed state if applicable
        for cb in self.callbacks:
            if hasattr(cb, "resume"):
                cb.resume(self)

        for cb in self.callbacks:
            cb.on_train_begin(self)

        # Training loop over epochs
        for epoch in range(self.current_epoch, self.n_epoch):
            self.current_epoch = epoch
            self._stop_training = False

            for cb in self.callbacks:
                cb.on_epoch_begin(self)

            epoch_loss_g = 0.0
            epoch_loss_d = 0.0
            self.model.train()

            # patience counters
            d_counter = 0
            g_counter = 0

            for batch_idx, batch in enumerate(
                tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.n_epoch}")
            ):
                self.current_batch = self.data_wrapper(batch)
                loss_g, loss_d = None, None  # reset each iteration

                for _ in range(self.disc_steps_per_gen):
                    self.opt_disc.zero_grad(set_to_none=True)
                    loss_d = self.model.train_step_discriminator(self.current_batch)
                    (loss_d / self.accumulate_grad_batches).backward()
                    if self.clip_grad_norm and self.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.discriminator.parameters(), self.clip_grad_norm
                        )
                    self.opt_disc.step()
                    epoch_loss_d += loss_d.item()

                # -------------------------------
                # Train generator M times
                # -------------------------------
                for _ in range(self.gen_steps_per_disc):
                    self.opt_gen.zero_grad(set_to_none=True)
                    loss_g = self.model.train_step_generator(self.current_batch)
                    (loss_g / self.accumulate_grad_batches).backward()

                    if self.clip_grad_norm and self.clip_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.generator.parameters(), self.clip_grad_norm
                        )
                    self.opt_gen.step()
                    epoch_loss_g += loss_g.item()

                    if self.ema is not None:
                        self.ema.update(self.model.generator)

                # ------------------------------------------------------------------
                # Callback update
                # ------------------------------------------------------------------
                self.loss_batch = {
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "loss_g": float(loss_g.item()) if loss_g is not None else None,
                    "loss_d": float(loss_d.item()) if loss_d is not None else None,
                }

                for cb in self.callbacks:
                    cb.on_batch_end(self)

            # ------------------------------------------------------------------
            # Epoch summary
            # ------------------------------------------------------------------
            self.last_train_loss_g = epoch_loss_g / max(1, len(train_loader))
            self.last_train_loss_d = epoch_loss_d / max(1, len(train_loader))
            log.info(
                f"[AdvTrainer] Epoch {epoch}: "
                f"loss_g={self.last_train_loss_g:.6f}, "
                f"loss_d={self.last_train_loss_d:.6f}"
            )

            # ------------------------------------------------------------------
            # Validation
            # ------------------------------------------------------------------
            if epoch % self.eval_patience == 0 and validation_loader is not None:
                self.val_loss = self.eval(validation_loader)
            else:
                self.val_loss = None

            for cb in self.callbacks:
                cb.on_validation_end(self)

            # ------------------------------------------------------------------
            # Scheduler step
            # ------------------------------------------------------------------
            if self.scheduler_g is not None:
                self.scheduler_g.step()
            if self.scheduler_d is not None:
                self.scheduler_d.step()

            for cb in self.callbacks:
                cb.on_epoch_end(self)

            # ------------------------------------------------------------------
            # Early stopping
            # ------------------------------------------------------------------
            if self._stop_training:
                log.info("[AdvTrainer] Early stopping triggered.")
                break

        for cb in self.callbacks:
            cb.on_train_end(self)

        # -------------------------------------------------------------------------

    # Validation
    # -------------------------------------------------------------------------
    def eval(self, validation_loader: DataLoader) -> float:
        """Evaluate generator quality or reconstruction loss over validation data.

        The generator is evaluated in isolation (no discriminator updates).
        If the model implements `eval_step()`, it will be used; otherwise,
        we fallback to `train_step_generator()`.
        """
        self.model.eval()
        total_loss = 0.0
        n_batches = len(validation_loader)

        with torch.no_grad():
            for batch in tqdm(
                validation_loader,
                desc=f"Validation: Epoch {self.current_epoch + 1}/{self.n_epoch}",
            ):
                dict_ord = self.data_wrapper(batch)

                # Prefer eval_step() if defined (e.g. for reconstruction / FID evaluation)
                if hasattr(self.model, "eval_step"):
                    loss = self.model.eval_step(dict_ord)
                else:
                    # Fallback: use generator loss (adversarial signal)
                    loss = self.model.train_step_generator(dict_ord)

                if loss is not None and torch.is_tensor(loss):
                    total_loss += loss.item()

        mean_loss = total_loss / max(1, n_batches)
        log.info(f"[AdvTrainer] Validation loss: {mean_loss:.6f}")

        # Return to training mode
        self.model.train()
        return mean_loss
