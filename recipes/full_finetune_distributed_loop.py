# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import pdb

import sys
import time
import copy
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
from warnings import warn
import math
import torch
from omegaconf import DictConfig, ListConfig

from torch import nn
from torch.distributed import destroy_process_group, init_process_group

from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import config, modules, training, utils
from torchtune.data import padded_collate_packed, padded_collate_sft
from torchtune.datasets import ConcatDataset
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.training import DummyProfiler, PROFILER_KEY
from torchtune.training.activations import apply_selective_activation_checkpointing
from torchtune.datasets import issai_text_completion_dataset,issai_instruct_dataset
import asyncio
import time
import threading
from datasets import load_dataset
from tqdm import tqdm

log = utils.get_logger("DEBUG")


class FullFinetuneRecipeDistributed(FTRecipeInterface):

    def __init__(self, cfg: DictConfig) -> None:
        if 'save_per_step' in cfg:
            self.save_per_step = cfg.save_per_step
        else:
            self.save_per_step = 1000
        # sub_data_length
        if 'sub_length' in cfg:
            self.sub_length = cfg.sub_length
        else:
            self.sub_length = 300

        if 'max_seq_len' in cfg:
            self.max_seq_len = cfg.max_seq_len
        else:
            self.max_seq_len = 2048
        self.global_total_tokens = 0
        self.train_type = cfg.train_type

        self._device = utils.get_device(device=cfg.device)
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)

        if self._dtype == torch.float16:
            raise ValueError(
                "full fp16 training is not supported with this recipe. Please use bf16 or fp32 instead."
            )

        if (
            cfg.get("fsdp_cpu_offload", False)
            and cfg.optimizer.get("fused", False)
            and not utils.torch_version_ge("2.4.0")
        ):
            raise RuntimeError(
                "Using fused optimizer on CPU is only supported in PyTorch nightly."
            )

        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)

        # _is_rank_zero is used primarily for logging. In the future, the logger
        # should directly take care of this
        world_size, rank = training.get_world_size_and_rank()
        self._is_rank_zero = rank == 0

        # Training cfg
        self._resume_from_checkpoint = cfg.resume_from_checkpoint
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._fsdp_sharding_strategy = torch.distributed.fsdp.ShardingStrategy[
            cfg.get("fsdp_sharding_strategy", "FULL_SHARD")
        ]

        # These are public properties which are updated by the checkpoint loader
        # when ``resume_from_checkpoint`` is `True` or validated in tests
        self.seed = training.set_seed(seed=cfg.seed)
        self.epochs_run = 0
        self.total_epochs = cfg.epochs
        self.max_steps_per_epoch = cfg.max_steps_per_epoch
        self.global_step = 0


        self.cfg = cfg
        self.dataset_cfg = cfg.dataset
        self.all_data = load_dataset(self.dataset_cfg.source,
                                     data_files=self.dataset_cfg.data_files,
                                     split=self.dataset_cfg.split).shuffle(seed=42)
        self.data_len = len(self.all_data)

        self._sampler, self._dataloader = None, None
        self.back_sample, self.back_dataloader = None, None

        self.batch_size=cfg.batch_size

        self.extra_cfg = cfg
        self.resource_flag = True # True origin, False back
        self.sub_data = None

        self.sub_index = 0
        
        self.pbar = None
        self.range_index = 0
        if 'start_index' in cfg:
            self.start_index = cfg.start_index
        else:
            self.start_index = 0

        self.counter = 0

    def load_checkpoint(self, cfg_checkpointer: DictConfig) -> Dict[str, Any]:
        """
        Extract the checkpoint state from file and validate. If resume_from_checkpoint
        is True, this also includes the recipe state.
        """
        self._checkpointer = config.instantiate(
            cfg_checkpointer,
            resume_from_checkpoint=self._resume_from_checkpoint,
        )
        checkpoint_dict = self._checkpointer.load_checkpoint()

        if self._resume_from_checkpoint:
            self._update_recipe_state(checkpoint_dict)
        return checkpoint_dict

    def _update_recipe_state(self, ckpt_dict: Dict[str, Any]) -> None:
        """
        Updates the recipe state from checkpoint.
        """
        try:
            self.epochs_run = ckpt_dict[training.EPOCHS_KEY]

            # on mismatch, warn the user and prevent the override
            if self.seed != ckpt_dict[training.SEED_KEY]:
                warn(
                    message=(
                        "Config value for seed does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.SEED_KEY]}"
                    )
                )
                self.seed = ckpt_dict[training.SEED_KEY]
            if self.max_steps_per_epoch != ckpt_dict[training.MAX_STEPS_KEY]:
                warn(
                    message=(
                        "Config value for max_steps_per_epoch does not match the checkpoint value, "
                        f"using the checkpoint value: {ckpt_dict[training.MAX_STEPS_KEY]}"
                    )
                )
                self.max_steps_per_epoch = ckpt_dict[training.MAX_STEPS_KEY]

            # on mismatch, warn the user but allow the override
            if self.total_epochs != ckpt_dict[training.TOTAL_EPOCHS_KEY]:
                warn(
                    message=(
                        "Config value for total_epochs does not match the checkpoint value, "
                        f"using the config value: {self.total_epochs}"
                    )
                )

        except KeyError as e:
            raise KeyError(
                "Checkpoint does not contain the required keys needed for updating recipe state. "
                "Are you sure you passed in the right recipe checkpoint?"
            ) from e

    def setup(self, cfg: DictConfig) -> None:
        """
        Setup the recipe. This includes training state (if resume_from_checkpoint is True),
        model, tokenizer, loss, optimizer, sampler, and dataloader.
        """
        if self._is_rank_zero:
            self._metric_logger = config.instantiate(cfg.metric_logger)

            # log config with parameter override
            self._metric_logger.log_config(cfg)

        checkpoint_dict = self.load_checkpoint(cfg_checkpointer=cfg.checkpointer)

        self._compile = cfg.get("compile", False)
        self._model = self._setup_model(
            cfg_model=cfg.model,
            enable_activation_checkpointing=cfg.enable_activation_checkpointing,
            custom_sharded_layers=cfg.get("custom_sharded_layers", None),
            fsdp_cpu_offload=cfg.get("fsdp_cpu_offload", False),
            reshard_after_forward=cfg.get("fsdp_reshard_after_forward", True),
            model_state_dict=checkpoint_dict[training.MODEL_KEY],
            ac_mode=cfg.get("ac_mode", None),
            ac_option=cfg.get("ac_option", None),
        )

        self._tokenizer = config.instantiate(cfg.tokenizer)

        self._optimizer, self._scheduler= self._setup_optimizer(
            cfg_optimizer=cfg.optimizer,
            opt_state_dict=checkpoint_dict[training.OPT_KEY]
            if self._resume_from_checkpoint
            else None,
        )

        # initialize loss
        self._loss_fn = config.instantiate(cfg.loss)

        if self._compile:
            training.compile_loss(self._loss_fn, verbose=self._is_rank_zero)

        if self._loss_fn.__class__.__name__ == "CEWithChunkedOutputLoss":
            # set num_output_chunks for model
            self._model.set_num_output_chunks(self._loss_fn.num_output_chunks)

        log.info("Loss is initialized.")


        # Set up profiler, returns DummyProfiler (nullcontext object with no-op `step` method)
        # if cfg is missing profiler key or if `cfg.profiler.enabled = False`
        self._profiler = self._setup_profiler(cfg.get(PROFILER_KEY, None))

        # Used to ignore labels for loss computation
        self.ignore_labels_cache = torch.full(
            (cfg.batch_size, 1), self._loss_fn.ignore_index, device=self._device
        )

    def _setup_profiler(
        self, cfg_profiler: Optional[DictConfig] = None
    ) -> Union[torch.profiler.profile, DummyProfiler]:
        """
        Parses the `profiler` section of top-level `cfg` and sets up profiler

        Args:
            cfg_profiler (Optional[DictConfig]): ``profiler`` section of the top-level ``cfg`` (the main config passed to
                `recipe.main`). Default None.

        Returns:
            profiler: Union[torch.profiler.profile, DummyProfiler] - DummyProfiler is a nullcontext with no-op methods
            for `start`, `stop`, and `step` that can be used in place of `torch.profiler.profile` if profiler is not enabled such
            that the instrumented training loop does not need to be changed profiling is disabled.

        The profiler config can be provided in configs under the `profiler` key with the following layout:

        .. code-block:: yaml
            profiler:
                enabled: bool

                #Output directory of trace artifacts
                output_dir: str

            #`torch.profiler.ProfilerActivity` types to trace
            cpu: bool
            cuda: bool

                #Trace options
                profile_memory: bool
                with_stack: bool
                record_shapes: bool
                with_flops: bool

            # `torch.profiler.schedule` options:
            # wait_steps -> wait, warmup_steps -> warmup, active_steps -> active, num_cycles -> repeat
            wait_steps: int
            warmup_steps: int
            active_steps: int
            num_cycles: int
        """
        # Missing profiler section in config, assume disabled
        if cfg_profiler is None:
            cfg_profiler = DictConfig({"enabled": False})

        # Check that component is included and set correctly
        if cfg_profiler.get("_component_", None) is None:
            cfg_profiler["_component_"] = "torchtune.training.setup_torch_profiler"
        else:
            assert (
                cfg_profiler.get("_component_")
                == "torchtune.training.setup_torch_profiler"
            ), "Only torch profiler supported currently: component must be `torchtune.training.setup_torch_profiler`"

        profiler, profiler_cfg = config.instantiate(cfg_profiler)

        if self._is_rank_zero:
            log.info(f" Profiler config after instantiation: {profiler_cfg}")

            self.profiler_profile_memory = profiler_cfg.get("profile_memory", False)
            if profiler_cfg["enabled"]:
                self.profiler_wait_steps = profiler_cfg["wait_steps"]
                self.profiler_warmup_steps = profiler_cfg["warmup_steps"]
                self.profiler_active_steps = profiler_cfg["active_steps"]

        return profiler

    def _setup_model(
        self,
        cfg_model: DictConfig,
        enable_activation_checkpointing: bool,
        custom_sharded_layers: Optional[List[str]],
        fsdp_cpu_offload: bool,
        reshard_after_forward: bool,
        model_state_dict: Dict[str, Any],
        ac_mode: Optional[str] = None,
        ac_option: Optional[int] = None,
    ) -> nn.Module:
        """
        Model initialization has some important considerations:
           a. To minimize GPU peak memory, we initialize the model on meta device with
              the right dtype
           b. All ranks calls ``load_state_dict`` without peaking CPU RAMs since
              full state dicts are loaded with ``torch.load(mmap=True)``
        """

        if self._is_rank_zero:
            log.info(
                "FSDP is enabled. Instantiating model and loading checkpoint on Rank 0 ..."
            )
            init_start = time.perf_counter()

        with training.set_default_dtype(self._dtype), torch.device("meta"):
            model = config.instantiate(cfg_model)

        # We currently have two versions of activation checkpointing in this recipe
        # for testing and BC purposes. ``enable_activation_checkpointing`` controls
        # the older version of AC and this behavior is unchanged
        # ac_mode and ac_option together control selective AC. This is only enabled
        # when these are set AND ``enable_activation_checkpointing`` is set to False
        # We'll clean this up as soon as testing of AC is complete
        if (not enable_activation_checkpointing) and (ac_mode is not None):
            apply_selective_activation_checkpointing(
                model,
                ac_mode,
                ac_option,
            )

        # original activation checkpointing (full) - flip the condition above
        if enable_activation_checkpointing and ac_mode is None:
            training.set_activation_checkpointing(
                model, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        # For FSDP sharding, we can condition on either the module or its name
        # Shard conditions should be callables taking name (relative to model root)
        # and the module itself and returning a bool on whether to shard the given module
        fsdp_shard_conditions = []

        # Shard transformer decoder layers (or AC-wrapped versions)
        # Alternatively we could condition on the module type (TransformerDecoder or CheckpointWrapper)
        # But directly using the name is more concise
        def _is_layer_fqn(s: str) -> bool:
            """
            Return True for layers.i and False for all other module names
            Covers sharding for both AC-wrapped and non-AC-wrapped modules in one shot
            """
            s_list = s.split(".")
            return len(s_list) == 2 and s_list[0] == "layers" and str.isdigit(s_list[1])

        fsdp_shard_conditions = [lambda n, m: _is_layer_fqn(n)]

        # If wrapping any layers separately, we can add another shard condition
        # A layer will be sharded if any of the fsdp_shard_conditions are met
        if custom_sharded_layers:
            fsdp_shard_conditions += [lambda n, m: n in custom_sharded_layers]

        training.shard_model(
            model=model,
            shard_conditions=fsdp_shard_conditions,
            cpu_offload=fsdp_cpu_offload,
            reshard_after_forward=reshard_after_forward,
        )

        with training.set_default_dtype(self._dtype), self._device:
            for m in model.modules():
                # RoPE is not covered in state dict
                if hasattr(m, "rope_init"):
                    m.rope_init()

        # This method will convert the full model state dict into a sharded state
        # dict and load into the model
        training.load_from_full_model_state_dict(
            model,
            model_state_dict,
            self._device,
            self._is_rank_zero,
            strict=True,
            cpu_offload=fsdp_cpu_offload,
        )

        # Ensure no params and buffers are on meta device
        training.validate_no_params_on_meta_device(model)

        if self._is_rank_zero:
            log.info(
                f"Instantiating model and loading checkpoint took {time.perf_counter() - init_start:.2f} secs"
            )
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        # synchronize before training begins
        torch.distributed.barrier()

        return model

    def _setup_optimizer(
        self, cfg_optimizer: DictConfig, opt_state_dict: Optional[Dict[str, Any]] = None
    ) -> Optimizer:
        optimizer = config.instantiate(cfg_optimizer, self._model.parameters())
        if opt_state_dict:
            training.load_from_full_optimizer_state_dict(
                optimizer,
                opt_state_dict,
                self._device,
            )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                        max_lr=cfg_optimizer.lr, 
                                                        epochs=self.cfg.epochs, 
                                                        steps_per_epoch=len(self.all_data),
                                                        pct_start=0.1, 
                                                        div_factor=2, 
                                                        final_div_factor=5)
        if self._is_rank_zero:
            log.info("Optimizer is initialized.")
        return optimizer, scheduler

    async def set_back_dataloader(self):

        world_size, rank = training.get_world_size_and_rank()
        
        if self.train_type=='base':
            ds = issai_text_completion_dataset(data=self.sub_data, tokenizer=self._tokenizer, column=self.dataset_cfg.column, max_seq_len=self.max_seq_len)
        else:
            ds = issai_instruct_dataset(data=self.sub_data, 
                                        tokenizer=self._tokenizer,
                                        template=self.dataset_cfg.template, 
                                        column_map=self.dataset_cfg.column_map, 
                                        max_seq_len=self.max_seq_len,
                                        packed=False)
            
        self.back_sample = DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=0,
        )
        self.back_dataloader = DataLoader(
            dataset=ds,
            batch_size=self.cfg.batch_size,
            sampler=self.back_sample,
            collate_fn=None
        )

        if self._is_rank_zero:
            log.info("back Dataset and Sampler are initialized.")

    async def set_sub_dataloader(self):

        world_size, rank = training.get_world_size_and_rank()
        # print(f'world_size: {world_size}, rank: {rank}')
        
        if self.train_type=='base':
            ds = issai_text_completion_dataset(data=self.sub_data, 
                                               tokenizer=self._tokenizer, 
                                               column=self.dataset_cfg.column, 
                                               max_seq_len=self.max_seq_len)
        else:
            ds = issai_instruct_dataset(data=self.sub_data, 
                                        tokenizer=self._tokenizer,
                                        template=self.dataset_cfg.template, 
                                        column_map=self.dataset_cfg.column_map, 
                                        max_seq_len=self.max_seq_len,
                                        packed=False)

        self._sampler = DistributedSampler(
            ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=0,
        )
        self._dataloader = DataLoader(
            dataset=ds,
            batch_size=self.cfg.batch_size,
            sampler=self._sampler,
            collate_fn=None
        )

        if self._is_rank_zero:
            log.info("sub Dataset and Sampler are initialized.")
            
    def start_sub_task(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        if self.resource_flag:
            loop.run_until_complete(self.set_back_dataloader())
        else:
            loop.run_until_complete(self.set_sub_dataloader())
        loop.close()
        
    def _setup_data(
        self,
        # cfg_dataset: DictConfig,
        # shuffle: bool,
        # batch_size: int,
    ) -> Tuple[DistributedSampler, DataLoader]:
        """
        All data related setup happens here. Currently this recipe only supports the
        DistributedSamplers with Map-style Datasets which fit into memory. Other samplers,
        iterable datasets and streaming datasets are not supported.
        """
        world_size, rank = training.get_world_size_and_rank()

        if self.train_type=='base':   
            ds = issai_text_completion_dataset(data=self.sub_data, 
                                               tokenizer=self._tokenizer, 
                                               column=self.dataset_cfg.column, 
                                               max_seq_len=self.max_seq_len)
        else:
            ds = issai_instruct_dataset(data=self.sub_data, 
                                        tokenizer=self._tokenizer,
                                        template=self.dataset_cfg.template, 
                                        column_map=self.dataset_cfg.column_map, 
                                        max_seq_len=self.max_seq_len,
                                        packed=False)

        self._sampler = DistributedSampler(
            ds, num_replicas=world_size, rank=rank, shuffle=True, seed=0
        )
        self._dataloader = DataLoader(
            dataset=ds,
            batch_size=self.cfg.batch_size,
            sampler=self._sampler,
            collate_fn=None
        )

        if self._is_rank_zero:
            log.info("Dataset and Sampler are initialized.")

        # return sampler, dataloader

    def save_checkpoint(
        self,
        epoch: int,
    ) -> None:
    
        # final dict passed onto the checkpointer
        checkpoint_dict = {}

        intermediate_checkpoint = epoch + 1 < self.total_epochs
        # To prevent GPU memory from spiking during checkpoint save,
        # we consolidate the full model and optim state dicts on CPU for rank 0
        cpu_state_dict = training.get_full_model_state_dict(
            self._model,
            self._is_rank_zero,
            device=self._device,
        )

        if intermediate_checkpoint:
            opt_state_dict = training.get_full_optimizer_state_dict(
                self._optimizer,
                self._is_rank_zero,
                device=self._device,
            )
        else:
            opt_state_dict = None

        # Now that we have the model and opt state dict, create the actual checkpoint dict
        # to be sent to the checkpointer and ultimately written to file
        if self._is_rank_zero:

            checkpoint_dict.update({training.MODEL_KEY: cpu_state_dict})

            # if training is in-progress, checkpoint the optimizer state and recipe state
            # as well.
            if intermediate_checkpoint:
                checkpoint_dict.update(
                    {
                        training.OPT_KEY: opt_state_dict,
                        training.SEED_KEY: self.seed,
                        training.EPOCHS_KEY: self.epochs_run,
                        training.TOTAL_EPOCHS_KEY: self.total_epochs,
                        training.MAX_STEPS_KEY: self.max_steps_per_epoch,
                    }
                )

            self._checkpointer.save_checkpoint(
                checkpoint_dict,
                epoch=epoch,
                intermediate_checkpoint=intermediate_checkpoint,
            )
    
    def sub_train(self,curr_epoch,world_size,rank):
        sub_start_steps = self.global_step

        if not self.loader_len:
            self._steps_per_epoch = (
                            len(self._dataloader) // self._gradient_accumulation_steps
                        )*self._all_range_len
            self.loader_len = len(self._dataloader)
        else:
            if not self.loader_len == len(self._dataloader):
                self._steps_per_epoch = self._steps_per_epoch -(
                            (self.loader_len - len(self._dataloader) )// self._gradient_accumulation_steps
                    )

        t0 = time.perf_counter()
        running_loss = 0
        num_tokens = 0

        if self.resource_flag:
            if not self._sampler:
                raise '_sampler not none'
            self._sampler.set_epoch(curr_epoch)
        else:
            if not self.back_sample:
                raise 'back_sample not none'
            self.back_sample.set_epoch(curr_epoch)
        if not self.pbar:
            self.pbar = tqdm(total=self._steps_per_epoch, disable=not (rank == 0))
        for idx, batch in enumerate(self._dataloader if self.resource_flag else self.back_dataloader):
            # self.global_step += 1
            if (
                    self.max_steps_per_epoch is not None
                    and (idx // self._gradient_accumulation_steps)
                    == self.max_steps_per_epoch
                ):
                    break

            # Start tracking CUDA memory for active steps for just the first epoch
            if (
                self._is_rank_zero
                and curr_epoch == 0
                and self.profiler_profile_memory
                and idx == self.profiler_wait_steps + self.profiler_warmup_steps
            ):
                torch.cuda.memory._record_memory_history()


            # Both are shape [b, s]
            tokens, labels = batch["tokens"], batch["labels"]
            # Get the attention mask and position ids from the dataset if they
            # exist. Currently, only sample packing in PackedDataset returns these
            mask = batch.get("mask", None)  # shape [b, s, s]
            input_pos = batch.get("input_pos", None)  # shape [b, s]

            tokens = tokens.to(self._device)
            token_size = tokens.numel()
            num_tokens += token_size
            self.global_total_tokens += (token_size*world_size)

            labels = labels.to(self._device)
            mask = mask.to(self._device) if mask is not None else None
            input_pos = (
                input_pos.to(self._device) if input_pos is not None else None
            )

            logits = self._model(tokens, mask=mask, input_pos=input_pos)

            # Shift labels to compute loss
            # equivalent to doing labels[..., 1:] and logits[..., :-1, :]
            # But this way we dont need to slice the logits. We just add an ignore index to labels.
            labels = torch.hstack(
                (labels[..., 1:], self.ignore_labels_cache[: labels.shape[0]])
            )
            if not isinstance(logits, list):
                labels = labels.reshape(-1)
                logits = logits.reshape(-1, logits.size(-1))

            # Compute loss
            loss = self._loss_fn(logits, labels)

            # free logits otherwise it peaks backward memory
            del logits

            loss = loss / self._gradient_accumulation_steps
            running_loss += loss


            loss.backward()

            # Step with optimizer
            if (idx + 1) % self._gradient_accumulation_steps == 0:
                self._optimizer.step()
                self._optimizer.zero_grad(set_to_none=True)

                self._scheduler.step()
                # Update the number of steps when the weights are updated
                self.global_step += 1

                loss_to_log = running_loss.item()
                self.pbar.update(1)
                self.pbar.set_description(
                    f"{curr_epoch + 1}|{self.range_index}|{self.global_step}|Loss: {loss_to_log}"
                )

                # Log per-step metrics
                if (
                    self.global_step % self._log_every_n_steps == 0
                    and self._is_rank_zero
                ):
                    time_per_step = time.perf_counter() - t0
                    # self.counter += 1
                    # print(f"\n\nCOUNTER: {self.counter}\n\n")
                    # if self.counter == 8:
                    #     pdb.set_trace()
                    log_dict = {
                        "train/loss": loss_to_log,
                        'train/perplexity': math.exp(loss_to_log),
                        "train/lr": self._optimizer.param_groups[0]["lr"],
                        "train/tokens_per_second_per_gpu": num_tokens / time_per_step,
                        "train/total_tokens": self.global_total_tokens,
                    }
                    if self._log_peak_memory_stats:
                        log_dict.update(
                            training.get_memory_stats(device=self._device)
                        )
                    self._metric_logger.log_dict(
                        log_dict,
                        step=self.global_step,
                    )

                # Reset running stats for the next step
                running_loss = 0
                num_tokens = 0
                t0 = time.perf_counter()

                # Stop tracking CUDA memory now that active steps are complete
                if (
                    self._is_rank_zero
                    and curr_epoch == 0
                    and self.profiler_profile_memory
                    and idx
                    == self.profiler_wait_steps
                    + self.profiler_warmup_steps
                    + self.profiler_active_steps
                ):
                    torch.cuda.memory._record_memory_history(enabled=None)

                # Step profiler
                # Note that this is called within gradient accumulation block, hence
                # will include multiple forward / backward passes if gradient accumulation > 1
                self._profiler.step()

            if (self.global_step +1)% 50==0:
                self.save_checkpoint(epoch=self.global_step + 1)

            pass

    def train(self) -> None:
        """
        The core training loop.
        """
        # clean up before training begins
        training.cleanup_before_training()

        world_size, rank = training.get_world_size_and_rank()

        # zero out the gradients before starting training
        self._optimizer.zero_grad()

        # Initialize tokens count and running loss (for grad accumulation)
        

        self._profiler.start()
        # self.epochs_run should be non-zero when we're resuming from a checkpoint
        for curr_epoch in range(self.epochs_run, self.total_epochs):

            _all_range = list(range(0, self.data_len, min(self.data_len, self.sub_length)))
            
            self.loader_len = None
            _all_range_len = len(_all_range)
            self._all_range_len = _all_range_len
            # if self._is_rank_zero:
            #     log.info(f"_all_range_len {_all_range_len}")

            next_range_index = 0

            for j in range(_all_range_len):

                if j< self.start_index:
                    continue
                start = time.time()
                if self.resource_flag:
                    if self._sampler and self._dataloader:
                        if j==self.sub_index and not _all_range_len<= j+1:
                            i = _all_range[self.sub_index+1]
                            sub_range = list(range(i,min(i+self.sub_length,self.data_len)))
                            self.sub_data = self.all_data.select(sub_range)
                            sub_task_thread = threading.Thread(target=self.start_sub_task)
                            sub_task_thread.start()

                        self.sub_train(curr_epoch=curr_epoch,world_size=world_size,rank=rank)
                        
                        if j==self.sub_index and not _all_range_len<= j+1:
                            sub_task_thread.join()

                        self.resource_flag=False
                    else:
                        i = _all_range[self.sub_index]
                        sub_range = list(range(i,min(i+self.sub_length,self.data_len)))
                        self.sub_data = self.all_data.select(sub_range)
                        self._setup_data()
                        
                        # next data
                        if j==self.sub_index and not _all_range_len<= j+1:
                            i = _all_range[self.sub_index+1]
                            sub_range = list(range(i,min(i+self.sub_length,self.data_len)))
                            self.sub_data = self.all_data.select(sub_range)
                            sub_task_thread = threading.Thread(target=self.start_sub_task)
                            sub_task_thread.start()

                        self.sub_train(curr_epoch=curr_epoch,world_size=world_size,rank=rank)
                        
                        if j==self.sub_index and not _all_range_len<= j+1:
                            sub_task_thread.join()

                        self.resource_flag=False
                else:
                    if self.back_sample and self.back_dataloader:
                        if j==self.sub_index and not _all_range_len<= j+1:
                            i = _all_range[self.sub_index+1]
                            sub_range = list(range(i,min(i+self.sub_length,self.data_len)))
                            self.sub_data = self.all_data.select(sub_range)
                            sub_task_thread = threading.Thread(target=self.start_sub_task)
                            sub_task_thread.start()

                        self.sub_train(curr_epoch=curr_epoch,world_size=world_size,rank=rank)

                        if j==self.sub_index and not _all_range_len< j+1:
                            sub_task_thread.join()
                        self.resource_flag=True
                    else:
                        self.resource_flag=True

                        i = _all_range[self.sub_index]
                        sub_range = list(range(i,min(i+self.sub_length,self.data_len)))
                        self.sub_data = self.all_data.select(sub_range)
                        self._setup_data()

                        if _all_range_len>self.sub_index+1:
                            i = _all_range[self.sub_index+1]
                            sub_range = list(range(i,min(i+self.sub_length,self.data_len)))
                            self.sub_data = self.all_data.select(sub_range)
                            sub_task_thread = threading.Thread(target=self.start_sub_task)
                            sub_task_thread.start()

                        self.sub_train(curr_epoch=curr_epoch,world_size=world_size,rank=rank)
                    
                        self.resource_flag=False
                
                # sub_data_duration = time.time()-start
                # print("STEP TIME:", sub_data_duration)
                
                # self.sub_index += 1

            self.epochs_run += 1
            # self.save_checkpoint(epoch=self.global_step)
            self.pbar=None

        self._profiler.stop()

    def cleanup(self) -> None:
        if self._is_rank_zero:
            self._metric_logger.close()
        destroy_process_group()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    if not training.is_distributed():
        raise RuntimeError(
            "Distributed finetune recipe should be run via a distributed launcher."
            "If using tune CLI, please specify --nnodes 1 and --nproc_per_node [num_gpus]"
        )
    init_process_group(backend="gloo" if cfg.device == "cpu" else "nccl")
    if cfg.get("fsdp_cpu_offload", False):
        # Utilize all available CPU cores for intra-op parallelism. This provides ~2x
        # speed up when benchmarking fused AdamW on CPU
        training.set_torch_num_threads()

    config.log_config(recipe_name="FullFinetuneRecipeDistributed", cfg=cfg)

    recipe = FullFinetuneRecipeDistributed(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
# torchrun --nproc-per-node=8 recipes/full_finetune_distributed_loop.py --config /data/nvme3n1p1/adal_workspace/new_era_of_3.2/torchtune/config_train/1B_full.yaml