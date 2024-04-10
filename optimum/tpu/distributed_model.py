# ruff: noqa: E402
import torch
import os
from enum import Enum
from typing import Dict
from loguru import logger

os.environ["PJRT_DEVICE"] = "TPU"

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch.multiprocessing as mp

from optimum.tpu.modeling import AutoModelForCausalLM
from transformers import PretrainedConfig


class ModelCommand(Enum):
    LEAVE = 0
    PREFILL = 1
    DECODE = 2


class RootMailbox:
    def __init__(self, manager: mp.Manager):
        self.root_bell = manager.Event()
        self.root_command = manager.list()
        self.model_ready = manager.Event()
        self.output_data = manager.Value(torch.Tensor, torch.tensor([]))
        self.model_config = manager.Value(PretrainedConfig, None)

    @property
    def config(self):
        while True:
            config = self.model_config.get()
            if config is not None:
                return config

    def send(self, command: ModelCommand, data: Dict = None):
        # First wait until model is ready to receive commands
        self.model_ready.wait()
        self.model_ready.clear()

        self.root_command[:] = [command, data]
        self.root_bell.set()
        # wait again until model is ready, meaning command has been processed
        self.model_ready.wait()
        ret = self.output_data.get()
        return ret


class AgentMailbox:
    def __init__(self, root_mailbox: RootMailbox):
        self.root_bell = root_mailbox.root_bell
        self.root_command = root_mailbox.root_command
        self.model_ready = root_mailbox.model_ready
        self.output_data = root_mailbox.output_data
        self.model_config = root_mailbox.model_config

    def receive(self):
        self.root_bell.wait()
        self.root_bell.clear()
        return self.root_command

    def send(self, data: torch.Tensor):
        # Data needs to be moved to CPU before setting it
        self.output_data.set(data.cpu())

    @property
    def command_data(self):
        command = self.root_command[0]
        data = self.root_command[1]
        return command, data


def _mp_fn(rank, model_id, root_mailbox: RootMailbox, sample_fn: callable):
    device = xm.xla_device()
    world_size = xm.xrt_world_size()
    # create agent mailbox out of root's one
    mailbox = AgentMailbox(root_mailbox)

    logger.debug(
        f"Rank {rank} on {device} real device {xm.xla_real_devices([device])} ordinal {xm.get_ordinal()} "
        + f"world size {world_size}"
    )

    # Model loading and sharding should happen here
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model = model.eval()
    model.to(device)
    if rank == 0:
        mailbox.model_config.set(model.config)

    def get_next_token(inputs):
        # move inputs to device in a new dict to avoid conflicts
        model_inputs = {}
        for key, value in inputs.items():
            model_inputs[key] = value.to(device)
        outputs = model(**model_inputs, return_dict=False)[0]
        xm.mark_step()
        # consider adding a rendezvous here
        if rank == 0:
            logger.debug(f"Rank {rank} getting tokens")
            next_token = sample_fn(outputs)
            xm.mark_step()
            logger.debug(f"Rank {rank} sending next_tokens {next_token.shape}")
            mailbox.send(next_token)

    while True:
        if rank == 0:
            mailbox.model_ready.set()
            logger.debug(f"Rank {rank} waiting for commands")
            mailbox.receive()
        # Wait for rank 0 to receive command
        xm.rendezvous("start")

        logger.debug(f"Rank {rank} waiting for command at rendezvous")
        command, inputs = mailbox.command_data
        if command == ModelCommand.PREFILL:
            logger.debug(f"Rank {rank} PREFILL")
            get_next_token(inputs)
        elif command == ModelCommand.DECODE:
            logger.debug(f"Rank {rank} DECODE")
            get_next_token(inputs)
        elif command == ModelCommand.LEAVE:
            logger.debug(f"Rank {rank} LEAVE")
            # Set model to ready
            mailbox.model_ready.set()
            break


def model_loop_fn(*args):
    """Spawn processes in the TPUs forwarding arguments"""
    xmp.spawn(_mp_fn, args=(args), join=True, daemon=False)


class DistributedModel:
    def __init__(self, model_id: str, sample_fn: callable):
        manager = mp.Manager()
        self.mailbox = RootMailbox(manager)

        self.model_loop = mp.Process(target=model_loop_fn, args=(model_id, self.mailbox, sample_fn))
        self.model_loop.start()

    def prefill(self, **model_args):
        assert self.mailbox is not None, "DistributedModel is not initialized"
        return self.mailbox.send(ModelCommand.PREFILL, model_args)

    def decode(self, **model_args):
        assert self.mailbox is not None, "DistributedModel is not initialized"
        return self.mailbox.send(ModelCommand.PREFILL, model_args)

    def leave(self):
        if self.mailbox is None:
            return
        self.mailbox.send(ModelCommand.LEAVE)
        logger.debug("Joining...")
        self.model_loop.join()
        logger.debug("Model loop finished")
        self.mailbox = None

    @property
    def config(self):
        return self.mailbox.config

    def __del__(self):
        self.leave()
