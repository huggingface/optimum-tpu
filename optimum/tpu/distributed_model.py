# ruff: noqa: E402
import os
from enum import Enum

from loguru import logger


os.environ["PJRT_DEVICE"] = "TPU"

import torch.multiprocessing as mp
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from optimum.tpu.modeling import AutoModelForCausalLM

from .xla_mp_comm import AgentMailbox, RootMailbox


class ModelCommand(Enum):
    LEAVE = 0
    PREFILL = 1
    DECODE = 2


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
            # Data needs to be moved to CPU before setting it
            mailbox.send(next_token.cpu())

    while True:
        if rank == 0:
            mailbox.agent_ready.set()
            logger.debug(f"Rank {rank} waiting for commands")
            mailbox.receive()
        # Wait for rank 0 to receive command
        xm.rendezvous("start")

        logger.debug(f"Rank {rank} waiting for command at rendezvous")
        command, data = mailbox.command_data
        inputs = data[0] if data else None
        if command == ModelCommand.PREFILL:
            logger.debug(f"Rank {rank} PREFILL")
            get_next_token(inputs)
        elif command == ModelCommand.DECODE:
            logger.debug(f"Rank {rank} DECODE")
            get_next_token(inputs)
        elif command == ModelCommand.LEAVE:
            logger.debug(f"Rank {rank} LEAVE")
            # Set model to ready
            mailbox.agent_ready.set()
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
        return self.mailbox.send(ModelCommand.PREFILL, model_args)[0]

    def decode(self, **model_args):
        assert self.mailbox is not None, "DistributedModel is not initialized"
        return self.mailbox.send(ModelCommand.PREFILL, model_args)[0]

    def leave(self):
        if self.mailbox is None:
            return
        self.mailbox.send(ModelCommand.LEAVE)
        logger.debug("Joining...")
        self.model_loop.join()
        logger.debug("Model loop finished")
        self.mailbox = None

    def __del__(self):
        self.leave()
