# ruff: noqa: E402
import os
from enum import Enum
import time
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
    logger.debug(f"[Rank {rank}] Starting _mp_fn")
    device = xm.xla_device()
    world_size = xm.xrt_world_size()
    mailbox = AgentMailbox(root_mailbox)

    logger.debug(
        f"Rank {rank} on {device} real device {xm.xla_real_devices([device])} ordinal {xm.get_ordinal()} "
        + f"world size {world_size}"
    )

    logger.debug(f"[Rank {rank}] Loading model")
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model = model.eval()
    model.to(device)
    logger.debug(f"[Rank {rank}] Model loaded and moved to {device=}")

    def get_next_token(inputs):
        logger.debug(f"[Rank {rank}] Starting get_next_token")
        model_inputs = {k: v.to(device) for k, v in inputs.items()}
        logger.debug(f"[Rank {rank}] Running model inference")
        outputs = model(**model_inputs, return_dict=False)[0]
        xm.mark_step()
        if rank == 0:
            logger.debug(f"[Rank {rank}] Sampling next token")
            next_token = sample_fn(outputs)
            xm.mark_step()
            logger.debug(f"[Rank {rank}] Sending next token")
            mailbox.send(next_token.cpu())
        logger.debug(f"[Rank {rank}] Finished get_next_token")

    while True:
        if rank == 0:
            mailbox.agent_ready.set()
            logger.debug(f"[Rank {rank}] Waiting for commands")
            mailbox.receive()
        xm.rendezvous("start")

        logger.debug(f"[Rank {rank}] Received command")
        command, data = mailbox.command_data
        inputs = data[0] if data else None
        if command == ModelCommand.PREFILL:
            logger.debug(f"[Rank {rank}] Executing PREFILL")
            get_next_token(inputs)
        elif command == ModelCommand.DECODE:
            logger.debug(f"[Rank {rank}] Executing DECODE")
            get_next_token(inputs)
        elif command == ModelCommand.LEAVE:
            logger.debug(f"[Rank {rank}] Executing LEAVE")
            mailbox.agent_ready.set()
            break
    logger.debug(f"[Rank {rank}] Exiting _mp_fn")


def model_loop_fn(*args):
    """Spawn processes in the TPUs forwarding arguments"""
    xmp.spawn(_mp_fn, args=(args), join=True, daemon=False)


class DistributedModel:
    def __init__(self, model_id: str, sample_fn: callable):
        logger.debug(f"Initializing DistributedModel with model_id: {model_id}")
        start_time = time.time()
        manager = mp.Manager()
        self.mailbox = RootMailbox(manager)

        self.model_loop = mp.Process(target=model_loop_fn, args=(model_id, self.mailbox, sample_fn))
        self.model_loop.start()
        logger.debug(f"DistributedModel initialization completed in {time.time() - start_time:.2f} seconds")

    def prefill(self, **model_args):
        logger.debug("Starting prefill operation")
        start_time = time.time()
        assert self.mailbox is not None, "DistributedModel is not initialized"
        result = self.mailbox.send(ModelCommand.PREFILL, model_args)[0]
        logger.debug(f"Prefill operation completed in {time.time() - start_time:.2f} seconds")
        return result

    def decode(self, **model_args):
        logger.debug("Starting decode operation")
        start_time = time.time()
        assert self.mailbox is not None, "DistributedModel is not initialized"
        result = self.mailbox.send(ModelCommand.PREFILL, model_args)[0]
        logger.debug(f"Decode operation completed in {time.time() - start_time:.2f} seconds")
        return result

    def leave(self):
        if self.mailbox is None:
            logger.debug("DistributedModel already left")
            return
        logger.debug("Initiating leave operation")
        start_time = time.time()
        self.mailbox.send(ModelCommand.LEAVE)
        logger.debug("Joining model loop...")
        self.model_loop.join()
        logger.debug(f"Model loop finished in {time.time() - start_time:.2f} seconds")
        self.mailbox = None

    def __del__(self):
        logger.debug("DistributedModel destructor called")
        self.leave()
