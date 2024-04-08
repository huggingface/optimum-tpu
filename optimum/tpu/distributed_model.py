# ruff: noqa: E402
import torch
import os
from enum import Enum

os.environ["PJRT_DEVICE"] = "TPU"

import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch.multiprocessing as mp

from optimum.tpu.modeling import TpuModelForCausalLM
from typing import Dict


DEBUG = False
if os.environ.get("DEBUG", "0") == "1":
    DEBUG = True


def debug(*args):
    if DEBUG:
        print(*args)


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

    def send(self, command: ModelCommand, data: Dict = None):
        # First wait until model is ready to receive commands
        debug(f"  MM Command {command} waiting for model to be ready")
        self.model_ready.wait()
        self.model_ready.clear()

        self.root_command[:] = [command, data]
        self.root_bell.set()
        debug(f"  MM Command {command} sent")
        # wait again until model is ready, meaning command has been processed
        self.model_ready.wait()
        ret = self.output_data.get()
        debug(f"  MM Command {command} output shape {ret.shape}")
        return ret


class AgentMailbox:
    def __init__(self, root_mailbox: RootMailbox):
        self.root_bell = root_mailbox.root_bell
        self.root_command = root_mailbox.root_command
        self.model_ready = root_mailbox.model_ready
        self.output_data = root_mailbox.output_data

    def receive(self):
        self.root_bell.wait()
        self.root_bell.clear()
        return self.root_command

    def send(self, data: torch.Tensor):
        debug(f"  MM Enqueueing data {data.shape}")
        # Data needs to be moved to CPU before setting it
        self.output_data.set(data.cpu())
        debug("  MM Enqueueing data done")

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

    debug(
        f"Rank {rank} on {device} real device {xm.xla_real_devices([device])} ordinal {xm.get_ordinal()} "
        + f"world size {world_size}"
    )

    # Model loading and sharding should happen here
    model = TpuModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
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
            debug(f"Rank {rank} getting tokens")
            next_token = sample_fn(outputs)
            xm.mark_step()
            debug(f"Rank {rank} sending next_tokens {next_token.shape}")
            mailbox.send(next_token)

    while True:
        if rank == 0:
            mailbox.model_ready.set()
            debug(f"Rank {rank} waiting for commands")
            mailbox.receive()
        # Wait for rank 0 to receive command
        xm.rendezvous("start")

        debug(f"Rank {rank} waiting for command at rendezvous")
        command, inputs = mailbox.command_data
        if command == ModelCommand.PREFILL:
            debug(f"Rank {rank} PREFILL")
            get_next_token(inputs)
        elif command == ModelCommand.DECODE:
            debug(f"Rank {rank} DECODE")
            get_next_token(inputs)
        elif command == ModelCommand.LEAVE:
            debug(f"Rank {rank} LEAVE")
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
        debug("Joining...")
        self.model_loop.join()
        debug("Model loop finished")
        self.mailbox = None

    def __del__(self):
        self.leave()


def sample_greedy(logits):
    next_logits = logits[:, -1]
    next_token_id = torch.argmax(next_logits, dim=-1)[:, None].int()
    return next_token_id
