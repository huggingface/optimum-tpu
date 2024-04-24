# ruff: noqa: E402
from typing import Dict

import torch
import torch.multiprocessing as mp


class RootMailbox:
    def __init__(self, manager: mp.Manager):
        self.root_bell = manager.Event()
        self.root_command = manager.list()
        self.model_ready = manager.Event()
        self.output_data = manager.Value(torch.Tensor, torch.tensor([]))

    def send(self, command: int, data: Dict = None):
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
