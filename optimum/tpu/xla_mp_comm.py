from multiprocessing.managers import ListProxy
from typing import List

import torch.multiprocessing as mp


class RootMailbox:
    """A simple multiprocessing mailbox to communicate between the root process and the agents."""
    def __init__(self, manager: mp.Manager):
        self.root_bell = manager.Event()
        self.root_command = manager.list()
        self.agent_ready = manager.Event()
        self.output_data = manager.list()
        self.agent_error = manager.Event()
        self.agent_error.clear()

    def send(self, command: int, *args) -> ListProxy:
        """Send a command and arguments to the agents and wait for the response.

        Args:
            command (int): Command to send to the agents.
            *args: Arguments to send to the agents.

        Returns:
            A list containing the response from the agents.
        """
        # First wait until agent is ready to receive commands
        self.agent_ready.wait()
        self.agent_ready.clear()

        self.root_command[:] = [command, *args]
        self.root_bell.set()
        # wait again until agent is ready, meaning command has been processed
        self.agent_ready.wait()
        if self.agent_error.is_set():
            raise RuntimeError("Error on one of threads, stopping.")
        ret = self.output_data
        return ret


class AgentMailbox:
    """The agent mailbox to communicate with the root process."""
    def __init__(self, root_mailbox: RootMailbox):
        self.root_bell = root_mailbox.root_bell
        self.root_command = root_mailbox.root_command
        self.agent_ready = root_mailbox.agent_ready
        self.output_data = root_mailbox.output_data
        self.agent_error = root_mailbox.agent_error

    def receive(self) -> ListProxy:
        """Wait for a command from the root process and return it.

        Returns:
            A list containing the command and arguments from the root process.
        """
        self.root_bell.wait()
        self.root_bell.clear()
        return self.root_command

    def send(self, *data):
        """Send the response to the root process.

        Args:
            *data: Data to send to the root process.
        """
        self.output_data[:] = [*data]

    @property
    def command_data(self) -> tuple[int, List]:
        """Property helper to split command and arguments sent by the root process.

        Returns:
            A tuple containing the command and arguments.
        """
        command = self.root_command[0]
        data = self.root_command[1:]
        return command, data
