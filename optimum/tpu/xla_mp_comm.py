import torch.multiprocessing as mp


class RootMailbox:
    def __init__(self, manager: mp.Manager):
        self.root_bell = manager.Event()
        self.root_command = manager.list()
        self.agent_ready = manager.Event()
        self.output_data = manager.list()

    def send(self, command: int, *args):
        # First wait until agent is ready to receive commands
        self.agent_ready.wait()
        self.agent_ready.clear()

        self.root_command[:] = [command, *args]
        self.root_bell.set()
        # wait again until agent is ready, meaning command has been processed
        self.agent_ready.wait()
        ret = self.output_data
        return ret


class AgentMailbox:
    def __init__(self, root_mailbox: RootMailbox):
        self.root_bell = root_mailbox.root_bell
        self.root_command = root_mailbox.root_command
        self.agent_ready = root_mailbox.agent_ready
        self.output_data = root_mailbox.output_data

    def receive(self):
        self.root_bell.wait()
        self.root_bell.clear()
        return self.root_command

    def send(self, *data):
        self.output_data[:] = [*data]

    @property
    def command_data(self):
        command = self.root_command[0]
        data = self.root_command[1:]
        return command, data
