import abc

class Scheduler(abc.ABC):
    def __init__(self) -> None:
        self.name = None
        self.size = None
        self.n_samples = None
        self.description = None

    @abc.abstractmethod
    def get_lr_for_iter(self, iteration):
        pass
    def _get_description(self):
        description = {'name':self.name,}
        return description