import abc

class Worker(abc.ABC):
    """An interface for (potentially) distributed workers."""

    @abc.abstractmethod
    def run(self):
        """Runs the worker."""