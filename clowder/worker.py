import abc
from rich.console import Console
from clowder.remote import Launchable
from typing import Any, Callable, List, NoReturn, Optional, Sequence, Union
from multiprocessing import Process

console = Console()

class Handle(abc.ABC):
  """Represents an interface of the service a.k.a remote functions of the worker.

  Call `.dereference()` to get the actual worker object of this service (to be
  implemented in subclasses).
  """

  def connect(self, worker: 'Worker', label: str) -> None:
    """Called to let this handle know about it's connecting to a worker.

    This is supposed to be called:

      1. Before creating any executables
      2. Before any address binding happens

    The motivation is we want to give the handle a chance to configure itself
    for the worker, before it's turned into executables and addresses are
    finalized.

    Args:
      worker: The worker that the handle connects to.
      label: Label of the worker.
    """
    pass

  def transform(self, executables: Sequence[Any]) -> Sequence[Any]:
    """Transforms the executables that make use of this handle."""
    return executables

class Worker(Launchable):
    """An interface for (potentially) distributed workers."""

    def __init__(self, name: str, addr: str, timeout: float = 60) -> None:
        self._name = name
        self._addr = addr
        self._timeout = timeout
        self._handle = None
        self._process = None

        self._handles = []

    def __repr__(self):
        return f'Worker(name={self._name} addr={self._addr})'

    @property
    def name(self) -> str:
        return self._name

    @property
    def addr(self) -> str:
        return self._addr

    @property
    def timeout(self) -> float:
        return self._timeout

    def add_handle(self, handles: Union[Handle,
                                         Sequence[Handle]]) -> None:
        if isinstance(handles, (list, tuple)):
            self._handles.extend(handles)
        else:
            self._handles.append(handles)

    def start(self) -> None:
        self.init_launching()
        self._process = Process(target=self.run)
        self._process.start()

    def join(self) -> None:
        self._process.join()

    def terminate(self) -> None:
        if self._process is not None:
            self._process.terminate()

    def run(self):
        pass

    def init_launching(self) -> None:
        pass

    def init_execution(self) -> None:
        pass

class WorkerList:

    def __init__(self, workers: Optional[Sequence[Worker]] = None) -> None:
        self._workers = []
        if workers is not None:
            self._workers.extend(workers)

    def __getitem__(self, index: int) -> Worker:
        return self._workers[index]

    @property
    def workers(self) -> List[Worker]:
        return self._workers

    def append(self, worker: Worker) -> None:
        self.workers.append(worker)

    def extend(self, workers: Union['WorkerList', Sequence[Worker]]) -> None:
        if isinstance(workers, WorkerList):
            self.workers.extend(workers.workers)
        else:
            self.workers.extend(workers)

    def start(self) -> None:
        for worker in self.workers:
            worker.start()

    def join(self) -> None:
        for worker in self.workers:
            worker.join()

    def terminate(self) -> None:
        for worker in self.workers:
            worker.terminate()

WorkerLike = Union[Worker, WorkerList]
