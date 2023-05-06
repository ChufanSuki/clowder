from clowder.variable import VariableSource
from clowder.worker import Worker
import abc
from typing import Generic, TypeVar, Optional
import itertools

T = TypeVar('T')


class Saveable(abc.ABC, Generic[T]):
    """An interface for saveable objects."""

    @abc.abstractmethod
    def save(self) -> T:
        """Returns the state from the object to be saved."""

    @abc.abstractmethod
    def restore(self, state: T):
        """Given the state, restores the object."""


class Learner(VariableSource, Worker, Saveable):

    @abc.abstractmethod
    def step(self):
        ...

    def run(self, num_steps: Optional[int] = None) -> None:
        iterator = range(
            num_steps) if num_steps is not None else itertools.count()

        for _ in iterator:
            self.step()

    def get_variables(self, names):
        pass
