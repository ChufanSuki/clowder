import abc
from typing import Iterator
class PrefetchingIterator(Iterator[T], abc.ABC):
    @abc.abstractmethod
    def ready(self) -> bool:

    @abc.abstractmethod
    def retrieved_elements(self) -> int: