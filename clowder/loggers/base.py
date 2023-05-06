"""Base logger."""

import abc
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Union
from datetime import datetime
import numpy as np
import tree

from loggers.aggregators import Dispatcher

LoggingData = Mapping[str, Any]


class Logger(abc.ABC):
    """A logger has a `write` method."""

    @abc.abstractmethod
    def write(self, data: LoggingData) -> None:
        """Writes `data` to destination (file, terminal, database, etc)."""

    @abc.abstractmethod
    def close(self) -> None:
        """Closes the logger, not expecting any further write."""


class LoggerLabel(Enum):
    Learner = 1
    Actor = 2
    Evaluator = 3


TaskInstance = int
LoggerStepsKey = str


class LoggerFactory(Protocol):

    def __call__(self,
                 label: LoggerLabel,
                 steps_key: Optional[LoggerStepsKey] = None,
                 instance: Optional[TaskInstance] = None) -> Logger:
        ...


class NoOpLogger(Logger):
    """Simple Logger which does nothing and outputs no logs.

  This should be used sparingly, but it can prove useful if we want to quiet an
  individual component and have it produce no logging whatsoever.
  """

    def write(self, data: LoggingData):
        pass

    def close(self):
        pass


class ClowderLogger(Logger):

    def __init__(
        self,
        label: LoggerLabel,
        directory: Union[Path, str],
        to_terminal: bool = True,
        to_csv: bool = False,
        to_tensorboard: bool = False,
        to_json: bool = False,
        time_delta: float = 1.0,
        print_fn: Callable[[str], None] = print,
        time_stamp: Optional[str] = None,
        extra_logger_kwargs: Dict = {},
    ) -> None:
        self._label = label
        is_s3_url = isinstance(directory, str) and directory.startswith('https://s3')
        if not is_s3_url and not isinstance(directory, Path):
            directory = Path(directory)
        self._directory = directory
        self._time_stamp = time_stamp if time_stamp else datetime.now().strftime('%Y%m%d-%H%M%S')
    
    def make_logger(self,
        to_terminal: bool,
        to_csv: bool,
        to_tensorboard: bool,
        to_json: bool,
        time_delta: float,
        print_fn: Callable[[str], None],
        extra_logger_kwargs: Dict,) -> Logger:
        logger = []
        if logger:
            logger = Dispatcher(logger)
        else:
            logger = NoOpLogger()
        return logger


def tensor_to_numpy(value: Any):
    if hasattr(value, 'numpy'):
        return value.numpy()  # tf.Tensor (TF2).
    if hasattr(value, 'device_buffer'):
        return np.asarray(value)  # jnp.DeviceArray.
    return value


def to_numpy(values: Any):
    """Converts tensors in a nested structure to numpy.

  Converts tensors from TensorFlow to Numpy if needed without importing TF
  dependency.

  Args:
    values: nested structure with numpy and / or TF tensors.

  Returns:
    Same nested structure as values, but with numpy tensors.
  """
    return tree.map_structure(tensor_to_numpy, values)
