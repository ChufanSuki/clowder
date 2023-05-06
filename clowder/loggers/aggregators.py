from typing import Callable, Optional, Sequence
from clowder.loggers import base


class Dispatcher(base.Logger):
    """Writes data to multiple `Logger` objects."""

    def __init__(
        self,
        to: Sequence[base.Logger],
        serialize_fn: Optional[Callable[[base.LoggingData], str]] = None,
    ):
        """Initialize `Dispatcher` connected to several `Logger` objects."""
        self._to = to
        self._serialize_fn = serialize_fn

    def write(self, values: base.LoggingData):
        """Writes `values` to the underlying `Logger` objects."""
        if self._serialize_fn:
            values = self._serialize_fn(values)
        for logger in self._to:
            logger.write(values)

    def close(self):
        for logger in self._to:
            logger.close()
