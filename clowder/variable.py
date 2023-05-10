from typing import Sequence, Union, List, Any
import datetime
import time
import abc
from clowder import specs


class VariableSource(abc.ABC):

    def get_variables(self, names: Sequence[str]):
        pass


class VariableClient(abc.ABC):

    def __init__(
        self,
        client: VariableSource,
        key: Union[str, Sequence[str]],
        update_period: Union[int, datetime.timedelta] = 1,
    ):
        self._client = client
        self._call_counter = 0
        self._last_call = time.time()
        self._params: Sequence[specs.Nest] = None
        if isinstance(key, str):
            key = [key]
        self._key = key
        self._update_period = update_period

        self._request = lambda k=key: self._client.get_variables(k)

    def update(self, wait: bool = False):
        self._call_counter += 1
        
        if isinstance(self._update_period, datetime.timedelta):
            if self._last_call + self._update_period.total_seconds() > time.time():
                return
        else:
            if self._call_counter < self._update_period:
                return
        
        if wait:
            self._call_counter = 0
            self._last_call = time.time()
            self.update_and_wait()
            return

    def update_and_wait(self):
        """Immediately update and block until we get the result."""
        self._callback(self._request())

    def _callback(self, params_list: Sequence[specs.Nest]):
        self._params = params_list
    
    @property
    def params(self) -> Union[specs.Nest, Sequence[specs.Nest]]:
        """Returns the first params for one key, otherwise the whole params list."""
        if self._params is None:
            self.update_and_wait()

        if len(self._params) == 1:
            return self._params[0]
        else:
            return self._params

    def poll(self):
        pass
