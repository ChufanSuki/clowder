from typing import Sequence
from clowder.variable import VariableClient, VariableSource
from clowder.specs import NestedArray
from typing import Optional
import pytest

class MockVariableSource(VariableSource):
    def __init__(self, variables: Optional[NestedArray] = None) -> None:
        super().__init__()
        self._variables = variables
    
    def get_variables(self, names: Sequence[str]):
        return (names)
