from typing import Sequence

class VariableSource:
    def get_variables(self, names: Sequence[str]):
        pass

class VariableClient:
    def poll(self):
        pass