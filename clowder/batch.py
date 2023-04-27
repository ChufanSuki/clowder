from typing import Optional, Union, Any, Sequence, Mapping, Dict
import numpy as np
import numpy.typing as npt
import tree

class Batch:
    def __init__(self, structure: Optional[Union[Dict[Any, "Batch"], np.ndarray, Sequence['Batch']]] = None):
        self.structure = structure
        self.to_dict()

    def to_dict(self):
        if isinstance(self.structure, dict):
            for k, v in self.structure.items():
                if isinstance(v, np.ndarray):
                    self.__dict__[k] = v
                else:
                    self.__dict__[k] = Batch(v)
        elif isinstance(self.structure, Sequence):
            arrays = {}
            for batch in self.structure:
                assert isinstance(batch, Batch)
                for key, array in batch.__dict__.items():
                    if key in arrays:
                        arrays[key].append(array)
                    else:
                        arrays[key] = [array]
            for key, array_list in arrays.items():
                # array_list is a list of Batch objects
                self.__dict__[key] =  tree.map_structure(lambda *args: np.stack(args), *array_list)
        elif isinstance(self.structure, np.ndarray):
            return self.structure
        else:
            return None