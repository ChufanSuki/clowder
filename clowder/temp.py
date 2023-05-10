import dm_env
import tree
from dm_env import specs
import numpy as np
from clowder.gym_wrapper import GymWrapper, _convert_to_spec, GymAtariAdapter
import gymnasium as gym
import ale_py

import numpy as np

def flatten_numpy(tensor):
    return np.reshape(tensor, (tensor.shape[0], -1))

# Usage example
input_tensor = np.random.rand(2, 3, 4)
flattened_tensor = flatten_numpy(input_tensor)
print(flattened_tensor)
