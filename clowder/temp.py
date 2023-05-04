import dm_env
import tree
from dm_env import specs
import numpy as np
from clowder.gym_wrapper import GymWrapper, _convert_to_spec, GymAtariAdapter
import gymnasium as gym
import ale_py

def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
    return np.zeros(spec.shape, spec.dtype)

env = gym.make('ALE/Pong-v5', full_action_space=True)
env = GymAtariAdapter(env)
episode_return = tree.map_structure(_generate_zeros_from_spec,
                                        env.reward_spec())
print(env.reward_spec())
print(episode_return)