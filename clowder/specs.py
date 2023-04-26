from dm_env import specs
import dm_env
from typing import Any, Union, Mapping, Sequence, Optional, Union
import numpy as np
from torch import Tensor

from dataclasses import dataclass

NestedSpec = Union[specs.Array, Mapping[Any, 'NestedSpec'], Sequence['NestedSpec']]
NestedArray = Union[np.ndarray, np.number, Mapping[Any, "NestedArray"], Sequence['NestedArray']]
NestedTensor = Union[Tensor, Mapping[Any, "NestedTensor"], Sequence['NestedTensor']]

Nest = Union[NestedArray, NestedSpec, NestedTensor]

@dataclass
class EnvironmentSpec:
    """Full specification of the domains used by a given environment."""
    observations: NestedSpec
    actions: NestedSpec
    rewards: NestedSpec
    discounts: NestedSpec
    

def make_environment_spec(environment: dm_env.Environment) -> EnvironmentSpec:
    """Returns an `EnvironmentSpec` describing values used by an environment."""
    return EnvironmentSpec(
        observations=environment.observation_spec(),
        actions=environment.action_spec(),
        rewards=environment.reward_spec(),
        discounts=environment.discount_spec())