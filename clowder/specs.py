from dm_env import specs
import dm_env
from typing import Any, Union, Mapping, Sequence

from dataclasses import dataclass

NestedSpec = Union[specs.Array, Mapping[Any, 'NestedSpec'], Sequence['NestedSpec']]

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