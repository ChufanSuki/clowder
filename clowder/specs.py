from dm_env import specs
import dm_env
from typing import Any, Union, Mapping, Sequence, Optional, Union
import numpy as np
import torch 
import tensorflow as tf
import tree
from dataclasses import dataclass

Tensor = Union[tf.Tensor, torch.Tensor]
NestedSpec = Union[specs.Array, Mapping[Any, 'NestedSpec'],
                   Sequence['NestedSpec']]
NestedArray = Union[np.ndarray, np.number, Mapping[Any, "NestedArray"],
                    Sequence['NestedArray']]
NestedTensor = Union[Tensor, Mapping[Any, "NestedTensor"],
                     Sequence['NestedTensor']]

Nest = Union[NestedArray, NestedSpec, NestedTensor]

def nestedarray_to_nestedtensor(nestedarray: NestedArray) -> NestedTensor:
    return tree.map_structure(torch.from_numpy, nestedarray)

def nestedtensor_to_nestedarray(nestedtensor: NestedTensor) -> NestedArray:
    return tree.map_structure(np.asarray, nestedtensor)

@dataclass
class EnvironmentSpec:
    """Full specification of the domains used by a given environment."""
    observations: NestedSpec
    actions: NestedSpec
    rewards: NestedSpec
    discounts: NestedSpec


def make_environment_spec(environment: dm_env.Environment) -> EnvironmentSpec:
    """Returns an `EnvironmentSpec` describing values used by an environment."""
    return EnvironmentSpec(observations=environment.observation_spec(),
                           actions=environment.action_spec(),
                           rewards=environment.reward_spec(),
                           discounts=environment.discount_spec())


@dataclass
class Transition:
    """Container for a transition."""
    observation: NestedArray
    action: NestedArray
    reward: NestedArray
    discount: NestedArray
    next_observation: NestedArray
    extras: NestedArray = ()
