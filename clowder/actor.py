import abc
from typing import Optional
from clowder.variable import VariableClient
from clowder import adders
from clowder import specs
from clowder.utils import batch
import tensorflow_probability as tfp
import dm_env
import tree
tfd = tfp.distributions
class Actor(abc.ABC):
    def select_action(self, observation):
        pass

    def observe(self, action, next_timestep):
        pass

    def observe_first(self, timestep):
        pass

    def update(self):
        pass

class FeedForwardActor(Actor):
    """A feed-forward actor.

    An actor based on a feed-forward policy which takes non-batched observations
    and outputs non-batched actions. It also allows adding experiences to replay
    and updating the weights from the policy on the learner.
    """

    def __init__(
        self,
        policy_network,
        adder: Optional[adders.Adder] = None,
        variable_client: Optional[VariableClient] = None,
    ):
        """Initializes the actor.

            Args:
            policy_network: the policy to run.
            adder: the adder object to which allows to add experiences to a
                dataset/replay buffer.
            variable_client: object which allows to copy weights from the learner copy
                of the policy to the actor copy (in case they are separate).
        """

        # Store these for later use.
        self._adder = adder
        self._variable_client = variable_client
        self._policy_network = policy_network

    def _policy(self, observation: specs.NestedArray) -> specs.NestedArray:
        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        batched_observation = batch.add_batch_dim(observation)
        batched_observation = specs.nestedarray_to_nestedtensor(batched_observation)

        # Compute the policy, conditioned on the observation.
        policy = self._policy_network(batched_observation)

        # Sample from the policy if it is stochastic.
        action = policy.sample() if isinstance(policy, tfd.Distribution) else policy
        action = specs.nestedtensor_to_nestedarray(action)
        return action

    def select_action(self, observation: specs.NestedArray) -> specs.NestedArray:
        # Pass the observation through the policy network.
        action = self._policy(observation)

        # Return a numpy array with squeezed out batch dimension.
        return batch.to_numpy_squeeze(action)

    def observe_first(self, timestep: dm_env.TimeStep):
        if self._adder:
            self._adder.add_first(timestep)

    def observe(self, action: specs.NestedArray, next_timestep: dm_env.TimeStep):
        if self._adder:
            self._adder.add(action, next_timestep)

    def update(self, wait: bool = False):
        if self._variable_client:
            self._variable_client.update(wait)