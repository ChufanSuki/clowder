"""Tests for the environment loop."""

from typing import Optional

from clowder import environment_loop
from clowder import specs
from tests import mocks
import numpy as np
from dm_env.specs import Array, BoundedArray, DiscreteArray

import pytest

EPISODE_LENGTH = 10

# Discount specs
F32_2_MIN_0_MAX_1 = BoundedArray(dtype=np.float32,
                                       shape=(2, ),
                                       minimum=0.0,
                                       maximum=1.0)
F32_2x1_MIN_0_MAX_1 = BoundedArray(dtype=np.float32,
                                         shape=(2, 1),
                                         minimum=0.0,
                                         maximum=1.0)
TREE_MIN_0_MAX_1 = {'a': F32_2_MIN_0_MAX_1, 'b': F32_2x1_MIN_0_MAX_1}

# Reward specs
F32 = Array(dtype=np.float32, shape=())
F32_1x3 = Array(dtype=np.float32, shape=(1, 3))
TREE = {'a': F32, 'b': F32_1x3}

TEST_CASES = [
    (None, None),
    (F32_2_MIN_0_MAX_1, F32),
    (F32_2x1_MIN_0_MAX_1, F32_1x3),
    (TREE_MIN_0_MAX_1, TREE),
]

TEST_CASES_LABEL = ['scalar_discount_scalar_reward', 'vector_discount_scalar_reward', 'matrix_discount_matrix_reward', 'tree_discount_tree_reward']


class TestEnvironmentLoop:
    @pytest.mark.parametrize("discount_spec,reward_spec", TEST_CASES, ids=TEST_CASES_LABEL)
    def test_one_episode(self, discount_spec, reward_spec):
        _, loop = _parameterized_setup(discount_spec, reward_spec)
        result = loop.run_episode()
        assert 'episode_length' in result
        assert EPISODE_LENGTH == result['episode_length']
        assert 'episode_return' in result
        assert 'steps_per_second' in result

    @pytest.mark.parametrize("discount_spec,reward_spec", TEST_CASES, ids=TEST_CASES_LABEL)
    def test_run_episodes(self, discount_spec, reward_spec):
        actor, loop = _parameterized_setup(discount_spec, reward_spec)

        # Run the loop. There should be EPISODE_LENGTH update calls per episode.
        loop.run(num_episodes=10)
        assert actor.num_updates == 10 * EPISODE_LENGTH

    @pytest.mark.parametrize("discount_spec,reward_spec", TEST_CASES, ids=TEST_CASES_LABEL)
    def test_run_steps(self, discount_spec, reward_spec):
        actor, loop = _parameterized_setup(discount_spec, reward_spec)

        # Run the loop. This will run 2 episodes so that total number of steps is
        # at least 15.
        loop.run(num_steps=EPISODE_LENGTH + 5)
        assert actor.num_updates == 2 * EPISODE_LENGTH


def _parameterized_setup(discount_spec: Optional[specs.NestedSpec] = None,
                         reward_spec: Optional[specs.NestedSpec] = None):
    """Common setup code that, unlike self.setUp, takes arguments.

  Args:
    discount_spec: None, or a (nested) specs.BoundedArray.
    reward_spec: None, or a (nested) specs.Array.
  Returns:
    environment, actor, loop
  """
    env_kwargs = {'episode_length': EPISODE_LENGTH}
    if discount_spec:
        env_kwargs['discount_spec'] = discount_spec
    if reward_spec:
        env_kwargs['reward_spec'] = reward_spec

    environment = mocks.MockDiscreteEnvironment(**env_kwargs)
    actor = mocks.MockActor(specs.make_environment_spec(environment))
    loop = environment_loop.EnvironmentLoop(actor, environment, should_update=True)
    return actor, loop
