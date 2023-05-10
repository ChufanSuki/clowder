"""Tests for gym_wrapper."""

import pytest

from dm_env import specs
import numpy as np
from clowder.gym_wrapper import GymWrapper, _convert_to_spec, GymAtariAdapter

SKIP_GYM_TESTS = False
SKIP_GYM_MESSAGE = 'gym not installed.'
SKIP_ATARI_TESTS = False
SKIP_ATARI_MESSAGE = ''

try:
    # pylint: disable=g-import-not-at-top
    import gymnasium as gym
    # pylint: enable=g-import-not-at-top
except ModuleNotFoundError:
    SKIP_GYM_TESTS = True

try:
    import ale_py  # pylint: disable=g-import-not-at-top
except ModuleNotFoundError as e:
    SKIP_ATARI_TESTS = True
    SKIP_ATARI_MESSAGE = str(e)
except Exception as e:  # pylint: disable=broad-except
    # This exception is raised by atari_py.get_game_path('pong') if the Atari ROM
    # file has not been installed.
    SKIP_ATARI_TESTS = True
    SKIP_ATARI_MESSAGE = str(e)
    del ale_py
else:
    del ale_py


@pytest.mark.skipif(SKIP_GYM_TESTS, reason=SKIP_GYM_MESSAGE)
class TestGymWrapper:

    def test_gym_cartpole(self):
        env = GymWrapper(gym.make('CartPole-v0'))

        # Test converted observation spec.
        observation_spec: specs.BoundedArray = env.observation_spec()
        assert isinstance(observation_spec, specs.BoundedArray)
        assert observation_spec.shape == (4, )
        assert observation_spec.minimum.shape == (4, )
        assert observation_spec.maximum.shape == (4, )
        assert observation_spec.dtype == np.dtype('float32')

        # Test converted action spec.
        action_spec: specs.BoundedArray = env.action_spec()
        assert isinstance(action_spec, specs.DiscreteArray)
        assert action_spec.shape == ()
        assert action_spec._minimum == 0
        assert action_spec._maximum == 1
        assert action_spec.num_values == 2
        assert action_spec.dtype == np.dtype('int64')

        # Test step.
        timestep = env.reset()
        assert timestep.first()
        timestep = env.step(1)
        assert timestep.reward == 1.0
        assert np.isscalar(timestep.reward)
        assert timestep.observation.shape == (4, )
        env.close()

    def test_early_truncation(self):
        # Pendulum has no early termination condition. Recent versions of gym force
        # to use v1. We try both in case an earlier version is installed.
        try:
            gym_env = gym.make('Pendulum-v1')
        except:  # pylint: disable=bare-except
            gym_env = gym.make('Pendulum-v0')
        env = GymWrapper(gym_env)
        ts = env.reset()
        while not ts.last():
            ts = env.step(env.action_spec().generate_value())
        assert ts.discount == 1.0
        assert np.isscalar(ts.reward)
        env.close()

    def test_multi_discrete(self):
        space = gym.spaces.MultiDiscrete([2, 3])
        spec = _convert_to_spec(space)

        spec.validate([0, 0])
        spec.validate([1, 2])

        pytest.raises(ValueError, spec.validate, [2, 2])
        pytest.raises(ValueError, spec.validate, [1, 3])


@pytest.mark.skipif(SKIP_ATARI_TESTS, reason=SKIP_ATARI_MESSAGE)
class TestAtariGymWrapper:

    def test_pong(self):
        env = gym.make('ALE/Pong-v5', full_action_space=True)
        env = GymAtariAdapter(env)

        # Test converted observation spec. This should expose (RGB, LIVES).
        observation_spec = env.observation_spec()
        assert isinstance(observation_spec[0], specs.BoundedArray)
        assert isinstance(observation_spec[1], specs.Array)

        # Test converted action spec.
        action_spec: specs.DiscreteArray = env.action_spec()[0]
        assert isinstance(action_spec, specs.DiscreteArray)
        assert action_spec.shape == ()
        assert action_spec.minimum == 0
        assert action_spec.maximum == 17
        assert action_spec.num_values == 18
        assert action_spec.dtype == np.dtype('int64')

        # Test step.
        timestep = env.reset()
        assert timestep.first()
        _ = env.step([np.array(0)])
        env.close()

