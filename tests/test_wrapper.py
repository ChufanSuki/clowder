"""Tests for base wrapper."""

import copy
import pickle

from tests import mocks
from clowder.wrappers import base

class TestBaseWrapper:

    def test_pickle_unpickle(self):
        test_env = base.EnvironmentWrapper(environment=mocks.MockDiscreteEnvironment())

        test_env_pickled = pickle.dumps(test_env)
        test_env_restored = pickle.loads(test_env_pickled)
        assert test_env.observation_spec() == test_env_restored.observation_spec()

    def test_deepcopy(self):
        test_env = base.EnvironmentWrapper(environment=mocks.MockDiscreteEnvironment())
        copied_env = copy.deepcopy(test_env)
        del copied_env
