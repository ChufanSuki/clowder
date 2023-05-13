import numpy as np
from clowder.ops import action_value_ops as rl
import pytest

@pytest.fixture()
def resource():
    print("setup")
    q_tm1 = np.array([[1, 1, 0], [1, 2, 0]], dtype=np.float32)
    q_t = np.array([[0, 1, 0], [1, 2, 0]], dtype=np.float32)
    a_tm1 = np.array([0, 1], dtype=np.int32)
    pcont_t = np.array([0, 1], dtype=np.float32)
    r_t = np.array([1, 1], dtype=np.float32)
    qlearning = rl.qlearning(q_tm1, a_tm1, r_t, pcont_t, q_t)
    yield qlearning
    print("teardown")

class TestQLearning:
    def test_target(self, resource):
        np.testing.assert_allclose(resource.extra.target, np.array([1, 3], dtype=np.float32))