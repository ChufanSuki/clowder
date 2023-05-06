from typing import List

from clowder.learner import Learner
from clowder import specs
import pytest


class StepCountingLearner(Learner):
    """A learner which counts `num_steps` and then raises `StopIteration`."""

    def __init__(self, num_steps: int):
        self.step_count = 0
        self.num_steps = num_steps

    def step(self):
        self.step_count += 1
        if self.step_count >= self.num_steps:
            raise StopIteration()

    def get_variables(self, unused: List[str]) -> List[specs.NestedArray]:
        del unused
        return []
    
    def restore(self, state):
        ...
    
    def save(self):
        ...


class TestLearner:

    def test_learner_run_with_limit(self):
        learner = StepCountingLearner(100)
        learner.run(7)
        assert learner.step_count == 7

    def test_learner_run_no_limit(self):
        learner = StepCountingLearner(100)
        with pytest.raises(StopIteration):
            learner.run()
        assert learner.step_count == 100
