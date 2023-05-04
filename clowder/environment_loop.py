import abc
import time
import operator
from enum import IntFlag
from typing import Any, Dict, Optional, List
import dm_env
import tree
import numpy as np
from actor import Actor
from dm_env import TimeStep, specs
from specs import Nest
from worker import Worker


def _generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
    return np.zeros(spec.shape, spec.dtype)


class Phase(IntFlag):
    NONE = 0
    TRAIN = 1
    EVAL = 2
    BOTH = 3


class Loop(Worker):

    @abc.abstractmethod
    def run(self, num_episodes: Optional[int] = None) -> None:
        """
        """


class EpisodeCallback:
    """Callback class for custom episode metrics.
    """

    def __init__(self) -> None:
        self._custom_metrics = {}

    @property
    def custom_metrics(self) -> Dict[str, Any]:
        return self._custom_metrics

    @custom_metrics.setter
    def custom_metrics(self, metrics: Dict[str, Any]) -> None:
        self._custom_metrics = metrics

    def reset(self) -> None:
        self._custom_metrics.clear()

    def on_episode_start(self) -> None:
        pass

    def on_episode_init(self, timestep: TimeStep) -> None:
        pass

    def on_episode_step(self, step: int, action: Nest,
                        timestep: TimeStep) -> None:
        pass

    def on_episode_end(self) -> None:
        pass


class EnvironmentLoop(Loop):

    def __init__(self,
                 actor: Actor,
                 environment: dm_env.Environment,
                 phase: Phase,
                 should_update: bool = False,
                 episode_callback: Optional[EpisodeCallback] = None):
        self._actor = actor
        self._environment = environment
        self._should_update = should_update
        self._phase = phase
        self._episode_callback = episode_callback or EpisodeCallback()

    def run_episode(self):
        # Reset any counts and start the environment.
        episode_start_time = time.perf_counter()
        episode_steps: int = 0
        select_action_durations: List[float] = []
        env_step_durations: List[float] = []
        episode_return = tree.map_structure(_generate_zeros_from_spec,
                                            self._environment.reward_spec())
        env_reset_start = time.perf_counter()
        timestep = self._environment.reset()
        env_reset_duration = time.perf_counter() - env_reset_start
        self._episode_callback.on_episode_init(timestep)

        self._actor.observe_first(timestep)
        self._episode_callback.on_episode_start()
        while not timestep.last():
            episode_steps += 1
            select_action_start = time.perf_counter()
            action = self._actor.select_action(timestep.observation)
            select_action_durations.append(time.perf_counter() -
                                           select_action_start)
            env_step_start = time.perf_counter()
            timestep = self._environment.step(action)
            env_step_durations.append(time.perf_counter() - env_step_start)
            self._actor.observe(action, timestep)
            self._episode_callback.on_episode_step(episode_steps, action,
                                                   timestep)
            if self._should_update:
                self._actor.update()
            # Equivalent to: episode_return += timestep.reward
            # We capture the return value because if timestep.reward is a JAX
            # Array, episode_return will not be mutated in-place. (In all other
            # cases, the returned episode_return will be the same object as the
            # argument episode_return.)
            episode_return = tree.map_structure(operator.iadd, episode_return,
                                                timestep.reward)

        steps_per_second = episode_steps / (time.perf_counter() -
                                            episode_start_time)
        result = {
            'episode_length': episode_steps,
            'episode_return': episode_return,
            'steps_per_second': steps_per_second,
            'env_reset_duration_sec': env_reset_duration,
            'select_action_duration_sec': np.mean(select_action_durations),
            'env_step_duration_sec': np.mean(env_step_durations),
        }
        return result
