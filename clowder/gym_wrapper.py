import dm_env
import gymnasium as gym

class GymWrapper(dm_env.Environment):
    def __init__(self, environment: gym.Env):
        self._environment = environment
    
    def reset(self) -> dm_env.TimeStep:
        observation = self._environment.reset()
        return dm_env.restart(observation)

    def step(self, action) -> dm_env.TimeStep:
        observation, reward, terminated, truncated, info = self._environment.step(action)
        if terminated:
            return dm_env.termination(reward, observation)
        elif truncated:
            return dm_env.truncation(reward, observation)
        else:
            return dm_env.transition(reward, observation)
    
    