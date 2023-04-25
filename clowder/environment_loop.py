from actor import Actor
import dm_env
import envlogger

class EnvironmentLoop:
    def __init__(self, actor: Actor, environment: dm_env.Environment):
        self._actor = actor
        self._environment = environment
    
    def run_episode(self):
        timestep = self._environment.reset()