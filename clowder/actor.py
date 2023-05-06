import abc
class Actor(abc.ABC):
    def select_action(self, observation):
        pass
    
    def observe(self, action, next_timestep):
        pass
    
    def observe_first(self, timestep):
        pass
    
    def update(self):
        pass