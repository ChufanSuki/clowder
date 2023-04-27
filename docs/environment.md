# Environment Design

The main interaction with the `environment` is via the `step()` method.
Each call to an environment's `step()` method takes an `action`($a_t$) returns a `TimeStep` namedtuple with fields `step_type, reward, discount, observation`.

Environments should return observations and accept actions in the form of `NestedArray`. Each environment also implements an `observation_spec()` and an `action_spec()` method. Each method should return a structure of Array specs `NestedSpec`, where the structure should correspond exactly to the format of the actions/observations.

## Compatibility

We `dm_env.Environment` class but it should be compatible with `Gymnasium`. We provide a wrapper class to convert `Gymnasium.Environment` to `dm_env.Environment`.

## Environment Pool

`EnvPool` is a class that manages a pool of environments. It is useful for parallelizing the interaction with the environment. It is also useful for managing multiple environments with different configurations.
