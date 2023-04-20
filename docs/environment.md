# Environment Design

The main interaction with the `environment` is via the `step()` method.
Each call to an environment's `step()` method takes an `action`($a_t$) returns a `TimeStep` namedtuple with fields `step_type, reward, discount, observation`.
