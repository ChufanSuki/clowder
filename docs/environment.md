# Environment Design

The main interaction with the `environment` is via the `step()` method.
Each call to an environment's `step()` method takes an `action`($a_t$) returns a `TimeStep` namedtuple with fields `step_type, reward, discount, observation`.

Environments should return observations and accept actions in the form of [NumPy arrays](https://docs.scipy.org/doc/numpy/reference/generated/numpy.array.html). Each environment also implements an `observation_spec()` and an `action_spec()` method. Each method should return a structure of Array specs, where the structure should correspond exactly to the format of the actions/observations.

### Batch

We design `Batch` class to store the batch of `TimeStep` and `Action`.  But it is a general data structure to store the batch of any data. We also need it tot support several useful operations, for example, `split`, `stack`. It also enables recursive definition, namely `batch` containing `batch`. In essence, it is a hierarchical named tensors. To implement this, we need a tree-like data container. [dm-tree](https://github.com/deepmind/tree) and [treevalue](https://github.com/opendilab/treevalue) are two good choices. We choose `tree-value` because it provides more function to manipulate `batch`s. Also, it claims it's the fastest compared with `dm-tree`, `jax`'s `pytree` and `tianshou.data.Batch`.

## Compatibility

We `dm-env.Environment` class but it should be compatible with `Gymnasium`. We provide a wrapper class to convert `Gymnasium.Environment` to `dm-env.Environment`.
