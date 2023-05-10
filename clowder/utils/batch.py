from clowder import specs
import numpy as np
import tree
def add_batch_dim(nest: specs.NestedArray) -> specs.NestedArray:
    """Adds a batch dimension to each leaf of a nested structure of numpy array."""
    return tree.map_structure(lambda x: np.expand_dims(x, axis=0), nest)

def squeeze_batch_dim(nest: specs.NestedArray) -> specs.NestedArray:
    """Squeezes out a batch dimension from each leaf of a nested structure."""
    return tree.map_structure(lambda x: np.squeeze(x, axis=0), nest)

def batch_concat(inputs: specs.NestedArray) -> specs.NestedArray:
    """Concatenate a collection of Tensors while preserving the batch dimension.

    This takes a potentially nested collection of tensors, flattens everything
    but the batch (first) dimension, and concatenates along the resulting data
    (second) dimension.

    Args:
        inputs: a tensor or nested collection of tensors.

    Returns:
        A concatenated tensor which maintains the batch dimension but concatenates
        all other data along the flattened second dimension.
    """
    flat_leaves = tree.map_structure(lambda x: np.reshape(x, (x.shape[0], -1)), inputs)
    return np.concatenate(tree.flatten(flat_leaves), axis=-1)

def to_numpy_squeeze(inputs: specs.NestedTensor) -> specs.NestedArray:
    """Converts to numpy and squeezes out dummy batch dimension."""
    return tree.map_structure(lambda x: np.asarray(x).squeeze(axis=0), inputs)
