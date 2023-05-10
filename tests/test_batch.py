
from clowder.batch import Batch
import numpy as np
import tree
from clowder.utils import batch

def test_batch():
    b1 = Batch({"obs": np.array([1, 2, 3])})
    np.testing.assert_array_equal(b1.obs, np.array([1, 2, 3]))
    b2 = Batch({"obs": np.array([2, 3, 4])})
    b3 = Batch({"obs": {"x": np.array([1, 2, 3]), "y": np.array([0, 1, 2])}})
    b4 = tree.map_structure(lambda *args: np.stack(args), b1.obs, b2.obs)
    b5 = Batch({"obs": {"img": np.zeros((3, 3)), "vector": np.zeros(5)}})
    b6 = Batch([Batch({"obs": np.array([1, 2, 3])}), Batch({"obs": np.array([4, 5, 6])})])
    np.testing.assert_array_equal(b6.obs, np.array([[1, 2, 3], [4, 5, 6]]))
    b7 = Batch([Batch({"obs": {"x": np.array([1, 2, 3]), "y": np.array([0, 1, 2])}}), Batch({"obs": {"x": np.array([2, 3, 4]), "y": np.array([1, 2, 3])}})])

def test_add_squeeze_batch_dim():
    inputs = [
        np.zeros(shape=(2)),
        {
            'foo': np.zeros(shape=(5, 3))
        },
        [np.zeros(shape=(1))],
    ]
    output = batch.add_batch_dim(inputs)
    squeezed_output = batch.squeeze_batch_dim(output)
    assert output[0].shape == (1, 2)
    assert output[1]['foo'].shape == (1, 5, 3)
    assert output[2][0].shape == (1, 1)
    assert squeezed_output[0].shape == (2,)
    assert squeezed_output[1]['foo'].shape == (5, 3)
    assert squeezed_output[2][0].shape == (1,)


def test_batch_concat():
    batch_size = 32
    inputs = [
        np.zeros(shape=(batch_size, 2)),
        {
            'foo': np.zeros(shape=(batch_size, 5, 3))
        },
        [np.zeros(shape=(batch_size, 1))],
    ]

    output_shape = batch.batch_concat(inputs).shape
    expected_shape = (batch_size, 2 + 5 * 3 + 1)
    assert output_shape == expected_shape


if __name__ == '__main__':
    test_batch()
    test_add_squeeze_batch_dim()
    test_batch_concat()