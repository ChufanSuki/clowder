from clowder.batch import Batch
import numpy as np
import tree

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

if __name__ == '__main__':
    test_batch()