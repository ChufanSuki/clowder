from clowder.ops import indexing_ops
import pytest
import numpy as np

class TestBatchIndexing:
    
    @pytest.mark.parametrize("keepdims", [None, True, False])
    def testOrdinaryValues(self, keepdims):
        """Indexing value functions by action for a minibatch of values."""
        values = np.array([[1.1, 1.2, 1.3],
                [1.4, 1.5, 1.6],
                [2.1, 2.2, 2.3],
                [2.4, 2.5, 2.6],
                [3.1, 3.2, 3.3],
                [3.4, 3.5, 3.6],
                [4.1, 4.2, 4.3],
                [4.4, 4.5, 4.6]])
        action_indices = np.array([0, 2, 1, 0, 2, 1, 0, 2])
        result = indexing_ops.batched_index(values, action_indices, keepdims=keepdims)
        expected_result = np.array([1.1, 1.6, 2.2, 2.4, 3.3, 3.5, 4.1, 4.6])
        if keepdims:
            expected_result = np.expand_dims(expected_result, axis=-1)
        
        np.testing.assert_allclose(result, expected_result)
    
    def testValueSequence(self):
        """Indexing value functions by action with a minibatch of sequences."""
        values = np.array([[[1.1, 1.2, 1.3], [1.4, 1.5, 1.6]],
                [[2.1, 2.2, 2.3], [2.4, 2.5, 2.6]],
                [[3.1, 3.2, 3.3], [3.4, 3.5, 3.6]],
                [[4.1, 4.2, 4.3], [4.4, 4.5, 4.6]]])
        action_indices = np.array([[0, 2],
                        [1, 0],
                        [2, 1],
                        [0, 2]])
        result = indexing_ops.batched_index(values, action_indices)
        expected_result = np.array([[1.1, 1.6],
                        [2.2, 2.4],
                        [3.3, 3.5],
                        [4.1, 4.6]])
        np.testing.assert_allclose(result, expected_result)