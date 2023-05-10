"""Tests for tree_utils."""

import functools
from typing import Sequence

from clowder.utils import tree_utils
import numpy as np
import tree
import pytest

TEST_SEQUENCE = [
    {
        'action': np.array([1.0]),
        'observation': (np.array([0.0, 1.0, 2.0]), ),
        'reward': np.array(1.0),
    },
    {
        'action': np.array([0.5]),
        'observation': (np.array([1.0, 2.0, 3.0]), ),
        'reward': np.array(0.0),
    },
    {
        'action': np.array([0.3]),
        'observation': (np.array([2.0, 3.0, 4.0]), ),
        'reward': np.array(0.5),
    },
]

EXPECTED_SEQUENCE_STACKED = {
    'action': np.array([[1.0], [0.5], [0.3]]),
    'observation': (np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]), ),
    'reward': np.array([1.0, 0.0, 0.5]),
}

NESTED_TEST_SEQUENCE = [
    {
        'action': np.array([1.0]),
        'observation': {
            'x': np.array([3.0, 4.0]),
            'y': np.array([1.0, 2.0]),
            'z': np.array([5.0, 6.0]),
        },
        'reward': np.array(1.0),
    },
    {
        'action': np.array([0.5]),
        'observation': {
            'x': np.array([7.0, 8.0]),
            'y': np.array([9.0, 10.0]),
            'z': np.array([11.0, 12.0]),
        },
        'reward': np.array(0.0),
    },
    {
        'action': np.array([0.3]),
        'observation': {
            'x': np.array([3.5, 4.5]),
            'y': np.array([1.5, 2.5]),
            'z': np.array([5.5, 6.5]),
        },
        'reward': np.array(0.5),
    },
]

EXPECTED_NESTED_SEQUENCE_STACKED = {
    'action': np.array([[1.0], [0.5], [0.3]]),
    'observation': {
        'x': np.array([[3.0, 4.0], [7.0, 8.0], [3.5, 4.5]]),
        'y': np.array([[1.0, 2.0], [9.0, 10.0], [1.5, 2.5]]),
        'z': np.array([[5.0, 6.0], [11.0, 12.0], [5.5, 6.5]]),
    },
    'reward': np.array([1.0, 0.0, 0.5]),
}

TEST_CASES = [(TEST_SEQUENCE, EXPECTED_SEQUENCE_STACKED), (NESTED_TEST_SEQUENCE, EXPECTED_NESTED_SEQUENCE_STACKED)]


class TestSequenceStack:
    """Tests for various tree utilities."""
    @pytest.mark.parametrize('sequence,expected', TEST_CASES)
    def test_stack_sequence_fields(self, sequence, expected):
        """Tests that `stack_sequence_fields` behaves correctly on nested data."""

        stacked = tree_utils.stack_sequence_fields(sequence)

        # Check that the stacked output has the correct structure.
        tree.assert_same_structure(stacked, sequence[0])

        # Check that the leaves have the correct array shapes.
        assert stacked['action'].shape == expected['action'].shape
        tree.assert_same_structure(stacked['observation'], expected['observation'])
        assert stacked['reward'].shape == expected['reward'].shape

        # Check values.
        if isinstance(stacked["observation"], dict):
            for key in stacked["observation"].keys():
                assert np.array_equal(stacked["observation"][key], expected["observation"][key])
        else:
            np.testing.assert_array_equal(stacked['observation'][0], expected['observation'][0])
        np.testing.assert_array_equal(stacked['action'], expected['action'])
        np.testing.assert_array_equal(stacked['reward'], expected['reward'])

    @pytest.mark.parametrize('sequence,expected', TEST_CASES)
    def test_unstack_sequence_fields(self, sequence, expected):
        """Tests that `unstack_sequence_fields(stack_sequence_fields(x)) == x`."""
        stacked = tree_utils.stack_sequence_fields(sequence)
        batch_size = len(sequence)
        unstacked = tree_utils.unstack_sequence_fields(stacked, batch_size)
        tree.map_structure(np.testing.assert_array_equal, unstacked,
                           sequence)

def test_fast_map_structure():
    structure = {
        'a': {
            'b': np.array([0.0])
        },
        'c': (np.array([1.0]), np.array([2.0])),
        'd': [np.array(3.0), np.array(4.0)],
    }
    def map_fn(x: np.ndarray, y: np.ndarray):
        return x + y
    
    single_arg_map_fn = functools.partial(map_fn, y=np.array([1.0]))
    
    expected_mapped_structure = (tree.map_structure(
        single_arg_map_fn, structure))
    mapped_structure = (tree_utils.fast_map_structure(
        single_arg_map_fn, structure))
    assert mapped_structure == expected_mapped_structure

    expected_double_mapped_structure = (tree.map_structure(
        map_fn, structure, expected_mapped_structure))
    double_mapped_structure = (tree_utils.fast_map_structure(
        map_fn, structure, mapped_structure))
    assert double_mapped_structure == expected_double_mapped_structure

def test_fast_map_structure_with_path():
    structure = {
        'a': {
            'b': np.array([0.0])
        },
        'c': (np.array([1.0]), np.array([2.0])),
        'd': [np.array(3.0), np.array(4.0)],
    }

    def map_fn(path: Sequence[str], x: np.ndarray, y: np.ndarray):
        return x + y + len(path)

    single_arg_map_fn = functools.partial(map_fn, y=np.array([0.0]))

    expected_mapped_structure = (tree.map_structure_with_path(
        single_arg_map_fn, structure))
    mapped_structure = (tree_utils.fast_map_structure_with_path(
        single_arg_map_fn, structure))
    assert mapped_structure == expected_mapped_structure

    expected_double_mapped_structure = (tree.map_structure_with_path(
        map_fn, structure, mapped_structure))
    double_mapped_structure = (tree_utils.fast_map_structure_with_path(
        map_fn, structure, mapped_structure))
    assert double_mapped_structure == expected_double_mapped_structure
