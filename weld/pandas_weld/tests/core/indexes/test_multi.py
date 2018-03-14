import unittest
import numpy as np
import pandas_weld as pdw
from weld.types import WeldLong
from lazy_data import LazyData
from pandas_weld.tests.utils import evaluate_array_if_necessary


def evaluate_multiindex_if_necessary(index):
    index.levels = evaluate_array_if_necessary(index.levels)
    index.labels = evaluate_array_if_necessary(index.labels)

    return index


def test_equal_multiindex(expected_index, result_index):
    expected_index = evaluate_multiindex_if_necessary(expected_index)
    result_index = evaluate_multiindex_if_necessary(result_index)

    np.testing.assert_array_equal(expected_index.names, result_index.names)
    np.testing.assert_array_equal(expected_index.levels, result_index.levels)
    np.testing.assert_array_equal(expected_index.labels, result_index.labels)


class MultiIndexTests(unittest.TestCase):
    @staticmethod
    def test_from_product_raw():
        levels = [np.array([1, 2]), np.array([3, 4])]
        names = ['a', 'b']

        result = pdw.MultiIndex.from_product(levels, names)

        expected_result = pdw.MultiIndex([np.array([1, 2]), np.array([3, 4])],
                                         [np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])],
                                         ['a', 'b'])

        test_equal_multiindex(expected_result, result)

    @staticmethod
    def test_from_product():
        levels = [LazyData(np.array([1, 2]), WeldLong(), 1), LazyData(np.array([3, 4]), WeldLong(), 1)]
        names = ['a', 'b']

        result = pdw.MultiIndex.from_product(levels, names)

        expected_result = pdw.MultiIndex([LazyData(np.array([1, 2]), WeldLong(), 1),
                                          LazyData(np.array([3, 4]), WeldLong(), 1)],
                                         [LazyData(np.array([0, 0, 1, 1]), WeldLong(), 1),
                                          LazyData(np.array([0, 1, 0, 1]), WeldLong(), 1)],
                                         ['a', 'b'])

        test_equal_multiindex(expected_result, result)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
