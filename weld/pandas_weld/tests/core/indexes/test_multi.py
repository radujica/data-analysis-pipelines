import unittest

import numpy as np
from grizzly.encoders import numpy_to_weld_type
from weld.types import WeldLong

import pandas_weld as pdw
from lazy_result import LazyResult
from pandas_weld.tests.utils import evaluate_array_if_necessary


def evaluate_multiindex_if_necessary(index):
    index.levels = evaluate_array_if_necessary(index.levels)
    index.labels = evaluate_array_if_necessary(index.labels)

    return index


def test_equal_multiindex(expected_index, result_index):
    expected_index = evaluate_multiindex_if_necessary(expected_index)
    result_index = evaluate_multiindex_if_necessary(result_index)

    np.testing.assert_array_equal(expected_index.names, result_index.names)
    assert len(expected_index.levels) == len(result_index.levels)
    assert len(expected_index.labels) == len(result_index.labels)
    for i in xrange(len(expected_index.levels)):
        np.testing.assert_array_equal(expected_index.levels[i], result_index.levels[i])
        np.testing.assert_array_equal(expected_index.labels[i], result_index.labels[i])


class MultiIndexTests(unittest.TestCase):
    # noinspection PyMethodMayBeStatic
    def test_from_product_raw(self):
        levels = [np.array([1, 2]), np.array([3, 4])]
        names = ['a', 'b']

        result = pdw.MultiIndex.from_product(levels, names)

        expected_result = pdw.MultiIndex([np.array([1, 2]), np.array([3, 4])],
                                         [np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])],
                                         ['a', 'b'])

        test_equal_multiindex(expected_result, result)

    # noinspection PyMethodMayBeStatic
    def test_from_product(self):
        levels = [LazyResult(np.array([1, 2]), WeldLong(), 1), LazyResult(np.array([3, 4]), WeldLong(), 1)]
        names = ['a', 'b']

        result = pdw.MultiIndex.from_product(levels, names)

        expected_result = pdw.MultiIndex([LazyResult(np.array([1, 2]), WeldLong(), 1),
                                          LazyResult(np.array([3, 4]), WeldLong(), 1)],
                                         [LazyResult(np.array([0, 0, 1, 1]), WeldLong(), 1),
                                          LazyResult(np.array([0, 1, 0, 1]), WeldLong(), 1)],
                                         ['a', 'b'])

        test_equal_multiindex(expected_result, result)

    # noinspection PyMethodMayBeStatic
    # TODO: this is broken! should be OrderedDict
    def test_from_arrays(self):
        arrays = [np.array([1, 1, 2, 2]), np.array([3, 4, 3, 4])]
        names = ['a', 'b']

        result = pdw.MultiIndex.from_arrays(arrays, names)

        expected_result = pdw.MultiIndex([np.array([1, 2]),
                                          np.array([3, 4])],
                                         [np.array([0, 0, 1, 1]),
                                          np.array([0, 1, 0, 1])],
                                         ['a', 'b'])

        test_equal_multiindex(expected_result, result)

    # noinspection PyMethodMayBeStatic
    def test_getitem_filter(self):
        levels = [LazyResult(np.array([1, 2]), WeldLong(), 1), LazyResult(np.array([3, 4]), WeldLong(), 1)]
        names = ['a', 'b']

        to_filter = LazyResult(np.array([True, False, True, False], dtype=np.bool),
                               numpy_to_weld_type(np.dtype(np.bool)),
                               1)
        result = pdw.MultiIndex.from_product(levels, names)[to_filter]

        expected_result = pdw.MultiIndex([LazyResult(np.array([1, 2]), WeldLong(), 1),
                                          LazyResult(np.array([3, 4]), WeldLong(), 1)],
                                         [LazyResult(np.array([0, 1]), WeldLong(), 1),
                                          LazyResult(np.array([0, 0]), WeldLong(), 1)],
                                         ['a', 'b'])

        test_equal_multiindex(expected_result, result)

    # noinspection PyMethodMayBeStatic
    def test_getitem_slice(self):
        levels = [LazyResult(np.array([1, 2]), WeldLong(), 1), LazyResult(np.array([3, 4]), WeldLong(), 1)]
        names = ['a', 'b']

        result = pdw.MultiIndex.from_product(levels, names)[:4]

        expected_result = pdw.MultiIndex([LazyResult(np.array([1, 2]), WeldLong(), 1),
                                          LazyResult(np.array([3, 4]), WeldLong(), 1)],
                                         [LazyResult(np.array([0, 0, 1, 1]), WeldLong(), 1),
                                          LazyResult(np.array([0, 1, 0, 1]), WeldLong(), 1)],
                                         ['a', 'b'])

        test_equal_multiindex(expected_result, result)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
