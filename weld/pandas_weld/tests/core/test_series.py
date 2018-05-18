import unittest

import numpy as np
from grizzly.encoders import numpy_to_weld_type, WeldObject

from indexes import test_equal_multiindex
from lazy_result import LazyResult
from pandas_weld import Series, RangeIndex, MultiIndex, Index
from ..utils import evaluate_if_necessary


# TODO all: reverse expected and actual; apparently numpy does actual first...
def test_equal_series(series1, series2):
    """ Assert if 2 series are equal

    It is assumed the indexes are of the same dimension

    Parameters
    ----------
    series1 : Series
    series2 : Series

    """
    np.testing.assert_equal(series1.dtype, series2.dtype)
    np.testing.assert_equal(series1.name, series2.name)
    if isinstance(series1.index, MultiIndex):
        test_equal_multiindex(series1.index, series2.index)
    else:
        np.testing.assert_array_equal(evaluate_if_necessary(series1.index),
                                      evaluate_if_necessary(series2.index))
    np.testing.assert_array_equal(evaluate_if_necessary(series1),
                                  evaluate_if_necessary(series2))


class SeriesTests(unittest.TestCase):
    # noinspection PyMethodMayBeStatic
    def test_getitem_slice_raw(self):
        data = np.array([1, 2, 3])
        series = Series(data, np.dtype(np.int64), RangeIndex(0, 3, 1))

        expected_result = Series(np.array([1, 2]), np.dtype(np.int64), RangeIndex(0, 2, 1))
        result = series[:2]

        test_equal_series(expected_result, result)

    # noinspection PyMethodMayBeStatic
    def test_getitem_slice(self):
        weld_type = numpy_to_weld_type('int64')
        data = LazyResult(np.array([1, 2, 3]), weld_type, 1)
        series = Series(data.expr, np.dtype(np.int64), RangeIndex(0, 3, 1))

        expected_result = Series(np.array([1, 2]), np.dtype(np.int64), RangeIndex(0, 2, 1))
        result = series[:2]

        test_equal_series(expected_result, result)

    # noinspection PyMethodMayBeStatic
    def test_getitem_series(self):
        data = np.array([1, 2, 3])
        series = Series(data, np.dtype(np.int64), RangeIndex(0, 3, 1))

        expected_result = Series(np.array([1, 3]), np.dtype(np.int64), Index(np.array([0, 2]), np.dtype(np.int64)))
        result = series[series != 2]

        test_equal_series(expected_result, result)

    # noinspection PyMethodMayBeStatic
    def test_head_raw(self):
        data = np.array([1, 2, 3])
        series = Series(data, np.dtype(np.int64), RangeIndex(0, 3, 1))

        expected_result = np.array([1, 2])
        result = series.head(2)

        np.testing.assert_array_equal(expected_result, result)

    # noinspection PyMethodMayBeStatic
    def test_head(self):
        data = LazyResult(np.array([1, 2, 3]), np.dtype(np.int64), 1)
        series = Series(data.expr, np.dtype(np.int64), RangeIndex(0, 2, 1))

        expected_result = np.array([1, 2])
        result = series.head(2)

        np.testing.assert_array_equal(expected_result, result)

    # noinspection PyMethodMayBeStatic
    def test_comparison(self):
        data = np.array([1, 2, 3, 4])
        series = Series(data, np.dtype(np.int64), RangeIndex(0, 4, 1))

        expected_result = Series(np.array([True, True, False, False]), np.dtype(np.bool), RangeIndex(0, 4, 1))
        result = series < 3

        test_equal_series(expected_result, result)

    # noinspection PyMethodMayBeStatic
    def test_bitwise_and(self):
        data = np.array([True, True, False, False])
        series = Series(data, np.dtype(np.bool), RangeIndex(0, 4, 1))
        data_other = np.array([True, False, True, False])
        series_other = Series(data_other, np.dtype(np.bool), RangeIndex(0, 4, 1))

        expected_result = Series(np.array([True, False, False, False]), np.dtype(np.bool), RangeIndex(0, 4, 1))
        result = series & series_other

        test_equal_series(expected_result, result)

    # noinspection PyMethodMayBeStatic
    def test_element_wise_operation(self):
        data = np.array([1, 2, 3])
        series = Series(data, np.dtype(np.int64), RangeIndex(0, 3, 1))

        expected_result = Series(np.array([2, 4, 6]), np.dtype(np.int64), RangeIndex(0, 3, 1))
        result = series * 2

        test_equal_series(expected_result, result)

    # noinspection PyMethodMayBeStatic
    def test_array_operation(self):
        data = np.array([1, 2, 3])
        series = Series(data, np.dtype(np.int64), RangeIndex(0, 3, 1))

        expected_result = Series(np.array([3, 5, 7]), np.dtype(np.int64), RangeIndex(0, 3, 1))

        result = series + Series(np.array([2, 3, 4]), np.dtype(np.int64), RangeIndex(0, 3, 1))

        test_equal_series(expected_result, result)

    # noinspection PyMethodMayBeStatic
    def test_aggregate(self):
        data = np.array([1, 2, 3])
        series = Series(data, np.dtype(np.int64), RangeIndex(0, 3, 1))

        expected_result = 6
        result = series.sum()

        np.testing.assert_equal(expected_result, evaluate_if_necessary(result))

    # noinspection PyMethodMayBeStatic
    def test_aggregate_min(self):
        data = np.array([1, 2, 3])
        series = Series(data, np.dtype(np.int64), RangeIndex(0, 3, 1))

        expected_result = 1
        result = series.min()

        np.testing.assert_equal(expected_result, evaluate_if_necessary(result))

    # noinspection PyMethodMayBeStatic
    def test_count(self):
        data = np.array([1, 2, 3])
        series = Series(data, np.dtype(np.int64), RangeIndex(0, 3, 1))

        expected_result = 3
        result = series.count()

        np.testing.assert_equal(expected_result, evaluate_if_necessary(result))

    # noinspection PyMethodMayBeStatic
    def test_mean(self):
        data = np.array([1, 2, 3, 4])
        series = Series(data, np.dtype(np.int64), RangeIndex(0, 4, 1))

        expected_result = 2.5
        result = series.mean()

        np.testing.assert_equal(expected_result, evaluate_if_necessary(result))

    # noinspection PyMethodMayBeStatic
    def test_std(self):
        data = np.array([1, 2, 3, 4])
        series = Series(data, np.dtype(np.int64), RangeIndex(0, 4, 1))

        expected_result = 1.2909944487358056
        result = series.std()

        np.testing.assert_equal(expected_result, evaluate_if_necessary(result))

    # noinspection PyMethodMayBeStatic
    def test_agg(self):
        data = np.array([1, 2, 3, 4])
        series = Series(data, np.dtype(np.int64), RangeIndex(0, 4, 1))

        expected_result = Series(np.array([1., 4.], dtype=np.float64),
                                 np.dtype('float64'),
                                 Index(np.array(['min', 'max'], dtype=np.str), np.dtype(np.str)))
        result = series.agg(['min', 'max'])

        test_equal_series(expected_result, result)

    # noinspection PyMethodMayBeStatic
    def test_map_weld_code(self):
        series = Series(np.array([1, 3, 4]), np.dtype(np.int64), RangeIndex(0, 3, 1))

        weld_template = "map(%(self)s, |e| e + %(scalar)s)"
        mapping = {'scalar': '2L'}
        result = series.map(weld_template, mapping)

        expected_result = Series(np.array([3, 5, 6]), np.dtype(np.int64), RangeIndex(0, 3, 1))

        test_equal_series(expected_result, result)

    # noinspection PyMethodMayBeStatic
    def test_map_weld_cudf(self):
        import os
        WeldObject.load_binary(os.path.dirname(__file__) + '/cudf/udf_c.so')

        series = Series(np.array([1, 3, 4]), np.dtype(np.int64), RangeIndex(0, 3, 1))

        weld_template = "cudf[udf_add, vec[i64]](%(self)s, %(scalar)s)"
        mapping = {'scalar': '2L'}
        result = series.map(weld_template, mapping)

        expected_result = Series(np.array([3, 5, 6]), np.dtype(np.int64), RangeIndex(0, 3, 1))

        test_equal_series(expected_result, result)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
