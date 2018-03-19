import unittest
import numpy as np
from grizzly.encoders import numpy_to_weld_type
from lazy_data import LazyData
from pandas_weld import Series, RangeIndex, MultiIndex, Index
from indexes import test_equal_multiindex
from ..utils import evaluate_if_necessary


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
    np.testing.assert_equal(series1.data_id, series2.data_id)
    if isinstance(series1.index, MultiIndex):
        test_equal_multiindex(series1.index, series2.index)
    else:
        np.testing.assert_array_equal(evaluate_if_necessary(series1.index),
                                      evaluate_if_necessary(series2.index))
    np.testing.assert_array_equal(evaluate_if_necessary(series1),
                                  evaluate_if_necessary(series2))


class SeriesTests(unittest.TestCase):
    @staticmethod
    def test_getitem_slice_raw():
        data = np.array([1, 2, 3])
        series = Series(data, np.dtype(np.int64), RangeIndex(0, 3, 1))

        expected_result = Series(np.array([1, 2]), np.dtype(np.int64), RangeIndex(0, 2, 1))
        result = series[:2]

        test_equal_series(expected_result, result)

    @staticmethod
    def test_getitem_slice():
        weld_type = numpy_to_weld_type('int64')
        data = LazyData(np.array([1, 2, 3]), weld_type, 1)
        series = Series(data.expr, np.dtype(np.int64), RangeIndex(0, 3, 1))

        expected_result = Series(np.array([1, 2]), np.dtype(np.int64), RangeIndex(0, 2, 1))
        result = series[:2]

        test_equal_series(expected_result, result)

    @staticmethod
    def test_getitem_series():
        data = np.array([1, 2, 3])
        series = Series(data, np.dtype(np.int64), RangeIndex(0, 3, 1))

        expected_result = Series(np.array([1, 3]), np.dtype(np.int64), Index(np.array([0, 2]), np.dtype(np.int64)))
        result = series[series != 2]

        test_equal_series(expected_result, result)

    @staticmethod
    def test_head_raw():
        data = np.array([1, 2, 3])
        series = Series(data, np.dtype(np.int64), RangeIndex(0, 3, 1))

        expected_result = np.array([1, 2])
        result = series.head(2)

        np.testing.assert_array_equal(expected_result, result)

    @staticmethod
    def test_head():
        data = LazyData(np.array([1, 2, 3]), np.dtype(np.int64), 1)
        series = Series(data.expr, np.dtype(np.int64), RangeIndex(0, 2, 1))

        expected_result = np.array([1, 2])
        result = series.head(2)

        np.testing.assert_array_equal(expected_result, result)

    @staticmethod
    def test_comparison():
        data = np.array([1, 2, 3, 4])
        series = Series(data, np.dtype(np.int64), RangeIndex(0, 4, 1))

        expected_result = Series(np.array([True, True, False, False]), np.dtype(np.bool), RangeIndex(0, 4, 1))
        result = series < 3

        test_equal_series(expected_result, result)

    @staticmethod
    def test_element_wise_operation():
        data = np.array([1, 2, 3])
        series = Series(data, np.dtype(np.int64), RangeIndex(0, 3, 1))

        expected_result = Series(np.array([2, 4, 6]), np.dtype(np.int64), RangeIndex(0, 3, 1))
        result = series * 2

        test_equal_series(expected_result, result)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
