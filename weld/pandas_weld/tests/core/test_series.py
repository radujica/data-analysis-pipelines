import unittest
import numpy as np
from grizzly.encoders import numpy_to_weld_type_mapping
from lazy_data import LazyData
from pandas_weld import Series
from pandas_weld.tests.utils import evaluate_if_necessary


class SeriesTests(unittest.TestCase):
    @staticmethod
    def test_getitem_raw():
        data = np.array([1, 2, 3])
        series = Series(data, numpy_to_weld_type_mapping['int64'])

        expected_result = np.array([1, 2])
        result = evaluate_if_necessary(series[:2])

        np.testing.assert_array_equal(expected_result, result)

    @staticmethod
    def test_getitem():
        weld_type = numpy_to_weld_type_mapping['int64']
        data = LazyData(np.array([1, 2, 3]), weld_type, 1)
        series = Series(data.expr, weld_type)

        expected_result = np.array([1, 2])
        result = evaluate_if_necessary(series[:2])

        np.testing.assert_array_equal(expected_result, result)

    @staticmethod
    def test_head_raw():
        data = np.array([1, 2, 3])
        series = Series(data, numpy_to_weld_type_mapping['int64'])

        expected_result = np.array([1, 2])
        result = series.head(2)

        np.testing.assert_array_equal(expected_result, result)

    @staticmethod
    def test_head():
        data = LazyData(np.array([1, 2, 3]), numpy_to_weld_type_mapping['int64'], 1)
        series = Series(data.expr, numpy_to_weld_type_mapping['int64'])

        expected_result = np.array([1, 2])
        result = series.head(2)

        np.testing.assert_array_equal(expected_result, result)

    @staticmethod
    def test_comparison():
        data = np.array([1, 2, 3, 4])
        series = Series(data, numpy_to_weld_type_mapping['int64'])

        expected_result = np.array([True, True, False, False])
        result = evaluate_if_necessary(series < 3)

        np.testing.assert_array_equal(expected_result, result)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
