import unittest
import numpy as np
from grizzly.encoders import numpy_to_weld_type_mapping
from lazy_data import LazyData
from pandas_weld import Series


class SeriesTests(unittest.TestCase):
    @staticmethod
    def test_getitem_raw():
        data = np.array([1, 2, 3])
        series = Series(data, numpy_to_weld_type_mapping['int64'])

        expected_result = np.array([1, 2])
        result = series[:2].evaluate(verbose=False)

        np.testing.assert_array_equal(expected_result, result)

    @staticmethod
    def test_getitem():
        weld_type = numpy_to_weld_type_mapping['int64']
        data = LazyData(np.array([1, 2, 3]), weld_type, 1)
        series = Series(data.expr, weld_type)

        expected_result = np.array([1, 2])
        result = series[:2].evaluate(verbose=False)

        np.testing.assert_array_equal(expected_result, result)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
