import unittest

import numpy as np
from grizzly.encoders import numpy_to_weld_type

import pandas_weld as pdw
from lazy_result import LazyResult
from pandas_weld.tests.utils import evaluate_if_necessary


class RangeIndexTests(unittest.TestCase):
    # noinspection PyMethodMayBeStatic
    def test_init(self):
        result = pdw.RangeIndex(0, 3, 1)

        expected_result = pdw.Index(np.array([0, 1, 2]), np.dtype(np.int64))

        np.testing.assert_array_equal(evaluate_if_necessary(expected_result).data,
                                      evaluate_if_necessary(result).data)

        np.testing.assert_equal(result.start, 0)
        np.testing.assert_equal(result.stop, 3)
        np.testing.assert_equal(result.step, 1)
        np.testing.assert_equal(expected_result.dtype, np.dtype(np.int64))

    # noinspection PyMethodMayBeStatic
    def test_getitem_slice(self):
        result = pdw.RangeIndex(0, 3, 1)[:2]

        expected_result = pdw.Index(np.array([0, 1]), np.dtype(np.int64))

        np.testing.assert_array_equal(evaluate_if_necessary(expected_result).data,
                                      evaluate_if_necessary(result).data)

    # noinspection PyMethodMayBeStatic
    def test_getitem_filter(self):
        to_filter = LazyResult(np.array([True, False, True], dtype=np.dtype(np.bool)),
                               numpy_to_weld_type(np.dtype(np.bool)),
                               1)
        result = pdw.RangeIndex(0, 3, 1)[to_filter]

        expected_result = pdw.Index(np.array([0, 2]), np.dtype(np.int64))

        np.testing.assert_array_equal(evaluate_if_necessary(expected_result).data,
                                      evaluate_if_necessary(result).data)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
