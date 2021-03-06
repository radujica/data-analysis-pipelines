import unittest

import numpy as np
from grizzly.encoders import numpy_to_weld_type

import pandas_weld as pdw
from lazy_result import LazyResult
from pandas_weld.tests.utils import evaluate_if_necessary


class IndexTests(unittest.TestCase):
    # noinspection PyMethodMayBeStatic
    def test_getitem_slice(self):
        result = pdw.Index(np.array([1, 2, 3]), np.dtype(np.int64))[:2]

        expected_result = pdw.Index(np.array([1, 2]), np.dtype(np.int64))

        np.testing.assert_array_equal(evaluate_if_necessary(expected_result).data,
                                      evaluate_if_necessary(result).data)

    # noinspection PyMethodMayBeStatic
    def test_getitem_filter(self):
        to_filter = LazyResult(np.array([True, False, True], dtype=np.dtype(np.bool)),
                               numpy_to_weld_type(np.dtype(np.bool)),
                               1)
        result = pdw.Index(np.array([1, 2, 3]), np.dtype(np.int64))[to_filter]

        expected_result = pdw.Index(np.array([1, 3]), np.dtype(np.int64))

        np.testing.assert_array_equal(evaluate_if_necessary(expected_result).data,
                                      evaluate_if_necessary(result).data)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
