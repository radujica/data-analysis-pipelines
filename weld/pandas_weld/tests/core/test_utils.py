import unittest
import numpy as np
import pandas_weld as pdw
from grizzly.encoders import numpy_to_weld_type
from lazy_data import LazyData
from pandas_weld.tests.utils import evaluate_if_necessary


class SubsetTests(unittest.TestCase):
    # noinspection PyMethodMayBeStatic
    def test_subset_arrays(self):
        array = np.array([1, 2, 3, 4, 5, 6])
        slice_ = slice(0, 3, 1)

        expected_result = np.array([1, 2, 3])
        result = pdw.subset(array, slice_)

        np.testing.assert_array_equal(expected_result, evaluate_if_necessary(result))

    # noinspection PyMethodMayBeStatic
    def test_subset_lazy_data(self):
        array = LazyData(np.array([1, 2, 3, 4, 5, 6]), numpy_to_weld_type('int64'), 1)
        slice_ = slice(0, 3, 1)

        expected_result = np.array([1, 2, 3])
        result = pdw.subset(array, slice_)

        np.testing.assert_array_equal(expected_result, evaluate_if_necessary(result))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
