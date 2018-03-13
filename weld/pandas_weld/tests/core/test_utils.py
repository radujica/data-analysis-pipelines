import unittest
import numpy as np
import pandas_weld as pdw
from grizzly.encoders import numpy_to_weld_type_mapping
from lazy_data import LazyData


class SubsetTests(unittest.TestCase):
    @staticmethod
    def test_subset_arrays():
        array = np.array([1, 2, 3, 4, 5, 6])
        slice_ = slice(0, 3, 1)

        expected_result = np.array([1, 2, 3])
        result = pdw.subset(array, slice_)

        np.testing.assert_array_equal(expected_result, result.evaluate(verbose=False))

    @staticmethod
    def test_subset_lazy_data():
        array = LazyData(np.array([1, 2, 3, 4, 5, 6]), numpy_to_weld_type_mapping['int64'], 1)
        slice_ = slice(0, 3, 1)

        expected_result = np.array([1, 2, 3])
        result = pdw.subset(array, slice_)

        np.testing.assert_array_equal(expected_result, result.evaluate(verbose=False))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
