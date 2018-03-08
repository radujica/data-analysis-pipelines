import unittest
import numpy as np
import pandas_weld as pdw
from weld.types import WeldLong
from lazy_data import LazyData


class MultiIndexTests(unittest.TestCase):
    @staticmethod
    def test_from_product_raw():
        levels = [np.array([1, 2]), np.array([3, 4])]
        names = ['a', 'b']

        result = pdw.MultiIndex.from_product(levels, names)

        expected_result = pdw.MultiIndex([np.array([1, 2]), np.array([3, 4])],
                                         [np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])],
                                         ['a', 'b'])

        np.testing.assert_array_equal(expected_result.levels[0], result.levels[0])
        np.testing.assert_array_equal(expected_result.levels[1], result.levels[1])
        np.testing.assert_array_equal(expected_result.labels[0], result.labels[0].evaluate(verbose=False))
        np.testing.assert_array_equal(expected_result.labels[1], result.labels[1].evaluate(verbose=False))
        np.testing.assert_array_equal(expected_result.names, result.names)

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

        for i in xrange(2):
            np.testing.assert_array_equal(expected_result.levels[i].evaluate(verbose=False),
                                          result.levels[i].evaluate(verbose=False))
            np.testing.assert_array_equal(expected_result.labels[i].evaluate(verbose=False),
                                          result.labels[i].evaluate(verbose=False))

        np.testing.assert_array_equal(expected_result.names, result.names)


def main():
    unittest.main()


if __name__ == '__main__':
    main()
