import unittest
import pandas_weld as pdw
import numpy as np


class DataFrameTests(unittest.TestCase):
    def setUp(self):
        data = {'col1': np.array([1, 2, 3, 4]),
                'col2': np.array([5., 6., 7., 8.])}
        index = pdw.MultiIndex.from_product([np.array([1, 2]), np.array([3, 4])], ['a', 'b'])
        self.df = pdw.DataFrame(data, index)

    def test_getitem_column(self):
        expected_result = np.array([1, 2, 3, 4])
        result = self.df['col1'].evaluate(verbose=False)

        np.testing.assert_array_equal(expected_result, result)

    def test_getitem_slice(self):
        data = {'col1': np.array([1, 2]),
                'col2': np.array([5., 6.])}
        index = pdw.MultiIndex.from_product([np.array([1, 2]), np.array([3, 4])], ['a', 'b'])
        expected_result = pdw.DataFrame(data, index)

        result = self.df[:2]

        np.testing.assert_array_equal(expected_result['col1'].evaluate(verbose=False),
                                      result['col1'].evaluate(verbose=False))
        np.testing.assert_array_equal(expected_result['col2'].evaluate(verbose=False),
                                      result['col2'].evaluate(verbose=False))

        for i in xrange(2):
            np.testing.assert_array_equal(expected_result.index.levels[i], result.index.levels[i])
            np.testing.assert_array_equal(expected_result.index.labels[i].evaluate(verbose=False),
                                          result.index.labels[i].evaluate(verbose=False))
        np.testing.assert_array_equal(expected_result.index.names, result.index.names)

    def test_setitem_new(self):
        new_column = np.array([11, 12, 13, 14])

        self.df['col3'] = new_column

        np.testing.assert_array_equal(new_column, self.df['col3'].evaluate(verbose=False))

    def test_setitem_replace(self):
        new_column = np.array([11, 12, 13, 14])

        self.df['col2'] = new_column

        np.testing.assert_array_equal(new_column, self.df['col2'].evaluate(verbose=False))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
