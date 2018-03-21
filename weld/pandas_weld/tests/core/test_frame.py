import unittest
import pandas_weld as pdw
import numpy as np
from indexes.test_multi import test_equal_multiindex
from pandas_weld.tests.core.test_series import test_equal_series
from pandas_weld.tests.utils import evaluate_if_necessary


# TODO: add method to check equal DataFrame
class DataFrameTests(unittest.TestCase):
    def setUp(self):
        data = {'col1': np.array([1, 2, 3, 4]),
                'col2': np.array([5., 6., 7., 8.])}
        index = pdw.MultiIndex.from_product([np.array([1, 2]), np.array([3, 4])], ['a', 'b'])
        self.df = pdw.DataFrame(data, index)

    def test_getitem_column(self):
        expected_result = np.array([1, 2, 3, 4])
        result = evaluate_if_necessary(self.df['col1'])

        np.testing.assert_array_equal(expected_result, result)

    def test_getitem_slice(self):
        data = {'col1': np.array([1, 2]),
                'col2': np.array([5., 6.])}
        index = pdw.MultiIndex.from_product([np.array([1, 2]), np.array([3, 4])], ['a', 'b'])
        expected_result = pdw.DataFrame(data, index)

        result = self.df[:2]

        np.testing.assert_array_equal(evaluate_if_necessary(expected_result['col1']),
                                      evaluate_if_necessary(result['col1']))
        np.testing.assert_array_equal(evaluate_if_necessary(expected_result['col2']),
                                      evaluate_if_necessary(result['col2']))

        test_equal_multiindex(expected_result.index, result.index)

    def test_getitem_list(self):
        data = {'col1': np.array([1, 2, 3, 4]),
                'col2': np.array([5., 6., 7., 8.])}
        index = pdw.MultiIndex.from_product([np.array([1, 2]), np.array([3, 4])], ['a', 'b'])
        expected_result = pdw.DataFrame(data, index)

        result = self.df[['col1', 'col2']]

        np.testing.assert_array_equal(evaluate_if_necessary(expected_result['col1']),
                                      evaluate_if_necessary(result['col1']))
        np.testing.assert_array_equal(evaluate_if_necessary(expected_result['col2']),
                                      evaluate_if_necessary(result['col2']))

        test_equal_multiindex(expected_result.index, result.index)

    # i.e. filter like df[df[column] < 10]
    def test_getitem_series(self):
        data = {'col1': np.array([1, 2]),
                'col2': np.array([5., 6.])}
        index = pdw.MultiIndex([np.array([1, 2]), np.array([3, 4])], [np.array([0, 0]), np.array([0, 1])], ['a', 'b'])
        expected_result = pdw.DataFrame(data, index)

        result = self.df[self.df['col1'] < 3]

        np.testing.assert_array_equal(evaluate_if_necessary(expected_result['col1']),
                                      evaluate_if_necessary(result['col1']))
        np.testing.assert_array_equal(evaluate_if_necessary(expected_result['col2']),
                                      evaluate_if_necessary(result['col2']))

        test_equal_multiindex(expected_result.index, result.index)

    def test_setitem_new(self):
        new_column = np.array([11, 12, 13, 14])

        self.df['col3'] = new_column

        np.testing.assert_array_equal(new_column, evaluate_if_necessary(self.df['col3']))

    def test_setitem_series(self):
        new_column = np.array([11, 12, 13, 14])

        self.df['col3'] = pdw.Series(new_column, new_column.dtype, self.df.index)

        np.testing.assert_array_equal(new_column, evaluate_if_necessary(self.df['col3']))

    def test_setitem_replace(self):
        new_column = np.array([11, 12, 13, 14])

        self.df['col2'] = new_column

        np.testing.assert_array_equal(new_column, evaluate_if_necessary(self.df['col2']))

    def test_drop_str(self):
        data = {'col2': np.array([5., 6., 7., 8.])}
        index = pdw.MultiIndex.from_product([np.array([1, 2]), np.array([3, 4])], ['a', 'b'])
        expected_result = pdw.DataFrame(data, index)

        result = self.df.drop('col1')

        self.assertListEqual(expected_result.data.keys(), result.data.keys())
        np.testing.assert_array_equal(evaluate_if_necessary(expected_result['col2']),
                                      evaluate_if_necessary(result['col2']))

        test_equal_multiindex(expected_result.index, result.index)

    def test_drop_list(self):
        data = {}
        index = pdw.MultiIndex.from_product([np.array([1, 2]), np.array([3, 4])], ['a', 'b'])
        expected_result = pdw.DataFrame(data, index)

        result = self.df.drop(['col1', 'col2'])

        self.assertListEqual(expected_result.data.keys(), result.data.keys())

        test_equal_multiindex(expected_result.index, result.index)

    @staticmethod
    def test_element_wise_operation():
        expected_data = {'col1': np.array([2, 4, 6, 8]),
                         'col2': np.array([10, 12, 14, 16])}
        expected_index = pdw.MultiIndex.from_product([np.array([1, 2]), np.array([3, 4])], ['a', 'b'])
        expected_result = pdw.DataFrame(expected_data, expected_index)

        data = {'col1': np.array([1, 2, 3, 4]),
                'col2': np.array([5, 6, 7, 8])}
        index = pdw.MultiIndex.from_product([np.array([1, 2]), np.array([3, 4])], ['a', 'b'])
        result = pdw.DataFrame(data, index) * 2

        np.testing.assert_array_equal(evaluate_if_necessary(expected_result['col1']),
                                      evaluate_if_necessary(result['col1']))
        np.testing.assert_array_equal(evaluate_if_necessary(expected_result['col2']),
                                      evaluate_if_necessary(result['col2']))

        test_equal_multiindex(expected_result.index, result.index)

    def test_aggregate(self):
        # reversed because of dict and not OrderedDict
        expected_result = pdw.Series(np.array([26., 10.], dtype=np.float64),
                                     np.dtype(np.float64),
                                     np.array(['col2', 'col1'], dtype=np.str))

        result = self.df.sum()

        test_equal_series(expected_result, result)

    def test_aggregate_min(self):
        # reversed because of dict and not OrderedDict
        expected_result = pdw.Series(np.array([5., 1.], dtype=np.float64),
                                     np.dtype(np.float64),
                                     np.array(['col2', 'col1'], dtype=np.str))

        result = self.df.min()

        test_equal_series(expected_result, result)

    def test_rename(self):
        data = {'col3': np.array([1, 2, 3, 4]),
                'col2': np.array([5., 6., 7., 8.])}
        index = pdw.MultiIndex.from_product([np.array([1, 2]), np.array([3, 4])], ['a', 'b'])
        expected_result = pdw.DataFrame(data, index)

        result = self.df.rename(columns={'col1': 'col3'})

        self.assertListEqual(expected_result.data.keys(), result.data.keys())
        np.testing.assert_array_equal(evaluate_if_necessary(expected_result['col3']),
                                      evaluate_if_necessary(result['col3']))
        np.testing.assert_array_equal(evaluate_if_necessary(expected_result['col2']),
                                      evaluate_if_necessary(result['col2']))

        test_equal_multiindex(expected_result.index, result.index)

    def test_count(self):
        # reversed because of dict and not OrderedDict
        expected_result = pdw.Series(np.array([4, 4], dtype=np.int64),
                                     np.dtype(np.int64),
                                     np.array(['col2', 'col1'], dtype=np.str))

        result = self.df.count()

        test_equal_series(expected_result, result)

    def test_mean(self):
        # reversed because of dict and not OrderedDict
        expected_result = pdw.Series(np.array([6.5, 2.5], dtype=np.float64),
                                     np.dtype(np.float64),
                                     np.array(['col2', 'col1'], dtype=np.str))

        result = self.df.mean()

        test_equal_series(expected_result, result)

    def test_std(self):
        # reversed because of dict and not OrderedDict
        expected_result = pdw.Series(np.array([1.2909944487358056, 1.2909944487358056], dtype=np.float64),
                                     np.dtype(np.float64),
                                     np.array(['col2', 'col1'], dtype=np.str))

        result = self.df.std()

        test_equal_series(expected_result, result)

    @staticmethod
    def test_join_1d_index():
        df1 = pdw.DataFrame({'col1': np.array([1, 2, 3, 4, 5])},
                            pdw.Index(np.array([1, 3, 4, 5, 6]), np.dtype(np.int64)))
        df2 = pdw.DataFrame({'col2': np.array([1, 2, 3])},
                            pdw.Index(np.array([2, 3, 5]), np.dtype(np.int64)))

        result = df1.merge(df2)

        expected_result = pdw.DataFrame({'col1': np.array([2, 4]), 'col2': np.array([2, 3])},
                                        pdw.Index(np.array([3, 5]), np.dtype(np.int64)))

        np.testing.assert_array_equal(evaluate_if_necessary(expected_result.index),
                                      evaluate_if_necessary(result.index))
        np.testing.assert_array_equal(evaluate_if_necessary(expected_result['col1']),
                                      evaluate_if_necessary(result['col1']))
        np.testing.assert_array_equal(evaluate_if_necessary(expected_result['col2']),
                                      evaluate_if_necessary(result['col2']))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
