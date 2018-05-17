import unittest
import numpy as np
import os
import netCDF4_weld
import pandas_weld as pdw
from weld.weldobject import WeldObject
from lazy_result import LazyResult


# TODO: tests with intermediate results
class IntegrationTests(unittest.TestCase):
    DIR_PATH = (os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/pandas_weld/tests/io'
    PATH_NETCDF4 = DIR_PATH + '/sample_ext.nc'
    PATH_CSV = DIR_PATH + '/sample.csv'

    def is_placeholder(self, value):
        self.assertTrue(value in LazyResult.data_mapping)

    def is_raw(self, value):
        self.assertIsInstance(value, np.ndarray)

    def setUp(self):
        self.ds = netCDF4_weld.Dataset(self.PATH_NETCDF4)
        self.df_netcdf4 = pdw.read_netcdf4(self.PATH_NETCDF4)
        self.df_csv = pdw.read_csv(self.PATH_CSV)
        self.raw_data30 = np.arange(30, dtype=np.float32)
        self.raw_data4 = np.arange(4, dtype=np.float64)

    def test_original_placeholders(self):
        # all data from file should be represented with placeholders
        for k in self.ds.variables:
            [self.is_placeholder(v) for v in self.ds.variables[k].expr.context.values()]
        for k in self.df_netcdf4:
            [self.is_placeholder(v) for v in self.df_netcdf4[k].expr.context.values()]

    # tests what evaluate does on a column
    def _test_column(self, column):
        self.assertEqual(1, len(column.expr.context))
        data_id = column.expr.context.values()[0]
        weld_input_id = column.expr.context.keys()[0]
        # the placeholder should be replaced by raw data
        self.is_placeholder(data_id)
        # raw data should be cached after evaluate
        self.assertNotIn(data_id, LazyResult.data_cache)
        # evaluate should run on just a placeholder
        self.assertEqual(weld_input_id, column.expr.weld_code)

        column.evaluate()

        self.assertEqual(1, len(column.expr.context))
        replaced_placeholder = column.expr.context.values()[0]
        new_weld_input_id = column.expr.context.keys()[0]
        self.is_raw(replaced_placeholder)
        self.assertEqual(weld_input_id, new_weld_input_id)
        self.assertIn(data_id, LazyResult.data_cache)
        self.is_raw(LazyResult.data_cache[data_id])
        np.testing.assert_array_equal(replaced_placeholder, LazyResult.data_cache[data_id])

    def test_evaluate_column_ds(self):
        self._test_column(self.ds.variables['tg'])

    def test_evaluate_column_df_netcdf4(self):
        self._test_column(self.df_netcdf4['tg'])

    def test_evaluate_column_df_csv(self):
        self._test_column(self.df_csv['x'])

    # tests that an operation is recorded and data is read only on evaluate
    def _test_operation_column(self, column, to_add):
        self.assertEqual(1, len(column.expr.context))
        data_id = column.expr.context.values()[0]
        weld_input_id = column.expr.context.keys()[0]
        self.is_placeholder(data_id)
        self.assertEqual(weld_input_id, column.expr.weld_code)
        self.assertNotIn(data_id, LazyResult.data_cache)

        new_column = column + to_add

        self.assertEqual(1, len(column.expr.context))
        new_data_id = new_column.expr.context.values()[0]
        new_weld_input_id = new_column.expr.context.keys()[0]
        self.is_placeholder(new_data_id)
        self.assertEqual(weld_input_id, new_weld_input_id)
        self.assertEqual(data_id, new_data_id)
        self.assertNotIn(new_data_id, LazyResult.data_cache)

        new_column.evaluate()

        self.assertEqual(1, len(column.expr.context))
        replaced_placeholder = new_column.expr.context.values()[0]
        evaluate_weld_input_id = new_column.expr.context.keys()[0]
        self.is_raw(replaced_placeholder)
        self.assertEqual(weld_input_id, new_weld_input_id)
        self.assertEqual(new_weld_input_id, evaluate_weld_input_id)
        self.assertIn(data_id, LazyResult.data_cache)
        self.is_raw(LazyResult.data_cache[data_id])
        np.testing.assert_array_equal(replaced_placeholder, LazyResult.data_cache[data_id])

    def test_operation_column_ds(self):
        self._test_operation_column(self.ds.variables['tg'], '2f')

    def test_operation_column_df_netcdf4(self):
        self._test_operation_column(self.df_netcdf4['tg'], '2f')

    def test_operation_column_df_csv(self):
        self._test_operation_column(self.df_csv['x'], '2.0')

    # tests a column which relies on both raw and lazy input
    def _test_evaluate_column_mixed_df(self, column):
        self.assertEqual(2, len(column.expr.context))
        data_ids = column.expr.context.values()
        weld_input_ids = column.expr.context.keys()
        # new array must be raw, one from file a placeholder; don't really like this code
        placeholder_index = 0
        raw_index = 1
        if isinstance(data_ids[raw_index], (str, unicode)):
            raw_index = 0
            placeholder_index = 1

        self.is_raw(data_ids[raw_index])
        self.is_placeholder(data_ids[placeholder_index])
        self.assertNotIn(data_ids[placeholder_index], LazyResult.data_cache)

        column.evaluate()

        self.assertEqual(2, len(column.expr.context))
        new_data_ids = column.expr.context.values()
        new_weld_input_ids = column.expr.context.keys()
        [self.is_raw(k) for k in new_data_ids]
        self.assertIn(data_ids[placeholder_index], LazyResult.data_cache)
        self.assertEqual(weld_input_ids, new_weld_input_ids)

    def test_operation_column_mixed_df_netcdf4(self):
        raw_as_series = pdw.Series(self.raw_data30, np.dtype(np.float32), pdw.RangeIndex(0, 30, 1))
        self.df_netcdf4['mixed'] = self.df_netcdf4['tg'] + raw_as_series

        self._test_evaluate_column_mixed_df(self.df_netcdf4['mixed'])

    def test_operation_column_mixed_df_csv(self):
        raw_as_series = pdw.Series(self.raw_data4, np.dtype(np.float64), pdw.RangeIndex(0, 4, 1))
        self.df_csv['mixed'] = self.df_csv['x'] + raw_as_series

        self._test_evaluate_column_mixed_df(self.df_csv['mixed'])

    def _test_head_column(self, column, expected_result, n):
        self.assertEqual(1, len(column.expr.context))
        data_id = column.expr.context.values()[0]
        weld_input_id = column.expr.context.keys()[0]
        self.assertNotIn(data_id, LazyResult.data_cache)
        self.is_placeholder(data_id)

        data = column.head(n)

        np.testing.assert_array_equal(expected_result, data)

        self.assertEqual(1, len(column.expr.context))
        new_data_id = column.expr.context.values()[0]
        new_weld_input_id = column.expr.context.keys()[0]
        # note data is not cached
        self.assertNotIn(data_id, LazyResult.data_cache)
        # note the context was only updated temporarily to read the data
        self.is_placeholder(new_data_id)
        self.assertEqual(weld_input_id, new_weld_input_id)

    def test_head_column_ds(self):
        column = self.ds.variables['tg']
        expected_result = np.array([-99.99, 10., 10.099999, -99.99, -99.99], dtype=np.float32)
        self._test_head_column(column, expected_result, 5)

    def test_head_column_df_netcdf4(self):
        column = self.df_netcdf4['tg']
        expected_result = np.array([-99.99, 10., 10.099999, -99.99, -99.99], dtype=np.float32)
        self._test_head_column(column, expected_result, 5)

    def test_head_column_df_csv(self):
        column = self.df_csv['x']
        expected_result = np.array([-5.958191, -5.95224, -5.9950867], dtype=np.float64)
        self._test_head_column(column, expected_result, 3)

    def _test_lazy_slice_rows_df(self, column, expected_read_data, expected_result, slice_):
        self.assertEqual(1, len(column.expr.context))
        data_id = column.expr.context.values()[0]
        weld_input_id = column.expr.context.keys()[0]
        self.assertNotIn(data_id, LazyResult.data_cache)
        self.is_placeholder(data_id)

        new_column = column[slice_]

        self.assertEqual(1, len(column.expr.context))
        new_data_id = new_column.expr.context.values()[0]
        new_weld_input_id = new_column.expr.context.keys()[0]
        self.is_placeholder(new_data_id)
        self.assertEqual(weld_input_id, new_weld_input_id)
        self.assertEqual(data_id, new_data_id)
        self.assertNotIn(new_data_id, LazyResult.data_cache)

        result = new_column.evaluate()

        self.assertEqual(1, len(column.expr.context))
        evaluated_data = new_column.expr.context.values()[0]
        evaluated_weld_input_id = new_column.expr.context.keys()[0]
        self.is_raw(evaluated_data)
        self.assertEqual(weld_input_id, new_weld_input_id)
        self.assertEqual(new_weld_input_id, evaluated_weld_input_id)
        self.assertIn(data_id, LazyResult.data_cache)
        self.is_raw(LazyResult.data_cache[data_id])

        np.testing.assert_array_equal(evaluated_data, LazyResult.data_cache[data_id])
        np.testing.assert_array_equal(expected_read_data, evaluated_data)
        np.testing.assert_array_equal(expected_result, result)

    def test_lazy_slice_rows_df_netcdf4(self):
        column = self.df_netcdf4['tg']
        # netcdf4 cannot read exactly 5 elements from a file with dimensions (1, 3, 2); it must read 6
        # however, the implementation of eager_read filters the extra out before returning
        expected_read_data = np.array([-99.99, 10., 10.099999, -99.99, -99.99], dtype=np.float32)
        expected_result = np.array([-99.99, 10., 10.099999, -99.99, -99.99], dtype=np.float32)
        self._test_lazy_slice_rows_df(column, expected_read_data, expected_result, slice(5))

    def test_lazy_slice_rows_df_csv(self):
        column = self.df_csv['x']
        # same as expected_result
        expected_read_data = np.array([-5.958191, -5.95224, -5.9950867], dtype=np.float64)
        self._test_lazy_slice_rows_df(column, expected_read_data, expected_read_data, slice(3))

    def _test_lazy_skip_columns_df(self, df, skipped_column_name, kept_column_name):
        skipped_column = df[skipped_column_name].expr.context.values()[0]
        kept_column = df[kept_column_name].expr.context.values()[0]

        self.assertNotIn(skipped_column, LazyResult.data_cache)
        self.assertNotIn(kept_column, LazyResult.data_cache)

        new_df = df.drop(skipped_column_name)
        new_df.evaluate()

        # original placeholders still mapped
        self.is_placeholder(skipped_column)
        self.is_placeholder(kept_column)
        # only one value is cached, so only one is read
        self.assertNotIn(skipped_column, LazyResult.data_cache)
        self.assertIn(kept_column, LazyResult.data_cache)
        # old df context changed for tg_ext
        new_data_id_tg = df[skipped_column_name].expr.context.values()[0]
        new_data_id_tg_ext = df[kept_column_name].expr.context.values()[0]
        self.assertEqual(skipped_column, new_data_id_tg)
        self.assertNotEqual(kept_column, str(new_data_id_tg_ext))
        # and new df context changed
        new_data_id_tg_ext = new_df[kept_column_name]
        self.assertNotEqual(kept_column, new_data_id_tg_ext)

    def test_lazy_skip_columns_df_netcdf4(self):
        self._test_lazy_skip_columns_df(self.df_netcdf4, 'tg', 'tg_ext')

    def test_lazy_skip_columns_df_csv(self):
        self._test_lazy_skip_columns_df(self.df_csv, 'x', 'y')

    def test_subset_raw(self):
        data = np.array([1, 2, 3, 4])
        series = pdw.Series(data, np.dtype(np.int64), pdw.RangeIndex(0, 4, 1))

        series.update_rows(slice(0, 2, 1))

        expected_result = np.array([1, 2])

        self.assertIsInstance(series.expr, WeldObject)
        np.testing.assert_array_equal(expected_result, series.evaluate())


def main():
    unittest.main()


if __name__ == '__main__':
    main()
