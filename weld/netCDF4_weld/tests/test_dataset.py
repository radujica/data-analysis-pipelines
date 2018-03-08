import unittest
import netCDF4
import netCDF4_weld
import pandas_weld
import numpy as np
import xarray as xr
import os
from datetime import datetime


class DatasetTests(unittest.TestCase):
    PATH = (os.path.dirname(__file__)) + '/sample.nc'
    PATH_EXT = (os.path.dirname(__file__)) + '/sample_ext.nc'

    @staticmethod
    def _read_dataset_1():
        return netCDF4.Dataset(DatasetTests.PATH)

    @staticmethod
    def _read_dataset_2():
        return netCDF4.Dataset(DatasetTests.PATH_EXT)

    def setUp(self):
        self.ds = netCDF4_weld.Dataset(DatasetTests._read_dataset_1)
        self.ds_ext = netCDF4_weld.Dataset(DatasetTests._read_dataset_2)

    def test_repr(self):
        expected_result = """columns:
\t[u'tg']
dimensions: [u'longitude', u'latitude']"""

        self.assertEqual(expected_result, repr(self.ds))

    def test_evaluate(self):
        pass

    def test_to_dataframe(self):
        data = {'tg': np.array([np.nan, 10, 10.099999, np.nan, np.nan, 10.2], dtype=np.float32)}
        index = pandas_weld.MultiIndex.from_product([np.array([10., 11., 12.], dtype=np.float32),
                                                     np.array([25.5, 26.], dtype=np.float32)],
                                                    ['latitude', 'longitude'])
        expected_result = pandas_weld.DataFrame(data, index)

        result = self.ds.to_dataframe()

        self.assertListEqual(expected_result.data.keys(), result.data.keys())
        np.testing.assert_array_equal(expected_result.data['tg'], result.data['tg'].evaluate(verbose=False))

        self.assertListEqual(expected_result.index.names, result.index.names)
        for i in xrange(2):
            np.testing.assert_array_equal(expected_result.index.levels[i],
                                          result.index.levels[i].evaluate(verbose=False))
            np.testing.assert_array_equal(expected_result.index.labels[i].evaluate(verbose=False),
                                          result.index.labels[i].evaluate(verbose=False))

    def test_to_dataframe_3_indexes(self):
        data = {'tg': np.array([np.nan, np.nan, 10.599999, np.nan, np.nan, 10.0, np.nan, 10.7, np.nan, np.nan,
                                10.099999, np.nan, 10.8, 11.0, np.nan, np.nan, 10.3, 10.9, 11.0, np.nan,
                                np.nan, 10.4, np.nan, 11.0, 12.0, 10.2, 10.5, np.nan, 11.0, 13.0],
                               dtype=np.float32),
                'tg_ext': np.array([np.nan, np.nan, 10603, np.nan, np.nan, 1000, np.nan, 10704, np.nan, np.nan,
                                    1010, np.nan, 10805, 11001, np.nan, np.nan, 1030, 10906, 11002, np.nan,
                                    np.nan, 10401, np.nan, 11003, 12005, 1020, 10502, np.nan, 11004, 13006],
                                   dtype=np.float32)}
        index = pandas_weld.MultiIndex.from_product([np.array([10., 11., 12.], dtype=np.float32),
                                                     np.array([25.5, 26.], dtype=np.float32),
                                                     np.array([datetime(1950, 1, 1), datetime(1950, 1, 2),
                                                               datetime(1950, 1, 3), datetime(1950, 1, 4),
                                                               datetime(1950, 1, 5)])],
                                                    ['latitude', 'longitude', 'time'])
        expected_result = pandas_weld.DataFrame(data, index)

        result = self.ds_ext.to_dataframe()

        self.assertListEqual(expected_result.data.keys(), result.data.keys())
        np.testing.assert_array_equal(expected_result.data['tg'], result.data['tg'].evaluate(verbose=False))
        np.testing.assert_array_equal(expected_result.data['tg_ext'], result.data['tg_ext'].evaluate(verbose=False))

        self.assertListEqual(expected_result.index.names, result.index.names)
        for i in xrange(2):
            np.testing.assert_array_equal(expected_result.index.levels[i],
                                          result.index.levels[i].evaluate(verbose=False))
            np.testing.assert_array_equal(expected_result.index.labels[i].evaluate(verbose=False),
                                          result.index.labels[i].evaluate(verbose=False))

        # date is a special case since xarray makes a DatetimeIndex of Timestamps, but
        # reading from Variable makes an array of Timestamps
        np.testing.assert_array_equal([k.date() for k in expected_result.index.levels[2]],
                                      result.index.levels[2].evaluate(verbose=False))
        np.testing.assert_array_equal(expected_result.index.labels[2].evaluate(verbose=False),
                                      result.index.labels[2].evaluate(verbose=False))

    # i.e. test if result is same as by doing with xarray
    def test_to_dataframe_convention(self):
        expected_result = xr.open_dataset(DatasetTests.PATH).to_dataframe()

        result = self.ds.to_dataframe()

        # data; if adding more columns to sample.nc, test those too here!
        np.testing.assert_array_equal(expected_result['tg'], result['tg'].evaluate(verbose=False))

        # index
        self.assertListEqual(expected_result.index.names, result.index.names)
        for i in xrange(2):
            np.testing.assert_array_equal(expected_result.index.levels[i],
                                          result.index.levels[i].evaluate(verbose=False))
            np.testing.assert_array_equal(expected_result.index.labels[i],
                                          result.index.labels[i].evaluate(verbose=False))

    def test_to_dataframe_convention_3_indexes(self):
        expected_result = xr.open_dataset(DatasetTests.PATH_EXT).to_dataframe()

        result = self.ds_ext.to_dataframe()

        # data; if adding more columns to sample_ext.nc, test those too here!
        np.testing.assert_array_equal(expected_result['tg'], result['tg'].evaluate(verbose=False))
        np.testing.assert_array_equal(expected_result['tg_ext'], result['tg_ext'].evaluate(verbose=False))

        # index
        self.assertListEqual(expected_result.index.names, result.index.names)
        for i in xrange(2):
            np.testing.assert_array_equal(expected_result.index.levels[i],
                                          result.index.levels[i].evaluate(verbose=False))
            np.testing.assert_array_equal(expected_result.index.labels[i],
                                          result.index.labels[i].evaluate(verbose=False))

        # date is a special case since xarray makes a DatetimeIndex of Timestamps, but
        # reading from Variable makes an array of Timestamps
        np.testing.assert_array_equal([k.date() for k in expected_result.index.levels[2]],
                                      result.index.levels[2].evaluate(verbose=False))
        np.testing.assert_array_equal(expected_result.index.labels[2],
                                      result.index.labels[2].evaluate(verbose=False))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
