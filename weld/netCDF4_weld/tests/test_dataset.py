import unittest
import netCDF4
import netCDF4_weld
import pandas_weld
import numpy as np
import os
from datetime import date


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
        index = pandas_weld.MultiIndex.from_product([np.array([25.5, 26.], dtype=np.float32),
                                                    np.array([10., 11., 12.], dtype=np.float32)],
                                                    ['longitude', 'latitude'])
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
        data = {'tg': np.array([np.nan, 10., 10.099999, np.nan, np.nan, 10.2, np.nan, np.nan, np.nan, 10.3, 10.4, 10.5,
                                10.599999, 10.7, 10.8, 10.9, np.nan, np.nan, np.nan, np.nan, 11., 11., 11., 11.,
                                np.nan, np.nan, np.nan, np.nan, 12., 13.],
                               dtype=np.float32),
                'tg_ext': np.array([np.nan, 1000., 1010., np.nan, np.nan, 1020., np.nan, np.nan, np.nan, 1030., 10401.,
                                    10502., 10603., 10704., 10805., 10906., np.nan, np.nan, np.nan, np.nan, 11001.,
                                    11002., 11003., 11004., np.nan, np.nan, np.nan, np.nan, 12005., 13006.],
                                   dtype=np.float32)}
        index = pandas_weld.MultiIndex.from_product([np.array([25.5, 26.], dtype=np.float32),
                                                     np.array([10., 11., 12.], dtype=np.float32),
                                                     np.array([date(1950, 1, 1), date(1950, 1, 2),
                                                               date(1950, 1, 3), date(1950, 1, 4),
                                                               date(1950, 1, 5)])],
                                                    ['longitude', 'latitude', 'time'])
        expected_result = pandas_weld.DataFrame(data, index)

        result = self.ds_ext.to_dataframe()

        self.assertListEqual(expected_result.data.keys(), result.data.keys())
        np.testing.assert_array_equal(expected_result.data['tg'], result.data['tg'].evaluate(verbose=False))
        np.testing.assert_array_equal(expected_result.data['tg_ext'], result.data['tg_ext'].evaluate(verbose=False))

        self.assertListEqual(expected_result.index.names, result.index.names)
        for i in xrange(3):
            np.testing.assert_array_equal(expected_result.index.levels[i],
                                          result.index.levels[i].evaluate(verbose=False))
            np.testing.assert_array_equal(expected_result.index.labels[i].evaluate(verbose=False),
                                          result.index.labels[i].evaluate(verbose=False))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
