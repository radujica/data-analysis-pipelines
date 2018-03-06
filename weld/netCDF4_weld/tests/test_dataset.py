import unittest
import netCDF4
import netCDF4_weld
import pandas_weld
import numpy as np
import xarray as xr
import os


class DatasetTests(unittest.TestCase):
    @staticmethod
    def _get_path():
        return (os.path.dirname(__file__))+'/sample.nc'

    @staticmethod
    def _read_dataset():
        return netCDF4.Dataset(DatasetTests._get_path())

    def setUp(self):
        self.ds = netCDF4_weld.Dataset(DatasetTests._read_dataset)

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

    # TODO
    def test_to_dataframe_3_indexes(self):
        pass

    # i.e. test if result is same as by doing with xarray
    def test_to_dataframe_convention(self):
        expected_result = xr.open_dataset(DatasetTests._get_path()).to_dataframe()

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

    # TODO
    def test_to_dataframe_convention_3_indexes(self):
        pass


def main():
    unittest.main()


if __name__ == '__main__':
    main()
