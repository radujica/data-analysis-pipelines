import unittest
import netCDF4
import netCDF4_weld
import pandas_weld
import numpy as np
import os


def read_dataset():
    return netCDF4.Dataset((os.path.dirname(__file__))+'/sample.nc')


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.ds = netCDF4_weld.Dataset(read_dataset)

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


def main():
    unittest.main()


if __name__ == '__main__':
    main()
