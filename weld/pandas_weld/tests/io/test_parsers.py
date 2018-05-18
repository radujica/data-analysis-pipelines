import unittest
from datetime import date

import numpy as np
import os

import pandas_weld as pdw
from pandas_weld.tests import test_equal_multiindex


class ParserTests(unittest.TestCase):
    PATH_EXT = (os.path.dirname(__file__)) + '/sample_ext.nc'

    def test_read_netcdf4(self):
        data = {'tg': np.array([-99.99, 10., 10.099999, -99.99, -99.99, 10.2, -99.99, -99.99, -99.99, 10.3, 10.4, 10.5,
                                10.599999, 10.7, 10.8, 10.9, -99.99, -99.99, -99.99, -99.99, 11., 11., 11., 11.,
                                -99.99, -99.99, -99.99, -99.99, 12., 13.],
                               dtype=np.float32),
                'tg_ext': np.array([-9999, 1000., 1010., -9999, -9999, 1020., -9999, -9999, -9999, 1030., 10401.,
                                    10502., 10603., 10704., 10805., 10906., -9999, -9999, -9999, -9999, 11001.,
                                    11002., 11003., 11004., -9999, -9999, -9999, -9999, 12005., 13006.],
                                   dtype=np.float32)}
        index = pdw.MultiIndex.from_product([np.array([25.5, 26.], dtype=np.float32),
                                             np.array([10., 11., 12.], dtype=np.float32),
                                             np.array([str(date(1950, 1, 1)), str(date(1950, 1, 2)),
                                                       str(date(1950, 1, 3)), str(date(1950, 1, 4)),
                                                       str(date(1950, 1, 5))])],
                                            ['longitude', 'latitude', 'time'])
        expected_result = pdw.DataFrame(data, index)

        result = pdw.read_netcdf4(ParserTests.PATH_EXT)

        self.assertListEqual(expected_result.data.keys(), result.data.keys())
        np.testing.assert_array_equal(expected_result.data['tg'], result.data['tg'].evaluate(verbose=False))
        np.testing.assert_array_equal(expected_result.data['tg_ext'], result.data['tg_ext'].evaluate(verbose=False))

        test_equal_multiindex(expected_result.index, result.index)

    # TODO
    def test_read_csv(self):
        pass


def main():
    unittest.main()


if __name__ == '__main__':
    main()
