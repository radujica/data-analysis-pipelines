import unittest
import netCDF4
import netCDF4_weld
import numpy as np
import os
from netCDF4_weld.utils import convert_row_to_nd_slices


def read_dataset():
    return netCDF4.Dataset((os.path.dirname(__file__))+'/sample.nc')


class VariableTests(unittest.TestCase):
    def setUp(self):
        self.ds = netCDF4_weld.Dataset(read_dataset)
        self.variable = self.ds.variables['tg']

    # mostly to test if it doesn't crash
    def test_repr(self):
        expected_repr = """variable: tg, dtype: float32
    dimensions: (u'latitude', u'longitude')
    attributes: OrderedDict([(u'_FillValue', -9999), (u'units', u'Celsius'), (u'scale_factor', 0.01)])
    expression: tg"""

        self.assertEqual(expected_repr, repr(self.variable))

    def test_evaluate(self):
        expected_result = np.array([-99.99, 10., 10.099999, -99.99, -99.99, 10.2], dtype=np.float32)

        np.testing.assert_array_equal(expected_result, self.variable.evaluate(verbose=False))

    def test_head(self):
        expected_result = np.array([-99.99, 10.], dtype=np.float32)

        np.testing.assert_array_equal(expected_result, self.variable.head(2))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
