import unittest
import netCDF4_weld
import os


class DatasetTests(unittest.TestCase):
    PATH = os.path.dirname(__file__)

    def setUp(self):
        self.ds = netCDF4_weld.Dataset(DatasetTests.PATH + '/sample.nc')
        self.ds_ext = netCDF4_weld.Dataset(DatasetTests.PATH + '/sample_ext.nc')

    def test_repr(self):
        expected_result = """columns:
\t[u'tg']
dimensions: [u'longitude', u'latitude']"""

        self.assertEqual(expected_result, repr(self.ds))

    def test_evaluate(self):
        pass


def main():
    unittest.main()


if __name__ == '__main__':
    main()
