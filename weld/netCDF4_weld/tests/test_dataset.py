import unittest
import netCDF4
import netCDF4_weld


def read_dataset():
    return netCDF4.Dataset('sample.nc')


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


def main():
    unittest.main()


if __name__ == '__main__':
    main()
