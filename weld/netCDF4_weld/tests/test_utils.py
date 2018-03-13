import unittest
from netCDF4_weld.utils import convert_row_to_nd_slices


class UtilsTests(unittest.TestCase):
    def test_convert_to_nd_slices(self):
        dimensions = (2, 3, 2)
        # expected and parameter to function
        pairs_to_test = [((slice(1), slice(1), slice(1)), (slice(1),)),
                         ((slice(1), slice(1), slice(2)), (slice(2),)),
                         ((slice(1), slice(2), slice(2)), (slice(3),)),
                         ((slice(1), slice(2), slice(2)), (slice(4),)),
                         ((slice(1), slice(3), slice(2)), (slice(5),)),
                         ((slice(1), slice(3), slice(2)), (slice(6),)),
                         ((slice(2), slice(3), slice(2)), (slice(7),)),
                         ((slice(2), slice(3), slice(2)), (slice(8),)),
                         ((slice(2), slice(3), slice(2)), (slice(9),)),
                         ((slice(2), slice(3), slice(2)), (slice(10),)),
                         ((slice(2), slice(3), slice(2)), (slice(11),)),
                         ((slice(2), slice(3), slice(2)), (slice(12),)),
                         ((slice(2), slice(3), slice(2)), (slice(13),))]

        for pair in pairs_to_test:
            self.assertTupleEqual(pair[0], convert_row_to_nd_slices(pair[1], dimensions))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
