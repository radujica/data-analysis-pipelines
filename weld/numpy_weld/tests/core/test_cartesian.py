import unittest
import numpy as np
import numpy_weld as npw


# noinspection PyMethodMayBeStatic
class CartesianProductTests(unittest.TestCase):
    def test_correct_raw_data_2_inputs(self):
        array1 = np.array([1, 2], dtype=np.dtype(np.int32))
        array2 = np.array([3, 5], dtype=np.dtype(np.int32))

        result = npw.cartesian_product_indices([array1, array2])

        expected_array1_result = np.array([0, 0, 1, 1], dtype=np.int64)
        expected_array2_result = np.array([0, 1, 0, 1], dtype=np.int64)

        np.testing.assert_array_equal(expected_array1_result,
                                      result[0].evaluate(verbose=False),
                                      "First array result is incorrect")
        np.testing.assert_array_equal(expected_array2_result,
                                      result[1].evaluate(verbose=False),
                                      "Second array result is incorrect")

    def test_correct_raw_data_3_inputs(self):
        array1 = np.array([1, 2], dtype=np.dtype(np.int32))
        array2 = np.array([3, 5], dtype=np.dtype(np.int32))
        array3 = np.array([4, 6], dtype=np.dtype(np.int32))

        result = npw.cartesian_product_indices([array1, array2, array3])

        expected_array1_result = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        expected_array2_result = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
        expected_array3_result = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)

        np.testing.assert_array_equal(expected_array1_result,
                                      result[0].evaluate(verbose=False),
                                      "First array result is incorrect")
        np.testing.assert_array_equal(expected_array2_result,
                                      result[1].evaluate(verbose=False),
                                      "Second array result is incorrect")
        np.testing.assert_array_equal(expected_array3_result,
                                      result[2].evaluate(verbose=False),
                                      "Third array result is incorrect")

    def test_correct_raw_data_different_size_first(self):
        array1 = np.array([1, 2], dtype=np.dtype(np.int32))
        array2 = np.array([3, 5, 7], dtype=np.dtype(np.int32))

        result = npw.cartesian_product_indices([array1, array2])

        expected_array1_result = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        expected_array2_result = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)

        np.testing.assert_array_equal(expected_array1_result,
                                      result[0].evaluate(verbose=False),
                                      "First array result is incorrect")
        np.testing.assert_array_equal(expected_array2_result,
                                      result[1].evaluate(verbose=False),
                                      "Second array result is incorrect")

    def test_correct_raw_data_different_size_second(self):
        array1 = np.array([1, 2, 3], dtype=np.dtype(np.int32))
        array2 = np.array([4, 5], dtype=np.dtype(np.int32))

        result = npw.cartesian_product_indices([array1, array2])

        expected_array1_result = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        expected_array2_result = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)

        np.testing.assert_array_equal(expected_array1_result,
                                      result[0].evaluate(verbose=False),
                                      "First array result is incorrect")
        np.testing.assert_array_equal(expected_array2_result,
                                      result[1].evaluate(verbose=False),
                                      "Second array result is incorrect")

    def test_correct_lazy_data_input_first(self):
        array1 = np.array([1, 2, 3], dtype=np.dtype(np.int32))
        array1 = npw.duplicate_elements_indices(array1, 2L)
        array2 = np.array([4, 5], dtype=np.dtype(np.int32))

        result = npw.cartesian_product_indices([array1, array2])

        expected_array1_result = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5], dtype=np.int64)
        expected_array2_result = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)

        np.testing.assert_array_equal(expected_array1_result,
                                      result[0].evaluate(verbose=False),
                                      "First array result is incorrect")
        np.testing.assert_array_equal(expected_array2_result,
                                      result[1].evaluate(verbose=False),
                                      "Second array result is incorrect")

    def test_correct_lazy_data_input_second(self):
        array1 = np.array([1, 2, 3], dtype=np.dtype(np.int32))
        array2 = np.array([4, 5], dtype=np.dtype(np.int32))
        array2 = npw.duplicate_elements_indices(array2, 2L)

        result = npw.cartesian_product_indices([array1, array2])

        expected_array1_result = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
        expected_array2_result = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)

        np.testing.assert_array_equal(expected_array1_result,
                                      result[0].evaluate(verbose=False),
                                      "First array result is incorrect")
        np.testing.assert_array_equal(expected_array2_result,
                                      result[1].evaluate(verbose=False),
                                      "Second array result is incorrect")

    def test_correct_lazy_data_input_both(self):
        array1 = np.array([1, 2, 3], dtype=np.dtype(np.int32))
        array1 = npw.duplicate_elements_indices(array1, 2L)
        array2 = np.array([4, 5], dtype=np.dtype(np.int32))
        array2 = npw.duplicate_elements_indices(array2, 2L)

        result = npw.cartesian_product_indices([array1, array2])

        # this was incorrect before; if one duplicate_elements([0, 0], 2L) it results in [0, 0, 0, 0]
        # as values, however the indices are in fact [0, 0, 1, 1] which is what we requested
        expected_array1_result = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5,
                                           5, 5],
                                          dtype=np.int64)
        expected_array2_result = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
                                          dtype=np.int64)

        np.testing.assert_array_equal(expected_array1_result,
                                      result[0].evaluate(verbose=False),
                                      "First array result is incorrect")
        np.testing.assert_array_equal(expected_array2_result,
                                      result[1].evaluate(verbose=False),
                                      "Second array result is incorrect")


# noinspection PyMethodMayBeStatic
class DuplicateElementsTests(unittest.TestCase):
    def test_correct(self):
        array = np.array([1, 2, 3], dtype=np.dtype(np.int32))
        n = 4L

        result = npw.duplicate_elements_indices(array, n)

        expected_array_result = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)

        np.testing.assert_array_equal(expected_array_result,
                                      result.evaluate(verbose=False))

    def test_correct_lazy_data_input(self):
        array = np.array([1, 2, 3], dtype=np.dtype(np.int32))
        array = npw.duplicate_elements_indices(array, 2L)
        n = 2L

        result = npw.duplicate_elements_indices(array, n)

        expected_array_result = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)

        np.testing.assert_array_equal(expected_array_result,
                                      result.evaluate(verbose=False))


# noinspection PyMethodMayBeStatic
class DuplicateArrayTests(unittest.TestCase):
    def test_correct(self):
        array = np.array([1, 2, 3], dtype=np.dtype(np.int32))
        n = 4L

        result = npw.duplicate_array_indices(array, n)

        expected_array_result = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)

        np.testing.assert_array_equal(expected_array_result,
                                      result.evaluate(verbose=False))

    def test_correct_lazy_data_input(self):
        array = np.array([1, 2, 3], dtype=np.dtype(np.int32))
        array = npw.duplicate_array_indices(array, 2L)
        n = 2L

        result = npw.duplicate_array_indices(array, n)

        expected_array_result = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], dtype=np.int64)

        np.testing.assert_array_equal(expected_array_result,
                                      result.evaluate(verbose=False))


def main():
    unittest.main()


if __name__ == '__main__':
    main()
