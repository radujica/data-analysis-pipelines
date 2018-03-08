import unittest
import numpy as np
import numpy_weld as npw


class CartesianProductTests(unittest.TestCase):
    @staticmethod
    def test_correct_raw_data_2_inputs():
        array1_dtype = np.dtype(np.int32)
        array1 = np.array([1, 2], dtype=array1_dtype)
        array2_dtype = np.dtype(np.int32)
        array2 = np.array([3, 5], dtype=array2_dtype)

        result = npw.cartesian_product_indices([array1, array2])

        expected_array1_result = np.array([0, 0, 1, 1], dtype=np.int64)
        expected_array2_result = np.array([0, 1, 0, 1], dtype=np.int64)

        np.testing.assert_array_equal(expected_array1_result,
                                      result[0].evaluate(verbose=False),
                                      "First array result is incorrect")
        np.testing.assert_array_equal(expected_array2_result,
                                      result[1].evaluate(verbose=False),
                                      "Second array result is incorrect")

    @staticmethod
    def test_correct_raw_data_3_inputs():
        array1_dtype = np.dtype(np.int32)
        array1 = np.array([1, 2], dtype=array1_dtype)
        array2_dtype = np.dtype(np.int32)
        array2 = np.array([3, 5], dtype=array2_dtype)
        array3_dtype = np.dtype(np.int32)
        array3 = np.array([4, 6], dtype=array3_dtype)

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

    @staticmethod
    def test_correct_raw_data_different_size_first():
        array1_dtype = np.dtype(np.int32)
        array1 = np.array([1, 2], dtype=array1_dtype)
        array2_dtype = np.dtype(np.int32)
        array2 = np.array([3, 5, 7], dtype=array2_dtype)

        result = npw.cartesian_product_indices([array1, array2])

        expected_array1_result = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        expected_array2_result = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)

        np.testing.assert_array_equal(expected_array1_result,
                                      result[0].evaluate(verbose=False),
                                      "First array result is incorrect")
        np.testing.assert_array_equal(expected_array2_result,
                                      result[1].evaluate(verbose=False),
                                      "Second array result is incorrect")

    @staticmethod
    def test_correct_raw_data_different_size_second():
        array1_dtype = np.dtype(np.int32)
        array1 = np.array([1, 2, 3], dtype=array1_dtype)
        array2_dtype = np.dtype(np.int32)
        array2 = np.array([4, 5], dtype=array2_dtype)

        result = npw.cartesian_product_indices([array1, array2])

        expected_array1_result = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        expected_array2_result = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)

        np.testing.assert_array_equal(expected_array1_result,
                                      result[0].evaluate(verbose=False),
                                      "First array result is incorrect")
        np.testing.assert_array_equal(expected_array2_result,
                                      result[1].evaluate(verbose=False),
                                      "Second array result is incorrect")

    @staticmethod
    def test_correct_lazy_data_input_first():
        array1_dtype = np.dtype(np.int32)
        array1 = np.array([1, 2, 3], dtype=array1_dtype)
        array1 = npw.duplicate_elements_indices(array1, 2L)
        array2_dtype = np.dtype(np.int32)
        array2 = np.array([4, 5], dtype=array2_dtype)

        result = npw.cartesian_product_indices([array1, array2])

        expected_array1_result = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
        expected_array2_result = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)

        np.testing.assert_array_equal(expected_array1_result,
                                      result[0].evaluate(verbose=False),
                                      "First array result is incorrect")
        np.testing.assert_array_equal(expected_array2_result,
                                      result[1].evaluate(verbose=False),
                                      "Second array result is incorrect")

    @staticmethod
    def test_correct_lazy_data_input_second():
        array1_dtype = np.dtype(np.int32)
        array1 = np.array([1, 2, 3], dtype=array1_dtype)
        array2_dtype = np.dtype(np.int32)
        array2 = np.array([4, 5], dtype=array2_dtype)
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

    @staticmethod
    def test_correct_lazy_data_input_both():
        array1_dtype = np.dtype(np.int32)
        array1 = np.array([1, 2, 3], dtype=array1_dtype)
        array1 = npw.duplicate_elements_indices(array1, 2L)
        array2_dtype = np.dtype(np.int32)
        array2 = np.array([4, 5], dtype=array2_dtype)
        array2 = npw.duplicate_elements_indices(array2, 2L)

        result = npw.cartesian_product_indices([array1, array2])

        expected_array1_result = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
                                          dtype=np.int64)
        expected_array2_result = np.array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
                                          dtype=np.int64)

        np.testing.assert_array_equal(expected_array1_result,
                                      result[0].evaluate(verbose=False),
                                      "First array result is incorrect")
        np.testing.assert_array_equal(expected_array2_result,
                                      result[1].evaluate(verbose=False),
                                      "Second array result is incorrect")


class DuplicateElementsTests(unittest.TestCase):
    @staticmethod
    def test_correct():
        array_dtype = np.dtype(np.int32)
        array = np.array([1, 2, 3], dtype=array_dtype)
        n = 4L

        result = npw.duplicate_elements_indices(array, n)

        expected_array_result = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)

        np.testing.assert_array_equal(expected_array_result,
                                      result.evaluate(verbose=False))

    @staticmethod
    def test_correct_lazy_data_input():
        array_dtype = np.dtype(np.int32)
        array = np.array([1, 2, 3], dtype=array_dtype)
        array = npw.duplicate_elements_indices(array, 2L)
        n = 2L

        result = npw.duplicate_elements_indices(array, n)

        expected_array_result = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)

        np.testing.assert_array_equal(expected_array_result,
                                      result.evaluate(verbose=False))


class DuplicateArrayTests(unittest.TestCase):
    @staticmethod
    def test_correct():
        array_dtype = np.dtype(np.int32)
        array = np.array([1, 2, 3], dtype=array_dtype)
        n = 4L

        result = npw.duplicate_array_indices(array, n)

        expected_array_result = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)

        np.testing.assert_array_equal(expected_array_result,
                                      result.evaluate(verbose=False))

    @staticmethod
    def test_correct_lazy_data_input():
        array_dtype = np.dtype(np.int32)
        array = np.array([1, 2, 3], dtype=array_dtype)
        array = npw.duplicate_array_indices(array, 2L)
        n = 2L

        result = npw.duplicate_array_indices(array, n)

        expected_array_result = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5], dtype=np.int64)

        np.testing.assert_array_equal(expected_array_result,
                                      result.evaluate(verbose=False))
