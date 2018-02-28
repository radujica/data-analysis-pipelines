import unittest
import numpy as np
from weld.types import WeldLong, WeldVec
from pandas_weld.core.indexes.multi import cartesian_product_indices, duplicate_elements_indices, \
    duplicate_array_indices

# TODO: implement cartesian product tests starting from weld objects


class CartesianProductTests(unittest.TestCase):
    @staticmethod
    def test_correct_raw_data_2_inputs():
        array1_dtype = np.dtype(np.int32)
        array1 = np.array([1, 2], dtype=array1_dtype)
        array2_dtype = np.dtype(np.int32)
        array2 = np.array([3, 5], dtype=array2_dtype)

        result = cartesian_product_indices([array1, array2], [array1_dtype, array2_dtype])

        expected_array1_result = np.array([0, 0, 1, 1], dtype=np.int64)
        expected_array2_result = np.array([0, 1, 0, 1], dtype=np.int64)

        np.testing.assert_array_equal(result[0].evaluate(WeldVec(WeldLong()), verbose=False),
                                      expected_array1_result,
                                      "First array result is incorrect")
        np.testing.assert_array_equal(result[1].evaluate(WeldVec(WeldLong()), verbose=False),
                                      expected_array2_result,
                                      "Second array result is incorrect")

    @staticmethod
    def test_correct_raw_data_3_inputs():
        array1_dtype = np.dtype(np.int32)
        array1 = np.array([1, 2], dtype=array1_dtype)
        array2_dtype = np.dtype(np.int32)
        array2 = np.array([3, 5], dtype=array2_dtype)
        array3_dtype = np.dtype(np.int32)
        array3 = np.array([4, 6], dtype=array1_dtype)

        result = cartesian_product_indices([array1, array2, array3],
                                           [array1_dtype, array2_dtype, array3_dtype])

        expected_array1_result = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        expected_array2_result = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
        expected_array3_result = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)

        np.testing.assert_array_equal(result[0].evaluate(WeldVec(WeldLong()), verbose=False),
                                      expected_array1_result,
                                      "First array result is incorrect")
        np.testing.assert_array_equal(result[1].evaluate(WeldVec(WeldLong()), verbose=False),
                                      expected_array2_result,
                                      "Second array result is incorrect")
        np.testing.assert_array_equal(result[2].evaluate(WeldVec(WeldLong()), verbose=False),
                                      expected_array3_result,
                                      "Third array result is incorrect")

    @staticmethod
    def test_correct_raw_data_different_size_first():
        array1_dtype = np.dtype(np.int32)
        array1 = np.array([1, 2], dtype=array1_dtype)
        array2_dtype = np.dtype(np.int32)
        array2 = np.array([3, 5, 7], dtype=array2_dtype)

        result = cartesian_product_indices([array1, array2], [array1_dtype, array2_dtype])

        expected_array1_result = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        expected_array2_result = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)

        np.testing.assert_array_equal(result[0].evaluate(WeldVec(WeldLong()), verbose=False),
                                      expected_array1_result,
                                      "First array result is incorrect")
        np.testing.assert_array_equal(result[1].evaluate(WeldVec(WeldLong()), verbose=False),
                                      expected_array2_result,
                                      "Second array result is incorrect")

    @staticmethod
    def test_correct_raw_data_different_size_second():
        array1_dtype = np.dtype(np.int32)
        array1 = np.array([1, 2, 3], dtype=array1_dtype)
        array2_dtype = np.dtype(np.int32)
        array2 = np.array([4, 5], dtype=array2_dtype)

        result = cartesian_product_indices([array1, array2], [array1_dtype, array2_dtype])

        expected_array1_result = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
        expected_array2_result = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)

        np.testing.assert_array_equal(result[0].evaluate(WeldVec(WeldLong()), verbose=False),
                                      expected_array1_result,
                                      "First array result is incorrect")
        np.testing.assert_array_equal(result[1].evaluate(WeldVec(WeldLong()), verbose=False),
                                      expected_array2_result,
                                      "Second array result is incorrect")

    @staticmethod
    def test_correct_weld_object_input_first():
        pass

    @staticmethod
    def test_correct_weld_object_input_second():
        pass

    @staticmethod
    def test_correct_weld_object_input_both():
        pass


class DuplicateElementsTests(unittest.TestCase):
    @staticmethod
    def test_correct():
        array_dtype = np.dtype(np.int32)
        array = np.array([1, 2, 3], dtype=array_dtype)
        n = 4L

        result = duplicate_elements_indices(array, n, array_dtype)

        expected_array_result = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)

        np.testing.assert_array_equal(result.evaluate(WeldVec(WeldLong()), verbose=False),
                                      expected_array_result)


class DuplicateArrayTests(unittest.TestCase):
    @staticmethod
    def test_correct():
        array_dtype = np.dtype(np.int32)
        array = np.array([1, 2, 3], dtype=array_dtype)
        n = 4L

        result = duplicate_array_indices(array, n, array_dtype)

        expected_array_result = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=np.int64)

        np.testing.assert_array_equal(result.evaluate(WeldVec(WeldLong()), verbose=False),
                                      expected_array_result)


def main():
    unittest.main()


if __name__ == '__main__':
    main()