from unittest import TestLoader, TestSuite, TextTestRunner
from indexes import DuplicateElementsTests, DuplicateArrayTests, CartesianProductTests

if __name__ == '__main__':
    loader = TestLoader()
    suite = TestSuite((loader.loadTestsFromTestCase(DuplicateElementsTests),
                       loader.loadTestsFromTestCase(DuplicateArrayTests),
                       loader.loadTestsFromTestCase(CartesianProductTests),
                       ))

    runner = TextTestRunner(verbosity=2)
    runner.run(suite)
