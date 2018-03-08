from unittest import TestLoader, TestSuite, TextTestRunner
from numpy_weld.tests import DuplicateElementsTests, DuplicateArrayTests, CartesianProductTests

# TODO: this could be extended a bit to have such a script in each sub-package
if __name__ == '__main__':
    loader = TestLoader()
    suite = TestSuite((loader.loadTestsFromTestCase(DuplicateElementsTests),
                       loader.loadTestsFromTestCase(DuplicateArrayTests),
                       loader.loadTestsFromTestCase(CartesianProductTests),
                       ))

    runner = TextTestRunner(verbosity=2)
    runner.run(suite)
