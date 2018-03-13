from unittest import TestLoader, TestSuite, TextTestRunner
from pandas_weld.tests import MultiIndexTests, DataFrameTests, SeriesTests

# TODO: this could be extended a bit to have such a script in each sub-package
if __name__ == '__main__':
    loader = TestLoader()
    suite = TestSuite((loader.loadTestsFromTestCase(MultiIndexTests),
                       loader.loadTestsFromTestCase(DataFrameTests),
                       loader.loadTestsFromTestCase(SeriesTests)))

    runner = TextTestRunner(verbosity=2)
    runner.run(suite)
