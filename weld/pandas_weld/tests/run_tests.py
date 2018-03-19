from unittest import TestLoader, TestSuite, TextTestRunner
from pandas_weld.tests import MultiIndexTests, DataFrameTests, SeriesTests, IndexTests, \
    RangeIndexTests, SubsetTests, ParserTests

# TODO: this could be extended a bit to have such a script in each sub-package
if __name__ == '__main__':
    loader = TestLoader()
    suite = TestSuite((loader.loadTestsFromTestCase(MultiIndexTests),
                       loader.loadTestsFromTestCase(IndexTests),
                       loader.loadTestsFromTestCase(RangeIndexTests),
                       loader.loadTestsFromTestCase(DataFrameTests),
                       loader.loadTestsFromTestCase(SeriesTests),
                       loader.loadTestsFromTestCase(SubsetTests),
                       loader.loadTestsFromTestCase(ParserTests)))

    runner = TextTestRunner(verbosity=2)
    runner.run(suite)
