from unittest import TestLoader, TestSuite, TextTestRunner

from netCDF4_weld.tests import VariableTests, DatasetTests, UtilsTests

if __name__ == '__main__':
    loader = TestLoader()
    suite = TestSuite((loader.loadTestsFromTestCase(VariableTests),
                       loader.loadTestsFromTestCase(DatasetTests),
                       loader.loadTestsFromTestCase(UtilsTests)))

    runner = TextTestRunner(verbosity=2)
    runner.run(suite)
