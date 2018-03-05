from unittest import TestLoader, TestSuite, TextTestRunner
from netCDF4_weld.tests import VariableTests, DatasetTests

if __name__ == '__main__':
    loader = TestLoader()
    suite = TestSuite((loader.loadTestsFromTestCase(VariableTests),
                       loader.loadTestsFromTestCase(DatasetTests),
                       ))

    runner = TextTestRunner(verbosity=2)
    runner.run(suite)
