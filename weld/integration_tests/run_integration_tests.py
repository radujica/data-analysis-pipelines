from unittest import TestLoader, TestSuite, TextTestRunner

from test_integration_lazy import IntegrationTests

if __name__ == '__main__':
    loader = TestLoader()
    suite = TestSuite((loader.loadTestsFromTestCase(IntegrationTests)))

    runner = TextTestRunner(verbosity=2)
    runner.run(suite)
