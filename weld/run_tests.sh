#!/bin/bash

pipenv run python -m netCDF4_weld.tests.run_tests
pipenv run python -m pandas_weld.tests.run_tests
pipenv run python -m numpy_weld.tests.run_tests
pipenv run python -m integration_tests.run_integration_tests
