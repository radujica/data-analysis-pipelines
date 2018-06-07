#!/bin/bash

path=${HOME2}
# the python implementation is the ground truth
file1=${path}"/datasets/ECAD/original/small_sample/output/python-libraries/agg.csv"
file2=${path}"/datasets/ECAD/original/small_sample/output/weld/agg.csv"
pipenv run python check_correctness.py ${file1} ${file2}
