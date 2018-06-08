#!/bin/bash

path=${HOME2}
input=${path}"/datasets/ECAD/data_0/"
output=${path}"/results/pipelines/data_0/output/python-libraries/"
slice="4718274:9007614"
# pipenv run python pipeline.py --input ${input} --slice ${slice}
pipenv run python pipeline.py --input ${input} --slice ${slice} --output ${output} --check
