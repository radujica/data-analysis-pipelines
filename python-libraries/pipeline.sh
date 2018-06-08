#!/bin/bash

path=${HOME2}
input=${path}"/datasets/ECAD/data_1/"
output=${path}"/results/pipelines/data_1/output/python-libraries/"
slice="12021312:24749760"
# pipenv run python pipeline.py --input ${input} --slice ${slice}
pipenv run python pipeline.py --input ${input} --slice ${slice} --output ${output} --check
