#!/bin/bash

path=${HOME2}
input=${path}"/datasets/ECAD/original/small_sample/"
output=${path}"/datasets/ECAD/original/small_sample/output/weld/"
pipenv run python pipeline.py --input ${input}
# pipenv run python pipeline.py --input ${input} --output ${output} --check
