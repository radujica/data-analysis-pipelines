#!/bin/bash

input=${HOME2}"/datasets/ECAD/data_0/"
slice="4718274:9007614"
pipenv run python pipeline.py --input ${input} --slice ${slice}
