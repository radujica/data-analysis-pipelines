#!/bin/bash

input=${HOME2}"/datasets/ECAD/data_0/"
slice="4718274:9007614"
output=${HOME2}"/results/"
Rscript pipeline.R --input ${input} --slice ${slice} --output ${output}
