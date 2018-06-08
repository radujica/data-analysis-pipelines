#!/bin/bash

path=${HOME2}
input=${path}"/datasets/ECAD/data_0/"
output=${path}"/results/pipelines/data_0/output/julia/"
slice="4718274:9007614"
# julia pipeline.jl --input ${input} --slice ${slice}
julia pipeline.jl --input ${input} --slice ${slice} --output ${output} --check
