#!/bin/bash

path=${HOME2}
input=${path}"/datasets/ECAD/data_1/"
output=${path}"/results/pipelines/data_1/output/julia/"
slice="12021312:24749760"
# julia pipeline.jl --input ${input} --slice ${slice}
julia pipeline.jl --input ${input} --slice ${slice} --output ${output} --check
