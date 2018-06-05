#!/bin/bash

path=${HOME2}
input=${path}"/datasets/ECAD/original/small_sample/"
output=${path}"/datasets/ECAD/original/small_sample/output/"
julia pipeline.jl --input ${input}
# julia pipeline.jl --input ${input} --output ${output} --check
