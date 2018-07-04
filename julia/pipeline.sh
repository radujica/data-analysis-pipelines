#!/bin/bash

input=${HOME2}"/datasets/ECAD/data_0/"
slice="4718274:9007614"
julia pipeline.jl --input ${input} --slice ${slice}
