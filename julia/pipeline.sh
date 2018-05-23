#!/bin/bash

path="$HOME2"
path+="/datasets/ECAD/original/small_sample/"
julia pipeline.jl --path ${path}
