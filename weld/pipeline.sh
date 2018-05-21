#!/bin/bash

path="$HOME2"
path+="/datasets/ECAD/original/small_sample/"
pipenv run time python pipeline.py -f $path
