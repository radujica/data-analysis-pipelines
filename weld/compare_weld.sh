#!/bin/bash

path="$HOME2"
path+="/datasets/ECAD/original/small_sample/"

# no lazy parsing and no caching
#export LAZY_WELD_CACHE='False'
pipenv run time python pipeline.py -f $path --eager e
# lazy parsing and no caching
pipenv run time python pipeline.py -f $path
# no lazy parsing and caching
export LAZY_WELD_CACHE='True'
pipenv run time python pipeline.py -f $path --eager e
# lazy parsing and caching
pipenv run time python pipeline.py -f $path
