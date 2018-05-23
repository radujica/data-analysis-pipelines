#!/bin/bash

path="$HOME2"
path+="/datasets/ECAD/original/small_sample/"
pipenv run python pipeline.py --path $path
