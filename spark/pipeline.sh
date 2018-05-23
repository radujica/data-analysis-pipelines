#!/bin/bash

path="$HOME2"
path+="/datasets/ECAD/original/small_sample/"
# requires spark-submit in PATH, ofc
spark-submit --master local --driver-memory 6g --executor-memory 2g --executor-cores 1 target/scala-2.11/spark-assembly-1.0.jar --path $path --partitions 4
