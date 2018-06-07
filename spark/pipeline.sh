#!/bin/bash

# make sure we use latest code
sbt assembly

path=${HOME2}
input=${path}"/datasets/ECAD/original/small_sample/"
output=${path}"/datasets/ECAD/original/small_sample/output/"
# requires spark-submit in PATH, ofc
spark-submit --master local --driver-memory 6g --executor-memory 2g --executor-cores 1 target/scala-2.11/spark-assembly-1.0.jar --input ${input} --partitions 4
# spark-submit --master local --driver-memory 6g --executor-memory 2g --executor-cores 1 target/scala-2.11/spark-assembly-1.0.jar --input ${input} --partitions 4 --output ${output} --check
