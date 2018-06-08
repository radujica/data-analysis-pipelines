#!/bin/bash

# make sure we use latest code
sbt assembly

path=${HOME2}
input=${path}"/datasets/ECAD/data_1/"
output=${path}"/results/pipelines/data_1/output/spark/"
slice="12021312:24749760"
# requires spark-submit in PATH, ofc
# TODO: spark goes wrong somewhere, takes waaay too lon; perhaps memory is the problem though data_1 is only 700MB
# --executor-memory 2g --executor-cores 1 seem unnecessary since executors live in the same JVM process as the driver
# spark-submit --master local[4] --driver-memory 10g target/scala-2.11/spark-assembly-1.0.jar --input ${input} --partitions 4 --slice ${slice}
spark-submit --master local[4] --driver-memory 10g target/scala-2.11/spark-assembly-1.0.jar --input ${input} --partitions 4 --slice ${slice} --output ${output} --check
