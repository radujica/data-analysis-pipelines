#!/bin/bash

input=${HOME2}"/datasets/ECAD/data_0/"
slice="4718274:9007614"
spark-submit --master "local[*]" --conf "spark.sql.shuffle.partitions=4" --driver-memory "10g" \
    target/scala-2.11/spark-assembly-1.0.jar --partitions 4 --input ${input} --slice ${slice}
