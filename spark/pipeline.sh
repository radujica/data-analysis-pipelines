#!/bin/bash

input=${HOME2}"/datasets/ECAD/data_0/"
slice="4718274:9007614"
output=${HOME2}"/results/"
spark-submit --master "local[32]" --conf "spark.sql.shuffle.partitions=64" --driver-memory "200g" \
	--conf "spark.eventLog.enabled=True" \
        --conf "spark.eventLog.dir="${output} \
    target/scala-2.11/spark-assembly-1.0.jar --partitions 64 --input ${input} --slice ${slice} --output ${output}
