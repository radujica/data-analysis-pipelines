#!/bin/bash

# cd in this directory for proper path of pipeline
cd ${BASH_SOURCE%/*}

# --executor-memory 2g --executor-cores 1 seem unnecessary since executors live in the same JVM process as the driver
# TODO: we probably want to restrict to 1 core ~ local for fair comparison

cores=4
partitions=$(( ${cores} * 10 ))
memory="12g"

# run default pipeline if no args passed
if [[ $# -eq 0 ]]
then
    input=${HOME2}"/datasets/ECAD/data_0/"
    slice="4718274:9007614"
    spark-submit --master "local[${cores}]" --conf "spark.sql.shuffle.partitions=${partitions}" --driver-memory ${memory} \
        target/scala-2.11/spark-assembly-1.0.jar --partitions ${partitions} --input ${input} --slice ${slice}
# if 2 args, they must be input and slice
elif [[ $# -eq 2 ]]
then
    spark-submit --master "local[${cores}]" --conf "spark.sql.shuffle.partitions=${partitions}" --driver-memory ${memory} \
        target/scala-2.11/spark-assembly-1.0.jar --partitions ${partitions} --input $1 --slice $2
# if 3, the third must be the output for check
elif [[ $# -eq 3 ]]
then
    spark-submit --master "local[${cores}]" --conf "spark.sql.shuffle.partitions=${partitions}" --driver-memory ${memory} \
        target/scala-2.11/spark-assembly-1.0.jar --partitions ${partitions} --input $1 --slice $2 --check --output $3
else
    echo "wrong number of args"
fi
