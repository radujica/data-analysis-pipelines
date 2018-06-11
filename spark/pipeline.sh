#!/bin/bash

# cd in this directory for proper path of pipeline
cd ${BASH_SOURCE%/*}

# TODO: spark goes wrong somewhere, takes waaay too lon; perhaps memory is the problem though data_1 is only 700MB
# --executor-memory 2g --executor-cores 1 seem unnecessary since executors live in the same JVM process as the driver
# TODO: we probably want to restrict to 1 core ~ local for fair comparison

# run default pipeline if no args passed
if [[ $# -eq 0 ]]
then
    input=${HOME2}"/datasets/ECAD/data_0/"
    slice="4718274:9007614"
    spark-submit --master "local[4]" --driver-memory 10g target/scala-2.11/spark-assembly-1.0.jar --partitions 4 --input ${input} --slice ${slice}
# if 2 args, they must be input and slice
elif [[ $# -eq 2 ]]
then
    spark-submit --master "local[4]" --driver-memory 10g target/scala-2.11/spark-assembly-1.0.jar --partitions 4 --input $1 --slice $2
# if 3, the third must be the output for check
elif [[ $# -eq 3 ]]
then
    spark-submit --master "local[4]" --driver-memory 10g target/scala-2.11/spark-assembly-1.0.jar --partitions 4 --input $1 --slice $2 --check --output $3
else
    echo "wrong number of args"
fi
