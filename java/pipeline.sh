#!/bin/bash

# cd in this directory for proper path of pipeline
cd ${BASH_SOURCE%/*}

# run default pipeline if no args passed
if [[ $# -eq 0 ]]
then
    input=${HOME2}"/datasets/ECAD/data_0/"
    slice="4718274:9007614"
    java -jar build/libs/pipeline.jar --input ${input} --slice ${slice}
# if 2 args, they must be input and slice
elif [[ $# -eq 2 ]]
then
    java -jar build/libs/pipeline.jar --input $1 --slice $2
# if 3, the third must be the output for check
elif [[ $# -eq 3 ]]
then
    java -jar build/libs/pipeline.jar --input $1 --slice $2 --check --output $3
else
    echo "wrong number of args"
fi
