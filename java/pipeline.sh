#!/bin/bash

# make sure we use latest code
./gradlew clean jar

path=${HOME2}
input=${path}"/datasets/ECAD/data_0/"
output=${path}"/results/pipelines/data_0/output/java/"
slice="4718274:9007614"
# java -jar build/libs/pipeline.jar --input ${input} --slice ${slice}
java -jar build/libs/pipeline.jar --input ${input} --slice ${slice} --output ${output} --check
