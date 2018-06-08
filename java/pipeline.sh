#!/bin/bash

# make sure we use latest code
./gradlew clean jar

path=${HOME2}
input=${path}"/datasets/ECAD/data_1/"
output=${path}"/results/pipelines/data_1/output/java/"
slice="12021312:24749760"
# java -jar build/libs/pipeline.jar --input ${input} --slice ${slice}
java -jar build/libs/pipeline.jar --input ${input} --slice ${slice} --output ${output} --check
