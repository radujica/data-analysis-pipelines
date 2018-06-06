#!/bin/bash

# make sure we use latest code
./gradlew clean jar

path=${HOME2}
input=${path}"/datasets/ECAD/original/small_sample/"
output=${path}"/datasets/ECAD/original/small_sample/output/"
java -jar build/libs/pipeline.jar --input ${input}
# java -jar build/libs/pipeline.jar --input ${input} --output ${output} --check
