#!/bin/bash

input=${HOME2}"/datasets/ECAD/data_0/"
slice="4718274:9007614"
java -jar build/libs/pipeline.jar --input ${input} --slice ${slice}
