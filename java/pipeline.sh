#!/bin/bash

# make sure we use latest code
./gradlew clean jar

path="$HOME2"
path+="/datasets/ECAD/original/small_sample/"
java -jar build/libs/pipeline.jar --path ${path}
