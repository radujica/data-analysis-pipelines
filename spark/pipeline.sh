#!/bin/bash

# requires spark-submit in PATH, ofc
spark-submit --master local[4] --driver-memory 6g --executor-memory 2g --executor-cores 1 target/scala-2.11/spark-assembly-1.0.jar 4
