#!/bin/bash

input=${HOME2}"/datasets/ECAD/data_0/"
slice="4718274:9007614"
output=${HOME2}"/results/"
profiler=${HOME2}"/statsd-jvm-profiler/target/statsd-jvm-profiler-2.1.1-SNAPSHOT-jar-with-dependencies.jar"
javaagent="-javaagent:"${profiler}"=server=localhost,port=8086,reporter=InfluxDBReporter,database=sparkprofiler,username=profiler,password=profiler,prefix=SparkPipelineSingle"
pipelinepath=${HOME2}"/data-analysis-pipelines/spark/target/scala-2.11/spark-assembly-1.0.jar"

spark-submit --master "local[32]" \
        --conf "spark.sql.shuffle.partitions=64" \
	--conf "spark.executor.heartbeatInterval=115" \
	--conf "spark.executor.extraJavaOptions="${javaagent} \
        --conf "spark.driver.extraJavaOptions="${javaagent} \
	--conf "spark.eventLog.enabled=True" \
	--conf "spark.eventLog.dir="${output} \
       	--driver-memory "200g" \
	--jars ${profiler} \
    ${pipelinepath} --partitions 64 --input ${input} --slice ${slice} --output ${output}

pipenv run python -u ${HOME2}"/statsd-jvm-profiler/visualization/influxdb_dump.py" -o "localhost" -u "profiler" -p "profiler" \
	-d "sparkprofiler" -e "SparkPipelineSingle" > spark-single-stack-traces

perl -w ${HOME2}/FlameGraph/flamegraph.pl spark-single-stack-traces > spark-single-graph.svg


