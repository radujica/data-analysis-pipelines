#!/bin/bash

# i.e, prepare these (sub)sets:
# 1950-01-01    1950-07-13  0.78125%    194days
# 1950-01-01    1951-01-23  1.5625%     388days
# 1950-01-01    1952-02-15  3.125%      776days
# 1950-01-01    1954-04-01  6.25%       1552days
# 1950-01-01    1958-07-01  12.5%       3104days
# 1950-01-01    1966-12-31  25%         6209days
# 1950-01-01    1983-12-31  50%         12418days
# 1950-01-01    2017-12-31  100%        24837days

# unzip all, select required subsets and create required data1.nc and data2.nc for each subset

path=${HOME2}
# all files should be downloaded from the E-OBS dataset v17 into data_100;
# for an overview of which files get combined how, check preprocess_data.py
input=${path}"/datasets/ECAD/data_100/"
output=${path}"/datasets/ECAD/"

# rename by removing _0.25etcetc from files
for f in ${input}*.nc.gz; do mv ${f} ${f%_0.25deg_reg_v17.0.nc.gz}.nc.gz; done
echo "renamed all"
# unzip everything
gunzip -k ${input}*.nc.gz
echo "unzipped all"

# output folders
declare -a outputs=("${output}data_0/"
                    "${output}data_1/"
                    "${output}data_3/"
                    "${output}data_6/"
                    "${output}data_12/"
                    "${output}data_25/"
                    "${output}data_50/")
start="1950-01-01"
declare -a stops=("1950-07-13"
                  "1951-01-23"
                  "1952-02-15"
                  "1954-04-01"
                  "1958-07-01"
                  "1966-12-31"
                  "1983-12-31")

for ((i=0; i<${#outputs[@]}; i++));
do
    # TODO: should combine the 2 scripts to avoid the intermediate files creation
    mkdir ${outputs[$i]}
    pipenv run python sample_data.py -i ${input} -o ${outputs[$i]} --start ${start} --stop ${stops[$i]}
    echo "created subset: ${outputs[$i]}"
    pipenv run python preprocess_data.py -i ${outputs[$i]}
    echo "combined: ${outputs[$i]}"
    # cleanup all except data1 and 2
    find ${outputs[$i]}. -type f | grep -v 'data1.nc' | grep -v 'data2.nc' | xargs rm
done
