#!/bin/bash

# i.e, prepare the following subsets; percentage is approximate as dates were calculated
# based on months, not days per year, resulting in actual percentage being slightly less than shown
# 1950-01-01    1951-01-16  12.5625m    1.046y  1.5625%     381days
# 1950-01-01    1952-02-04  25.125m     2.093y  3.125%      765days
# 1950-01-01    1954-03-07  50.25m      4.187y  6.25%       1527days
# 1950-01-01    1958-05-16  100.5m      8.37y   12.5%       3058days
# 1950-01-01    1963-04-30  201m        13.3y   25%         4868days
# 1950-01-01    1983-06-30  402m        33.5y   50%         12234days
# 1950-01-01    2017-12-31  804m        67y     100%        24837days

# unzip all, select required subsets and create required data1.nc and data2.nc for each subset
# there are 201 unique latitude, 464 unique longitude, and x unique days depending on subset; lat 42.375 is at index 68
# and lat 60.125 is at index 139; therefore, to compute the slice required: 68 * 464 * <n_days> and 140 * 464 * <n_days>

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
declare -a outputs=("${output}data_1/"
                    "${output}data_3/"
                    "${output}data_6/"
                    "${output}data_12/"
                    "${output}data_25/"
                    "${output}data_50/")
start="1950-01-01"
declare -a stops=("1951-01-16"
                  "1952-02-04"
                  "1954-03-07"
                  "1958-05-16"
                  "1963-04-30"
                  "1983-06-30")

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
