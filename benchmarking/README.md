# Requirements
- python 3.6
- pipenv
- collectl: http://collectl.sourceforge.net/
- input data downloaded into data_100 folder as seen in prepare_data.sh
- requirements of all desired pipeline runs

# Install
    cd benchmarking
    pipenv install

# Run
    # prepare the data subsets
    ./prepare_data.sh
    # run all benchmarks; check out --help
    pipenv run python run.py
    # check all outputs; check out --help
    pipenv run python check.py
    # generate plots; check out --help
    pipenv run python plot.py

## Folder structure
    # this git repo
    /data-analysis-pipelines
    # data_100, data_1, etc input folders
    /datasets/data_*/
    # pipeline csv outputs for correctness checks: head, agg, result
    /results/pipelines/data_*/output/<pipeline*>/
    # profiling aka collectl outputs as csv
    /results/pipelines/data_*/profile/<pipeline*>/
    # time command's outputs as csv
    /results/pipelines/data_*/time/<pipeline*>/
    # graphs output
    /results/graphs/data_*/
    # weld comparison output folder
    /results/weld/data_*/
