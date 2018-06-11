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
    # run all benchmarks; check out --help on how to run only a selection of them
    pipenv run python run.py

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
    # weld comparison folder with profile/ and time/
    /results/weld/
