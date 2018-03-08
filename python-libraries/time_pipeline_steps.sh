#!/bin/bash

testt="pipenv run python -m timeit -n 1"

echo 'READ DATASETS'
$testt 'import pipeline_steps as pipe' 'pipe.read_datasets()'
echo 'CONVERT TO DATAFRAMES'
$testt 'import pipeline_steps as pipe' 'pipe.convert_to_dataframe(pipe._read_datasets())'
echo 'JOIN DATAFRAMES'
$testt 'import pipeline_steps as pipe' 'pipe.join_dataframes(pipe._read_dataframes())'

# TODO: add others
