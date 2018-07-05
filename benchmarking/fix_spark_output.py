import argparse
import fnmatch

import os

HOME2 = os.environ.get('HOME2')
if HOME2 is None:
    raise RuntimeError('Cannot find HOME2 environment variable')

parser = argparse.ArgumentParser(description='Fix Spark output')
parser.add_argument('--input',
                    help='Which input to use. If more than 1, separate with comma, e.g. --input data_1,data_25')
parser.add_argument('--runs',
                    help='How many runs per pipeline-input pair. Default=5')
parser.add_argument('--conf',
                    help='If one of spark-single or spark-par')
args = parser.parse_args()

all_inputs = ['data_0', 'data_1', 'data_3', 'data_6', 'data_12', 'data_25', 'data_50', 'data_100']
all_runs = 5
spark_confs = ['spark-single', 'spark-par']
to_fix = ['agg', 'grouped']

inputs = [k for k in args.input.split(',')] if args.input is not None else all_inputs
runs = int(args.runs) if args.runs is not None else all_runs
sparks = [k for k in args.conf.split(',')] if args.conf is not None else spark_confs

# beautiful
for input_ in inputs:
    for spark in sparks:
        for run in range(runs):
            for f in to_fix:
                path = HOME2 + '/results/pipelines/' + input_ + '/output/' + spark + '/' + str(run) + '_' + f

                for file in os.listdir(path):
                    if fnmatch.fnmatch(file, 'part*'):
                        os.rename(path + '/' + file, path + '.csv')
                        os.system('rm -rf ' + path)
