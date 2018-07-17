import argparse

import os

"Script to check any/all benchmark outputs. This script uses the python-libraries output as the ground truth"

HOME2 = os.environ.get('HOME2')
if HOME2 is None:
    raise RuntimeError('Cannot find HOME2 environment variable')

pipelines_to_check = ['weld-single', 'weld-par', 'julia', 'java', 'R', 'spark-single', 'spark-par']
all_inputs = ['data_0', 'data_1', 'data_3', 'data_6', 'data_12', 'data_25', 'data_10', 'data_100']
all_files = ['head.csv', 'agg.csv', 'grouped.csv']
all_runs = [0, 1, 2, 3, 4]

parser = argparse.ArgumentParser(description='Check Pipelines')
parser.add_argument('--pipeline',
                    help='Which pipeline to check. '
                         'If more than 1, separate with comma, e.g. --pipeline weld,java')
parser.add_argument('--input',
                    help='Which input to check. If more than 1, separate with comma, e.g. --input data_1,data_25')
parser.add_argument('--files',
                    help='Which files to check. If more than 1, separate with comma, e.g. --files head.csv,agg.csv')
parser.add_argument('--runs',
                    help='Which runs to check. If more than 1, separate with comma, e.g. --runs 2,3')
args = parser.parse_args()


def check_arg(arg, default):
    if arg is not None:
        return arg.split(',')
    else:
        return default


pipelines = check_arg(args.pipeline, pipelines_to_check)
inputs = check_arg(args.input, all_inputs)
files = check_arg(args.files, all_files)
runs = [int(i) for i in args.runs.split(',')] if args.runs is not None else all_runs


for input_ in inputs:
    truth_path = HOME2 + '/results/pipelines/' + input_ + '/output/python-libraries/'

    for f in files:
        # picking one of the python output files
        filess = [truth_path + '0_' + f]
        for run in runs:
            for pipeline in pipelines:
                # skip spark head files (yea, could rewrite to avoid continue) as they're expected to fail
                # due to different 'first' values in a DataFrame
                if pipeline in ['spark-single', 'spark-par'] and f == 'head.csv':
                    continue

                output_path = HOME2 + '/results/pipelines/' + input_ + '/output/' + pipeline + '/' + str(run) + '_'
                filess.append(output_path + f)

        print('Checking input={} file={}'.format(input_, f))
        print('----------------')

        command = ['pipenv', 'run', 'python', '-u', 'check_correctness.py'] + filess
        os.system(' '.join(command))
        print('')
