import argparse

import os

"Script to check any/all benchmark outputs. This script uses the python-libraries output as the ground truth"

HOME2 = '/export/scratch1/radujica'
pipelines_to_check = ['weld', 'julia', 'java', 'spark']
all_inputs = ['data_0', 'data_1', 'data_3', 'data_6', 'data_12', 'data_25', 'data_10', 'data_100']
all_files = ['head.csv', 'agg.csv', 'result.csv']

parser = argparse.ArgumentParser(description='Run Pipelines')
parser.add_argument('--pipeline',
                    help='Which pipeline to check. '
                         'If more than 1, separate with comma, e.g. --pipeline weld,java')
parser.add_argument('--input',
                    help='Which input to check. If more than 1, separate with comma, e.g. --input data_1,data_25')
parser.add_argument('--files',
                    help='Which files to check. If more than 1, separate with comma, e.g. --files head,agg')
args = parser.parse_args()


def check_arg(arg, default):
    if arg is not None:
        return arg.split(',')
    else:
        return default


pipelines = check_arg(args.pipeline, pipelines_to_check)
inputs = check_arg(args.input, all_inputs)
files = check_arg(args.files, all_files)


for input_ in inputs:
    truth_path = HOME2 + '/results/pipelines/' + input_ + '/output/python-libraries/'

    for pipeline in pipelines:
        output_path = HOME2 + '/results/pipelines/' + input_ + '/output/' + pipeline + '/'

        for f in files:
            file1 = truth_path + f
            file2 = output_path + f
            # check_correctness can handle more inputs but oh, well... this looks cleaner here
            command = ['pipenv run python check_correctness.py', file1, file2]

            print('Checking input={} pipeline={} file={}'.format(input_, pipeline, f))
            os.system(' '.join(command))
