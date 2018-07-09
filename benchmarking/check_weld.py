import argparse

import os

"Script to assert weld_compare outputs as equal. Ground truth chosen as Weld with lazy and cache which is already" \
    "checked for correctness against the other pipelines implying the correctness of these here"

HOME2 = os.environ.get('HOME2')
if HOME2 is None:
    raise RuntimeError('Cannot find HOME2 environment variable')

configurations = ['no-lazy-no-cache', 'no-cache', 'no-lazy', 'all']
all_inputs = ['data_0', 'data_1', 'data_3', 'data_6', 'data_12', 'data_25', 'data_50', 'data_100']
all_files = ['head.csv', 'agg.csv', 'grouped.csv']
all_runs = [0, 1, 2, 3, 4]

parser = argparse.ArgumentParser(description='Check Pipelines')
parser.add_argument('--input',
                    help='Which input to check. If more than 1, separate with comma, e.g. --input data_1,data_25')
parser.add_argument('--files',
                    help='Which files to check. If more than 1, separate with comma, e.g. --files head,agg')
parser.add_argument('--runs',
                    help='Which runs to check. If more than 1, separate with comma, e.g. --runs 2,3')
args = parser.parse_args()
runs = [int(i) for i in args.runs.split(',')] if args.runs is not None else all_runs


def check_arg(arg, default):
    if arg is not None:
        return arg.split(',')
    else:
        return default


inputs = check_arg(args.input, all_inputs)
files = check_arg(args.files, all_files)


# essentially checking if the outputs are the same since check.py checks if they're correct
for input_ in inputs:
    truth_path = HOME2 + '/results/weld/' + input_ + '/output_all_'

    for f in files:
        # picking one of the python output files
        files = [truth_path + '0_' + f]
        for run in runs:
            for configuration in configurations[:-1]:
                output_path = HOME2 + '/results/weld/' + input_ + '/output_' + configuration + '_' + str(run) + '_'
                files.append(output_path + f)

        print('Checking input={} file={}'.format(input_, f))
        print('----------------')
        command = ['pipenv', 'run', 'python', '-u', 'check_correctness.py'] + files
        os.system(' '.join(command))
        print('')
