import argparse

import os

"Script to assert weld_compare outputs as equal. Ground truth chosen as Weld with lazy and cache which is already" \
    "checked for correctness against the other pipelines implying the correctness of these here"

HOME2 = os.environ.get('HOME2')
if HOME2 is None:
    raise RuntimeError('Cannot find HOME2 environment variable')

all_inputs = ['data_0', 'data_1', 'data_3', 'data_6', 'data_12', 'data_25', 'data_50', 'data_100']
all_files = ['head.csv', 'agg.csv', 'grouped.csv']
all_runs = [0, 1, 2, 3, 4]
all_experiments = {'lazy': ['eager', 'lazy'],#, 'eager-csv', 'lazy-csv'],
                   'cache': ['no-cache', 'cache'],
                   'ir-cache': ['no-ir-cache', 'ir-cache']}

parser = argparse.ArgumentParser(description='Check Pipelines')
parser.add_argument('--input',
                    help='Which input to check. If more than 1, separate with comma, e.g. --input data_1,data_25')
parser.add_argument('--files',
                    help='Which files to check. If more than 1, separate with comma, e.g. --files head,agg')
parser.add_argument('--runs',
                    help='Which runs to check. If more than 1, separate with comma, e.g. --runs 2,3')
parser.add_argument('--experiment',
                    help='Which experiment to run out of: lazy,cache,ir-cache')

args = parser.parse_args()


def check_arg(arg, default):
    if arg is not None:
        return arg.split(',')
    else:
        return default


inputs = check_arg(args.input, all_inputs)
files = check_arg(args.files, all_files)
runs = [int(i) for i in args.runs.split(',')] if args.runs is not None else all_runs
experiments = args.experiment.split(',') if args.experiment is not None else all_experiments.keys()


# essentially checking if the outputs are the same since check.py checks if they're correct
for input_ in inputs:
    truth_path = HOME2 + '/results/weld/' + input_ + '/lazy/output_lazy_'

    for f in files:
        # picking one of the python output files
        files_ = [truth_path + '0_' + f]
        for run in runs:
            for experiment in experiments:
                for output_name in all_experiments[experiment]:
                    output_path = HOME2 + '/results/weld/' + input_ + '/' + experiment + \
                                  '/output_' + output_name + '_' + str(run) + '_'
                    files_.append(output_path + f)

        print('Checking input={} file={}'.format(input_, f))
        print('----------------')

        command = ['pipenv', 'run', 'python', '-u', 'check_correctness.py'] + files_
        os.system(' '.join(command))

        print('')
