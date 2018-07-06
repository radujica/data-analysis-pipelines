import argparse

import os

"Script to assert weld_compare outputs as equal. Ground truth chosen as Weld with lazy and cache which is already" \
    "checked for correctness against the other pipelines implying the correctness of these here"

HOME2 = os.environ.get('HOME2')
if HOME2 is None:
    raise RuntimeError('Cannot find HOME2 environment variable')

configurations = ['no-lazy-no-cache', 'no-cache', 'no-lazy', 'all']
all_inputs = ['data_0', 'data_1', 'data_3', 'data_6', 'data_12', 'data_25', 'data_10', 'data_100']
all_files = ['head.csv', 'agg.csv', 'grouped.csv']

parser = argparse.ArgumentParser(description='Check Pipelines')
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


inputs = check_arg(args.input, all_inputs)
files = check_arg(args.files, all_files)


for input_ in inputs:
    truth_path = HOME2 + '/results/weld/' + input_ + '/output_all_'

    for f in files:
        files = [truth_path + f]

        for configuration in configurations[:-1]:
            output_path = HOME2 + '/results/weld/' + input_ + '/output_' + configuration + '_'
            files.append(output_path + f)

        print('Checking input={} file={}'.format(input_, f))

        command = ['pipenv run python check_correctness.py'] + files
        os.system(' '.join(command))
