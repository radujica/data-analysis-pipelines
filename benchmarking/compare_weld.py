import argparse
import subprocess

import os
from datetime import datetime

"Script to run all weld variants. This script only generates data"

HOME2 = os.environ.get('HOME2')
if HOME2 is None:
    raise RuntimeError('Cannot find HOME2 environment variable')

PIPELINE_PATH = HOME2 + '/data-analysis-pipelines/weld'

parser = argparse.ArgumentParser(description='Compare Weld')
parser.add_argument('--input',
                    help='Which input to check. If more than 1, separate with comma, e.g. --input data_1,data_25')
parser.add_argument('--runs',
                    help='How many runs per pipeline-input pair. Default=5')
parser.add_argument('--experiment',
                    help='Which experiment to run out of: lazy,cache,ir-cache')
args = parser.parse_args()


def run_pipeline(pipeline_command, name, output_path, run):
    print('{} Running={}. Run={}/{}'.format(str(datetime.now()), name, run + 1, runs))

    output_args = ['--output', output_path + '/output_' + name + '_' + str(run) + '_']

    # clear caches; this should work on the cluster
    os.system('echo 3 | sudo /usr/bin/tee /proc/sys/vm/drop_caches > /dev/null 2>&1')

    # setup the WeldObject compile-etc output and custom markers
    log = open(output_path + '/compile_' + name + '_' + str(run) + '.txt', 'w')

    # setup the time command
    time_path = output_path + '/time_' + name + '_' + str(run) + '.csv'
    time_command = ['/usr/bin/time', '-a', '-o', time_path, '-f', '%e,%U,%S,%P,%K,%M,%F,%R,%W,%w,%I,%O']

    # add csv header to output file
    time_header = '"real,user,sys,cpu,mem_avg_tot(K),mem_max(K),' \
                  'major_page_faults,minor_page_faults,swaps,voluntary_context_switch,input,output\n"'
    os.system('printf ' + time_header + ' > ' + time_path)

    # start pipeline
    os.chdir(PIPELINE_PATH)
    pipeline_process = subprocess.Popen(time_command + pipeline_command + output_args,
                                        stdout=log, stderr=subprocess.DEVNULL)

    # start profiling
    collectl_path = output_path + '/profile'
    collectl_process = subprocess.Popen(['collectl', '-scmd', '-P', '-f' + collectl_path,
                                         '--sep', ',', '--procfilt', str(pipeline_process.pid)],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # wait for pipeline to finish, then sigterm (ctrl-C) profiling
    pipeline_process.wait()
    collectl_process.terminate()

    # make sure log is saved to file
    log.flush()
    os.fsync(log.fileno())
    log.close()

    # extract and rename the collectl output to csv
    # extract_command = ['gunzip', collectl_path + '*.gz']
    # os.system(' '.join(extract_command))
    rename_command = ['mv', collectl_path + '*.tab', collectl_path + '_' + name + '_' + str(run) + '.csv']
    os.system(' '.join(rename_command))

    print('{} Done'.format(str(datetime.now())))


def run_lazy_experiment(command, n_runs, output_path):
    output_path = output_path + '/lazy/'
    # delete previous data and remake directory
    os.system('rm -rf ' + output_path)
    os.system('mkdir -p ' + output_path)

    os.putenv('LAZY_WELD_CACHE', 'False')
    os.putenv('WELD_INPUT_CACHE', 'False')

    for run in range(n_runs):
        # netcdf
        run_pipeline(command + ['--eager'], 'eager', output_path, run)
        run_pipeline(command, 'lazy', output_path, run)

        # # csv
        # run_pipeline(command + ['--eager', '--csv'], 'eager-csv', output_path, run)
        # run_pipeline(command + ['--csv'], 'lazy-csv', output_path, run)


def run_cache_experiment(command, n_runs, output_path):
    output_path = output_path + '/cache/'
    # delete previous data and remake directory
    os.system('rm -rf ' + output_path)
    os.system('mkdir -p ' + output_path)

    os.putenv('LAZY_WELD_CACHE', 'False')

    for run in range(n_runs):
        os.putenv('WELD_INPUT_CACHE', 'False')
        run_pipeline(command, 'no-cache', output_path, run)

        os.putenv('WELD_INPUT_CACHE', 'True')
        run_pipeline(command, 'cache', output_path, run)


def run_ir_cache_experiment(command, n_runs, output_path):
    output_path = output_path + '/ir-cache/'
    # delete previous data and remake directory
    os.system('rm -rf ' + output_path)
    os.system('mkdir -p ' + output_path)

    for run in range(n_runs):
        os.putenv('LAZY_WELD_CACHE', 'False')
        run_pipeline(command, 'no-ir-cache', output_path, run)

        os.putenv('LAZY_WELD_CACHE', 'True')
        run_pipeline(command, 'ir-cache', output_path, run)


all_inputs = {'data_0': (4718274, 9007614), 'data_1': (9436548, 18015228), 'data_3': (18873096, 36030456),
              'data_6': (37746192, 72060912), 'data_12': (75492384, 144121824), 'data_25': (151009089, 288290079),
              'data_50': (302018178, 576580158), 'data_100': (604060677, 1153206747)}
number_runs = 5
all_experiments = {'lazy': run_lazy_experiment, 'cache': run_cache_experiment, 'ir-cache': run_ir_cache_experiment}

inputs = {k: all_inputs[k] for k in args.input.split(',')} if args.input is not None else all_inputs
runs = int(args.runs) if args.runs is not None else number_runs
experiments = args.experiment.split(',') if args.experiment is not None else all_experiments.keys()


for input_, slice_ in inputs.items():
    print('Running on input={}'.format(input_))

    output_folder = HOME2 + '/results/weld/' + input_

    input_path = HOME2 + '/datasets/ECAD/' + input_ + '/'
    base_command = ['pipenv', 'run', 'python', '-u', 'pipeline.py', '--input', input_path,
                    '--slice', '{}:{}'.format(slice_[0], slice_[1])]

    for experiment in experiments:
        experiment_func = all_experiments[experiment]
        experiment_func(base_command, runs, output_folder)
