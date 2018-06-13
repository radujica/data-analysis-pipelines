import argparse
import subprocess

import os

"Script to run any/all benchmarks. This script only generates data"

HOME2 = '/export/scratch1/radujica'

all_pipelines = ['python-libraries', 'weld', 'julia', 'java', 'spark']
# to obtain 'west' Europe, given longitude is the first dimension:
# there are 464 unique longitude, 201 unique latitude, and x unique days depending on subset;
# long -10.125 is index 121 and long 17.125 is at index 230;
# therefore, to compute the slice required: 121 * 201 * <n_days> and 231 * 201 * <n_days>
all_inputs = {'data_0': (4718274, 9007614), 'data_1': (9436548, 18015228), 'data_3': (18873096, 36030456),
              'data_6': (37746192, 72060912), 'data_12': (75492384, 144121824), 'data_25': (151009089, 288290079),
              'data_50': (302018178, 576580158), 'data_100': (604060677, 1153206747)}

parser = argparse.ArgumentParser(description='Run Pipelines')
parser.add_argument('--pipeline',
                    help='Which pipeline to run. '
                    'If more than 1, separate with comma, e.g. --pipeline weld,java')
parser.add_argument('--input',
                    help='Which input to use. If more than 1, separate with comma, e.g. --input data_1,data_25')
parser.add_argument('--no-check',
                    dest='check',
                    action='store_false',
                    help='If desiring to skip the correctness runs')
args = parser.parse_args()

if args.pipeline is not None:
    pipelines = args.pipeline.split(',')
else:
    pipelines = all_pipelines

if args.input is not None:
    inputs = {k: all_inputs[k] for k in args.input.split(',')}
else:
    inputs = all_inputs

# make sure buildable stuff is built
# os.system('cd ' + HOME2 + '/data-analysis-pipelines/java' + ' && ' + './gradlew clean jar')
# os.system('cd ' + HOME2 + '/data-analysis-pipelines/spark' + ' && ' + 'sbt assembly')

for input_, slice_ in inputs.items():
    for pipeline in pipelines:
        # delete previous data (spark for example complains if file exists)
        os.system('rm -rf ' + HOME2 + '/results/pipelines/' + input_ + '/profile/' + pipeline)
        os.system('rm -rf ' + HOME2 + '/results/pipelines/' + input_ + '/time/' + pipeline)

        # make required directories
        os.system('mkdir -p ' + HOME2 + '/results/pipelines/' + input_ + '/profile/' + pipeline)
        os.system('mkdir -p ' + HOME2 + '/results/pipelines/' + input_ + '/time/' + pipeline)

        # clear caches; this should work on the cluster
        # os.system('echo 3 | sudo /usr/bin/tee | /proc/sys/vm/drop_caches')

        # setup the time command
        time_path = HOME2 + '/results/pipelines/' + input_ + '/time/' + pipeline + '/time.csv'
        time_command = ['/usr/bin/time', '-a', '-o', time_path, '-f', '%e,%U,%S,%P,%K,%M,%F,%W,%I,%O']

        # setup the command to run a pipeline
        input_path = HOME2 + '/datasets/ECAD/' + input_ + '/'
        # TODO replace with HOME2
        pipeline_path = '/ufs/radujica/Workspace' + '/data-analysis-pipelines/' + pipeline + '/pipeline.sh'
        pipeline_command = ['sh', pipeline_path, input_path, '{}:{}'.format(slice_[0], slice_[1])]

        # add csv header to output file
        time_header = '"real,user,sys,cpu,mem_avg_tot(K),mem_max(K),page_faults,swaps,input,output\n"'
        os.system('printf ' + time_header + ' > ' + time_path)

        # start pipeline
        print('Running pipeline={} on input={}'.format(pipeline, input_))
        pipeline_process = subprocess.Popen(time_command + pipeline_command,
                                            stdout=subprocess.DEVNULL)
        # start profiling
        collectl_path = HOME2 + '/results/pipelines/' + input_ + '/profile/' + pipeline + '/profile'
        collectl_process = subprocess.Popen(['collectl', '-scmd', '-P', '-f' + collectl_path,
                                             '--sep', ',', '--procfilt', str(pipeline_process.pid)],
                                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # wait for pipeline to finish, then sigterm (ctrl-C) profiling
        pipeline_process.wait()
        print('Done')
        collectl_process.terminate()

        # generate output for correctness check
        if args.check:
            output_path = HOME2 + '/results/pipelines/' + input_ + '/output/' + pipeline + '/'
            # delete previous data and make sure directory exists
            os.system('rm -rf ' + output_path)
            os.system('mkdir -p ' + output_path)

            print('Computing output of pipeline={} on input={}'.format(pipeline, input_))
            check_process = subprocess.Popen(pipeline_command + [output_path],
                                             stdout=subprocess.DEVNULL)
            check_process.wait()
            print('Done')
