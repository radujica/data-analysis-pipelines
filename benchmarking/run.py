import argparse
import subprocess

import os

"Script to run any/all benchmarks. This script only generates data"

# nohup pipenv run python run.py &

HOME2 = os.environ.get('HOME2')
if HOME2 is None:
    raise RuntimeError('Cannot find HOME2 environment variable')

all_pipelines = {'python-libraries': ('python-libraries', 'pipenv run python pipeline.py'),
                 'weld-single': ('weld', 'pipenv run python pipeline.py --threads 1'),
                 'weld-multi': ('weld', 'pipenv run python pipeline.py --threads 32'),
                 'julia': ('julia', 'julia pipeline.jl'),
                 'java': ('java', 'java -jar build/libs/pipeline.jar'),
                 'R': ('R', 'R -f arg.R --args'),
                 'spark-single': ('spark', 'spark-submit --master "local[1] --conf "spark.sql.shuffle.partitions=10" --driver-memory 200g \
        target/scala-2.11/spark-assembly-1.0.jar --partitions 10'),
                 'spark-multi': ('spark', 'spark-submit --master "local[32] --conf "spark.sql.shuffle.partitions=320" --driver-memory 200g \
        target/scala-2.11/spark-assembly-1.0.jar --partitions 320')}
number_runs = 3
# to obtain 'west' Europe, given longitude is the first dimension:
# there are 464 unique longitude, 201 unique latitude, and x unique days depending on subset;
# long -10.125 is index 121 and long 17.125 is at index 230;
# therefore, to compute the slice required: 121 * 201 * <n_days> and 231 * 201 * <n_days>;
# the subset is hence approximately 25% of the data
all_inputs = {'data_0': (4718274, 9007614), 'data_1': (9436548, 18015228), 'data_3': (18873096, 36030456),
              'data_6': (37746192, 72060912), 'data_12': (75492384, 144121824), 'data_25': (151009089, 288290079),
              'data_50': (302018178, 576580158), 'data_100': (604060677, 1153206747)}

parser = argparse.ArgumentParser(description='Run Pipelines')
parser.add_argument('--pipeline',
                    help='Which pipeline to run. '
                    'If more than 1, separate with comma, e.g. --pipeline weld,java')
parser.add_argument('--input',
                    help='Which input to use. If more than 1, separate with comma, e.g. --input data_1,data_25')
parser.add_argument('--runs',
                    help='How many runs per pipeline-input pair. Default=3')
args = parser.parse_args()

pipelines = args.pipeline.split(',') if args.pipeline is not None else all_pipelines.keys()
inputs = {k: all_inputs[k] for k in args.input.split(',')} if args.input is not None else all_inputs
runs = int(args.runs) if args.runs is not None else number_runs

# make sure buildable stuff is built
os.system('cd ' + HOME2 + '/data-analysis-pipelines/java' + ' && ' + './gradlew clean jar')
os.system('cd ' + HOME2 + '/data-analysis-pipelines/spark' + ' && ' + 'sbt assembly')

for input_, slice_ in inputs.items():
    for pipeline in pipelines:
        # delete previous data (spark for example complains if file exists)
        os.system('rm -rf ' + HOME2 + '/results/pipelines/' + input_ + '/profile/' + pipeline)
        os.system('rm -rf ' + HOME2 + '/results/pipelines/' + input_ + '/time/' + pipeline)

        # make required directories
        os.system('mkdir -p ' + HOME2 + '/results/pipelines/' + input_ + '/profile/' + pipeline)
        os.system('mkdir -p ' + HOME2 + '/results/pipelines/' + input_ + '/time/' + pipeline)

        for i in range(runs):
            # clear caches; this should work on the cluster
            os.system('sync; echo 3 | sudo /usr/bin/tee /proc/sys/vm/drop_caches > /dev/null 2>&1')

            # setup the time command
            time_path = HOME2 + '/results/pipelines/' + input_ + '/time/' + pipeline + '_' + str(i) + '_time.csv'
            time_command = ['/usr/bin/time', '-a', '-o', time_path, '-f', '%e,%U,%S,%P,%K,%M,%F,%R,%W,%w,%I,%O']

            # setup the command to run a pipeline
            input_path = HOME2 + '/datasets/ECAD/' + input_ + '/'
            output_path = HOME2 + '/results/pipelines/' + input_ + '/output/' + pipeline + '_' + str(i) + '_'
            pipeline_run = all_pipelines[pipeline]
            pipeline_path = HOME2 + '/data-analysis-pipelines/' + pipeline_run[0] + '/' + pipeline_run[1]
            pipeline_command = [pipeline_path,
                                '--input', input_path,
                                '--slice', '{}:{}'.format(slice_[0], slice_[1]),
                                '--output', output_path]

            # add csv header to output file
            time_header = '"real,user,sys,cpu,mem_avg_tot(K),mem_max(K),' \
                          'major_page_faults,minor_page_faults,swaps,voluntary_context_switch,input,output\n"'
            os.system('printf ' + time_header + ' > ' + time_path)

            # start pipeline
            print('Running pipeline={} on input={}'.format(pipeline, input_))
            pipeline_process = subprocess.Popen(time_command + pipeline_command,
                                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # start profiling
            collectl_path = HOME2 + '/results/pipelines/' + input_ + '/profile/' + pipeline + '_' + str(i) + '_profile'
            collectl_process = subprocess.Popen(['collectl', '-scmd', '-P', '-f' + collectl_path,
                                                 '--sep', ',', '--procfilt', str(pipeline_process.pid)],
                                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # wait for pipeline to finish, then sigterm (ctrl-C) profiling
            pipeline_process.wait()
            print('Done')
            collectl_process.terminate()

            # extract and rename the collectl output to csv
            extract_command = ['gunzip', collectl_path + '*.gz']
            os.system(' '.join(extract_command))
            rename_command = ['mv', collectl_path + '*.tab', collectl_path + '.csv']
            os.system(' '.join(rename_command))
