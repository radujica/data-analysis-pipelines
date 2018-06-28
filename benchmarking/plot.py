import argparse

import matplotlib.pyplot as plt
import os
import pandas as pd

HOME2 = os.environ.get('HOME2')
if HOME2 is None:
    raise RuntimeError('Cannot find HOME2 environment variable')

OUTPUT_FOLDER = HOME2 + '/results/graphs'

all_pipelines = ['python-libraries', 'weld', 'julia', 'java', 'spark']
all_inputs = ['data_0', 'data_1', 'data_3', 'data_6', 'data_12', 'data_25', 'data_10', 'data_100']


parser = argparse.ArgumentParser(description='Plot results')
parser.add_argument('--input',
                    help='Which input to use. If more than 1, separate with comma, e.g. --input data_1,data_25')
parser.add_argument('--plot', help='Choose a specific plot out of: time_bars, profile_scatter')
parser.add_argument('--save',
                    dest='save',
                    action='store_true',
                    help='If desiring to save the plots, else show')
args = parser.parse_args()


def read_time_input_df(input_, pipeline):
    input_path = HOME2 + '/results/pipelines/' + input_ + '/time/' + pipeline + '/time.csv'
    df = pd.read_csv(input_path)
    # keep track of which pipeline the data refers to
    df['pipeline'] = pipeline

    return df


def plot_time_bars_single(input_):
    dfs = [read_time_input_df(input_, pipeline) for pipeline in all_pipelines]

    # combine into 1 df
    df = dfs[0]
    for d in dfs[1:]:
        df = df.append(d)

    # fix index
    df = df.reset_index().drop(columns='index')

    plt.figure()
    plt.subplots_adjust(hspace=0.4)

    plt.subplot(211)
    p1 = plt.bar(df.index.values, df['sys'])
    p2 = plt.bar(df.index.values, df['user'], bottom=df['sys'])
    plt.ylabel('Time (s)')
    plt.title('Time to run pipeline for input={}'.format(input_))
    plt.xticks(df.index.values, df['pipeline'])
    plt.legend((p1[0], p2[0]), ('sys', 'user'))

    plt.subplot(212)
    p3 = plt.bar(df.index.values, df['real'])
    plt.ylabel('Time (s)')
    plt.title('Time to run pipeline for input={}'.format(input_))
    plt.xticks(df.index.values, df['pipeline'])
    plt.legend((p3,), ('real',))

    if args.save:
        plt.savefig(OUTPUT_FOLDER + '/' + input_ + '/time_bars.png')
    else:
        plt.show()


def plot_time_bars(inputs):
    for input_ in inputs:
        # make sure directory exists
        os.system('mkdir -p ' + OUTPUT_FOLDER + '/' + input_)

        plot_time_bars_single(input_)


def read_collectl_input_df(input_, pipeline):
    input_path = HOME2 + '/results/pipelines/' + input_ + '/profile/' + pipeline + '/profile.csv'
    df = pd.read_csv(input_path, skiprows=range(0, 14))
    # keep track of which pipeline the data refers to
    df['pipeline'] = pipeline

    return df


def plot_profile_scatter_single(input_):
    dfs = {pipeline: read_collectl_input_df(input_, pipeline) for pipeline in all_pipelines}

    plt.figure()
    [plt.plot(df.index.values, df['[MEM]Used'], label=name) for name, df in dfs.items()]
    plt.ylabel('Memory (B)')
    plt.xlabel('Time (s)')
    plt.title('Memory usage over input={}'.format(input_))
    plt.legend()

    if args.save:
        plt.savefig(OUTPUT_FOLDER + '/' + input_ + 'profile_scatter.png')
    else:
        plt.show()


def plot_profile_scatter(inputs):
    for input_ in inputs:
        # make sure directory exists
        os.system('mkdir -p ' + OUTPUT_FOLDER + '/' + input_)

        plot_profile_scatter_single(input_)


# all available plots
all_plots = {'time_bars': plot_time_bars, 'profile_scatter': plot_profile_scatter}

if args.plot is not None:
    plots = {k: all_plots[k] for k in args.plot.split(',')}
else:
    plots = all_plots

if args.input is not None:
    inputs = args.input.split(',')
else:
    inputs = all_inputs

# call the functions to plot the required plots
for plot_func in plots.values():
    plot_func(inputs)
