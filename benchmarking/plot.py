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
all_weld_types = ['no-lazy-no-cache', 'no-cache', 'no-lazy', 'all']


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
    df = df.reset_index().drop(columns='index').sort_values(by='real').reset_index()

    plt.figure()
    plt.bar(df.index.values, df['real'] / 60)
    plt.ylabel('Time (min)')
    plt.title('Real time to run pipeline for input={}'.format(input_))
    plt.xticks(df.index.values, df['pipeline'])

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
    df = pd.read_csv(input_path, skiprows=range(0, 15))
    # keep track of which pipeline the data refers to
    df['pipeline'] = pipeline
    df = df[0:len(df) - 2]

    return df


def plot_profile_scatter_single(input_):
    dfs = {pipeline: read_collectl_input_df(input_, pipeline) for pipeline in all_pipelines}

    plt.figure()
    [plt.plot(df.index.values, df['[MEM]Used'] / 1000000, label=name) for name, df in dfs.items()]
    plt.ylabel('Memory (GB)')
    plt.xlabel('Time (s)')
    plt.title('Memory usage over input={}'.format(input_))
    plt.legend()

    if args.save:
        plt.savefig(OUTPUT_FOLDER + '/' + input_ + '/profile_scatter.png')
    else:
        plt.show()


def plot_profile_scatter(inputs):
    for input_ in inputs:
        # make sure directory exists
        os.system('mkdir -p ' + OUTPUT_FOLDER + '/' + input_)

        plot_profile_scatter_single(input_)


def read_weld_time_input_df(input_, pipeline):
    input_path = HOME2 + '/results/weld/' + input_ + '/time_' + pipeline + '.csv'
    df = pd.read_csv(input_path)
    # keep track of which pipeline the data refers to
    df['pipeline'] = pipeline

    # read the compile times
    compile_input_path = HOME2 + '/results/weld/' + input_ + '/compile_' + pipeline + '.txt'
    df_compile = pd.read_csv(compile_input_path, header=None, names=['type'])
    # filter out the markers
    df_compile = df_compile[df_compile['type'].str[0] != '#']
    # split into 2 columns
    df_compile['type'], df_compile['time'] = df_compile['type'].str.split(':', 1).str
    df_compile['time'] = df_compile['time'].apply(pd.to_numeric)
    df_compile = df_compile.groupby('type').sum().transpose().reset_index()
    df = pd.concat([df, df_compile], axis=1)
    df['total_compile'] = df['Python->Weld'] + df['Weld'] + df['Weld compile time'] + df['Weld->Python']

    return df


def plot_weld_time_bars_single(input_):
    dfs = [read_weld_time_input_df(input_, pipeline) for pipeline in all_weld_types]

    # combine into 1 df
    df = dfs[0]
    for d in dfs[1:]:
        df = df.append(d)

    # fix index
    df = df.reset_index().drop(columns='index')

    fig, ax = plt.subplots()
    p1 = plt.bar(df.index.values, df['sys'])
    p2 = plt.bar(df.index.values, df['Weld compile time'], bottom=df['sys'])
    p3 = plt.bar(df.index.values, df['user'] - df['Weld compile time'], bottom=df['Weld compile time'] + df['sys'])
    plt.ylabel('Time (s)')
    plt.title('Time to run pipeline for input={}'.format(input_))
    plt.xticks([])
    plt.legend((p1[0], p2[0], p3[0]), ('sys', 'compile', 'user'))

    # add values as numbers over the bars and celltext
    cell_text = []
    for rect1, rect2, rect3 in zip(p1, p2, p3):
        h1 = rect1.get_height()
        h2 = rect2.get_height()
        h3 = rect3.get_height()
        # total numbers
        total = '{0:.2f}'.format(h1 + h2 + h3)
        ax.text(rect2.get_x() + rect2.get_width() / 3.5, h1 + h2 + h3 + 2, total)
        cell_text.append(['{0:.2f}'.format(h1), '{0:.2f}'.format(h2), '{0:.2f}'.format(h3), total])

    # transpose
    cell_text = list(map(list, zip(*cell_text)))

    # build table with weld compile time
    plt.table(cellText=cell_text,
              rowLabels=['sys', 'Weld compile', 'user - compile', 'total'],
              colLabels=df['pipeline'],
              cellLoc='center',
              loc='bottom')
    plt.subplots_adjust(left=0.2, bottom=0.2)

    if args.save:
        plt.savefig(OUTPUT_FOLDER + '/' + input_ + '/weld_time_bars.png')
    else:
        plt.show()


def plot_weld_time_bars(inputs):
    for input_ in inputs:
        # make sure directory exists
        os.system('mkdir -p ' + OUTPUT_FOLDER + '/' + input_)

        plot_weld_time_bars_single(input_)


def read_weld_collectl_input_df(input_, pipeline):
    input_path = HOME2 + '/results/weld/' + input_ + '/profile_' + pipeline + '.csv'
    df = pd.read_csv(input_path, skiprows=range(0, 15))
    # keep track of which pipeline the data refers to
    df['pipeline'] = pipeline

    return df


def read_weld_markers(input_, pipeline):
    input_path = HOME2 + '/results/weld/' + input_ + '/compile_' + pipeline + '.txt'
    df = pd.read_csv(input_path, header=None, names=['data'])
    # filter out the markers
    df = df[df['data'].str[0] == '#']
    # split the data
    df['data'] = df['data'].str.split(' ', 1).str[1]
    df['time'], df['marker'] = df['data'].str.split('-', 1).str
    df['time'] = df['time'].apply(lambda x: x.split('.', 1)[0])
    del df['data']
    # keep track of which pipeline the data refers to
    df['pipeline'] = pipeline

    return df


def plot_weld_profile_scatter_single(input_):
    dfs = {pipeline: read_weld_collectl_input_df(input_, pipeline) for pipeline in all_weld_types}
    markers = {pipeline: read_weld_markers(input_, pipeline) for pipeline in all_weld_types}

    plt.figure()
    plt.subplots_adjust(hspace=0.6, wspace=0.4)
    plt.suptitle('Weld Memory Usage', fontsize=16)
    # find max y
    max_y = max([df['[MEM]Used'].max() for df in dfs.values()])
    # find max x
    max_x_all = max(dfs['all'].index.values[-1], dfs['no-lazy'].index.values[-1])
    max_x_other = max(dfs['no-lazy-no-cache'].index.values[-1], dfs['no-cache'].index.values[-1])
    # scale factor for B -> GB
    scale = 1000000

    subplots = [221, 222, 223, 224]
    max_xs = [max_x_other, max_x_all, max_x_other, max_x_all]
    for name, df, subplot, max_x in zip(dfs.keys(), dfs.values(), subplots, max_xs):
        plt.subplot(subplot)
        # change to GB
        plt.plot(df.index.values, df['[MEM]Used'] / scale, label=name)
        plt.ylim(0, max_y / scale + 5)
        plt.xlim(xmax=max_x * 1.1)
        # plot markers
        time_index = pd.Index(df['Time'])
        previous_time = "0"
        for m, t in zip(markers[name]['marker'], markers[name]['time']):
            x_val = time_index.get_loc(t)
            if t != previous_time:
                plt.axvline(x=x_val, color='red')
                plt.text(x_val, max_y / scale - 5, m, rotation=90)
            previous_time = t

        plt.ylabel('Memory (GB)')
        plt.xlabel('Time (s)')
        plt.title(name)

    if args.save:
        plt.savefig(OUTPUT_FOLDER + '/' + input_ + '/weld_profile_scatter.png')
    else:
        plt.show()


def plot_weld_profile_scatter(inputs):
    for input_ in inputs:
        # make sure directory exists
        os.system('mkdir -p ' + OUTPUT_FOLDER + '/' + input_)

        plot_weld_profile_scatter_single(input_)


# all available plots
all_plots = {'time_bars': plot_time_bars,
             'profile_scatter': plot_profile_scatter,
             'weld_bars': plot_weld_time_bars,
             'weld_scatter': plot_weld_profile_scatter}

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
