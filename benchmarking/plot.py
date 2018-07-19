import argparse

import math
import matplotlib.pyplot as plt
import os
import pandas as pd


##### DISCLAIMER #####
# This is by no means a nicely maintained file.
# Many temporary changes (e.g. filter out last 2 values, combine graphs into 1 due to overleaf limit)
# have been done while generating the actual graphs used in the paper.

HOME2 = os.environ.get('HOME2')
if HOME2 is None:
    raise RuntimeError('Cannot find HOME2 environment variable')

OUTPUT_FOLDER = HOME2 + '/results/graphs'

all_pipelines = ['python-libraries', 'weld-single', 'weld-par', 'julia', 'java', 'R', 'spark-single', 'spark-par']
all_inputs = ['data_0', 'data_1', 'data_3', 'data_6', 'data_12', 'data_25', 'data_50', 'data_100']
all_weld_experiments = {'lazy': ['eager', 'lazy'],
                        'cache': ['no-cache', 'cache'],
                        'ir-cache': ['no-ir-cache', 'ir-cache']}
pipelines_colors = {'python-libraries': 'blue',
                    'python': 'blue',
                    'weld-single': 'green',
                    'weld-par': 'limegreen',
                    'julia': 'm',
                    'java': 'red',
                    'R': 'grey',
                    'spark-single': 'darkorange',
                    'spark-par': 'gold'}
# colors: time = b, memory = g, io = m, over-inputs = time (c), mem (y)


parser = argparse.ArgumentParser(description='Plot results')
parser.add_argument('--input',
                    help='Which input to use. If more than 1, separate with comma, e.g. --input data_1,data_25')
parser.add_argument('--plot', help='Choose a specific plot out of: time_bars, profile_scatter')
parser.add_argument('--save',
                    dest='save',
                    action='store_true',
                    help='If desiring to save the plots, else show')
args = parser.parse_args()


def read_time_pipelines(input_, pipeline):
    first_df_path = HOME2 + '/results/pipelines/' + input_ + '/time/' + pipeline + '/0_time.csv'
    df = pd.read_csv(first_df_path)
    for run in range(1, 5):
        input_path = HOME2 + '/results/pipelines/' + input_ + '/time/' + pipeline + '/' + str(run) + '_time.csv'
        df = df.append(pd.read_csv(input_path))

    res = pd.DataFrame({'real_mean': [df['real'].mean()],
                        'real_diff': [(df['real'].max() - df['real'].min()) / 2],
                        'pipeline': [pipeline],
                        'mem_mean': [df['mem_max(K)'].mean()],
                        'mem_diff': [(df['mem_max(K)'].max() - df['mem_max(K)'].min()) / 2]})

    return res


def read_collectl_pipelines_mem(input_, pipeline):
    first_df_path = HOME2 + '/results/pipelines/' + input_ + '/profile/' + pipeline + '/0_profile.csv'
    df = pd.DataFrame({'real_0': pd.read_csv(first_df_path, skiprows=range(0, 15), usecols=['[MEM]Used'])['[MEM]Used'] / 1000000})
    for run in range(1, 5):
        input_path = HOME2 + '/results/pipelines/' + input_ + '/profile/' + pipeline + '/' + str(run) + '_profile.csv'
        df['real_' + str(run)] = pd.read_csv(input_path, skiprows=range(0, 15), usecols=['[MEM]Used'])['[MEM]Used'] / 1000000

    series_mean = df.mean(axis=1)
    df['mem_diff'] = df.std(axis=1)
    df['mem_mean'] = series_mean
    df = df[['mem_mean', 'mem_diff']]

    # keep track of which pipeline the data refers to
    df['pipeline'] = pipeline
    # remove the last 2 values
    df = df[0:len(df) - 2]

    return df


def read_collectl_pipelines_cpu(input_, pipeline):
    first_df_path = HOME2 + '/results/pipelines/' + input_ + '/profile/' + pipeline + '/0_profile.csv'
    df = pd.DataFrame({'real_0': pd.read_csv(first_df_path, skiprows=range(0, 15), usecols=['[CPU]Totl%'])['[CPU]Totl%'] * 32})
    for run in range(1, 5):
        input_path = HOME2 + '/results/pipelines/' + input_ + '/profile/' + pipeline + '/' + str(run) + '_profile.csv'
        df['real_' + str(run)] = pd.read_csv(input_path, skiprows=range(0, 15), usecols=['[CPU]Totl%'])['[CPU]Totl%'] * 32

    series_mean = df.mean(axis=1)
    # standard error of the mean
    df['cpu_diff'] = df.std(axis=1) / math.sqrt(5)
    df['cpu_mean'] = series_mean
    df = df[['cpu_mean', 'cpu_diff']]

    # keep track of which pipeline the data refers to
    df['pipeline'] = pipeline
    # remove the last 2 values
    df = df[0:len(df) - 2]

    return df


def read_collectl_pipelines_time_single(input_, pipeline, run):
    first_df_path = HOME2 + '/results/pipelines/' + input_ + '/profile/' + pipeline + '/' + str(run) + '_profile.csv'
    df = pd.read_csv(first_df_path, skiprows=range(0, 15), usecols=['Time', '[MEM]Used'])

    # # add an extra row same as previous for slight delays in markers
    # last_row = df.tail(1).copy()
    # last_row['Time'] = last_row['Time'].apply(lambda x: x[:6] + str(int(x[6:8]) + 1))
    # df = df.append(last_row, ignore_index=True)

    return df


def read_markers_pipelines_single(input_, pipeline, run):
    input_path = HOME2 + '/results/pipelines/' + input_ + '/markers/' + pipeline + '/' + str(run) + '_markers.txt'
    df = pd.read_csv(input_path, header=None, names=['data'])
    # filter out the markers
    df = df[df['data'].str[0] == '#']
    # split the data
    df['time'], df['marker'] = df['data'].str.split('-', 1).str
    df['time'] = df['time'].apply(lambda x: x[1:])
    del df['data']

    return df


def read_weld_compile_single_run(input_, experiment, experiment_factor, run):
    # read the compile times
    compile_input_path = HOME2 + '/results/weld/' + input_ + '/' + experiment + '/compile_' + experiment_factor + '_' + str(run) + '.txt'
    df = pd.read_csv(compile_input_path, header=None, names=['type'])
    # filter out the markers
    df = df[df['type'].str[0] != '#']
    # split into 2 columns
    df['type'], df['time'] = df['type'].str.split(':', 1).str
    df['time'] = df['time'].apply(pd.to_numeric)
    df = df.groupby('type').sum().transpose().reset_index()
    df['total_convert'] = df['Python->Weld'] + df['Weld->Python']

    return df


def read_weld_compile_single(input_, experiment, experiment_factor):
    df = read_weld_compile_single_run(input_, experiment, experiment_factor, 0)
    for i in range(1, 5):
        df = df.append(read_weld_compile_single_run(input_, experiment, experiment_factor, i))

    df = df[['Weld compile time', 'total_convert', 'Weld']]

    series_mean = df.mean(axis=0)
    df_mean = pd.DataFrame({series_mean.index[i] + '_mean': [series_mean[i]] for i in range(len(series_mean))})
    series_diff = (df.max(axis=0) - df.min(axis=0)) / 2
    df_diff = pd.DataFrame({series_diff.index[i] + '_diff': [series_diff[i]] for i in range(len(series_diff))})

    df = pd.concat([df_mean, df_diff], axis=1)
    df['pipeline'] = experiment_factor.capitalize()

    return df


def read_weld_compile(input_, experiment):
    experiment_factors = all_weld_experiments[experiment]
    df = read_weld_compile_single(input_, experiment, experiment_factors[0])
    df = df.append(read_weld_compile_single(input_, experiment, experiment_factors[1]))

    # fix index
    df = df.reset_index().drop(columns='index')

    return df


def read_weld_time_single(input_, experiment, experiment_factor):
    input_path = HOME2 + '/results/weld/' + input_ + '/' + experiment + '/time_' + experiment_factor + '_0.csv'
    df = pd.read_csv(input_path)

    for i in range(1, 5):
        df2 = pd.read_csv(HOME2 + '/results/weld/' + input_ + '/' + experiment + '/time_' + experiment_factor + '_' + str(i) + '.csv')
        df = df.append(df2)

    df = df[['real', 'mem_max(K)', 'input']]

    series_mean = df.mean(axis=0)
    df_mean = pd.DataFrame({series_mean.index[i] + '_mean': [series_mean[i]] for i in range(len(series_mean))})
    series_diff = (df.max(axis=0) - df.min(axis=0)) / 2
    df_diff = pd.DataFrame({series_diff.index[i] + '_diff': [series_diff[i]] for i in range(len(series_diff))})

    df = pd.concat([df_mean, df_diff], axis=1)
    df['pipeline'] = experiment_factor.capitalize()

    return df


def read_weld_time(input_, experiment):
    experiment_factors = all_weld_experiments[experiment]
    df = read_weld_time_single(input_, experiment, experiment_factors[0])
    df = df.append(read_weld_time_single(input_, experiment, experiment_factors[1]))

    # fix index
    df = df.reset_index().drop(columns='index')

    return df


def read_weld_collectl_single(input_, experiment, experiment_factor, run):
    input_path = HOME2 + '/results/weld/' + input_ + '/' + experiment + '/profile_' + experiment_factor + '_' + str(run) + '.csv'
    df = pd.read_csv(input_path, skiprows=range(0, 15))
    # keep track of which pipeline the data refers to
    df['pipeline'] = experiment_factor.capitalize()

    # try remove the lines going down
    last_index = len(df) - 1
    for i in range(len(df) - 1, 1, -1):
        last_mem = df.iloc[i]['[MEM]Used']
        second_last_mem = df.iloc[i - 1]['[MEM]Used']
        if last_mem < second_last_mem:
            last_index -= 1
        else:
            break

    # but don't actually remove all in case markers are slightly late
    df = df[:last_index + 2]

    return df


def read_weld_markers_single(input_, experiment, experiment_factor, run):
    input_path = HOME2 + '/results/weld/' + input_ + '/' + experiment + '/compile_' + experiment_factor + '_' + str(run) + '.txt'
    df = pd.read_csv(input_path, header=None, names=['data'])
    # filter out the markers
    df = df[df['data'].str[0] == '#']
    # split the data
    df['time'], df['marker'] = df['data'].str.split('-', 1).str
    df['time'] = df['time'].apply(lambda x: x[1:])
    del df['data']
    # keep track of which pipeline the data refers to
    df['pipeline'] = experiment_factor.capitalize()

    return df


def plot_time_bars_single_full(input_):
    pipelines = ['python-libraries', 'weld-single', 'weld-par', 'julia', 'java', 'R', 'spark-single', 'spark-par']
    dfs = [read_time_pipelines(input_, pipeline) for pipeline in pipelines]

    # shorten names
    pipelines_new = ['python', 'weld-s', 'weld-p', 'julia', 'java', 'R', 'spark-s', 'spark-p']
    for i in range(len(pipelines_new)):
        dfs[i]['pipeline'] = pipelines_new[i]

    # combine into 1 df
    df = dfs[0]
    for d in dfs[1:]:
        df = df.append(d)

    # fix index
    df = df.reset_index().drop(columns='index').sort_values(by='real_mean').reset_index().drop(columns='index')

    # set scale
    scale = 60
    df['real_mean'] = df['real_mean'] / scale
    df['real_diff'] = df['real_diff'] / scale

    # remove error bars if too small?
    # df['real_diff'] = df['real_diff'].map(lambda x: 0 if x < 10 else x)

    # plt.figure(figsize=(6, 4))
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex='all')
    ax1.bar(df['pipeline'], df['real_mean'], width=0.7, yerr=df['real_diff'], capsize=2.5, color='royalblue')
    ax2.bar(df['pipeline'], df['real_mean'], width=0.7, yerr=df['real_diff'], capsize=2.5, color='royalblue')

    # set y_axis limits
    spark_single_y = df.loc[df['pipeline'] == 'spark-s']
    spark_par_y = df.loc[df['pipeline'] == 'spark-p']
    ax1.set_ylim(spark_par_y['real_mean'].values - spark_par_y['real_diff'].values - 0.05 * spark_par_y['real_mean'].values,
                 spark_single_y['real_mean'].values + spark_single_y['real_diff'].values + 0.03 * spark_single_y['real_mean'].values)
    julia_y = df.loc[df['pipeline'] == 'julia']
    ax2.set_ylim(0, julia_y['real_mean'].values + julia_y['real_diff'].values + 0.01 * julia_y['real_mean'].values)

    # hide spines
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop='off')
    ax2.xaxis.tick_bottom()

    # diagonal lines
    d = .015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    fig.subplots_adjust(hspace=0.1)
    ax1.set_ylabel('Time (minutes)', fontsize=11)
    ax2.set_ylabel('Time (minutes)', fontsize=11)
    ax1.set_title('Mean time to run pipelines over input={}'.format(input_), fontsize=15)

    if args.save:
        plt.savefig(OUTPUT_FOLDER + '/' + input_ + '/time_bars.png')
    else:
        plt.show()


def plot_time_bars_full(inputs):
    for input_ in inputs:
        # make sure directory exists
        os.system('mkdir -p ' + OUTPUT_FOLDER + '/' + input_)

        plot_time_bars_single_full(input_)


def plot_time_bars_single(input_, pipelines):
    # pipelines = ['python-libraries', 'weld-single', 'weld-par', 'julia', 'java', 'R', 'spark-single', 'spark-par']
    dfs = [read_time_pipelines(input_, pipeline) for pipeline in pipelines]
    renames = {'python-libraries': 'python',
               'weld-single': 'weld-s',
               'weld-par': 'weld-p'}

    # combine into 1 df
    df = dfs[0]
    for d in dfs[1:]:
        df = df.append(d)

    df['pipeline'] = df['pipeline'].map(lambda x: renames[x] if x in renames else x)

    # fix index
    df = df.reset_index().drop(columns='index').sort_values(by='real_mean').reset_index().drop(columns='index')

    # set scale
    scale = 60
    df['real_mean'] = df['real_mean'] / scale
    df['real_diff'] = df['real_diff'] / scale

    # remove error bars if too small?
    # df['real_diff'] = df['real_diff'].map(lambda x: 0 if x < 10 else x)

    plt.bar(df['pipeline'], df['real_mean'], width=0.7, yerr=df['real_diff'], capsize=2.5, color='royalblue')

    plt.ylabel('Time (minutes)', fontsize=10)
    plt.title(input_, fontsize=12)


def plot_time_bars(inputs):
    # make sure directory exists
    os.system('mkdir -p ' + OUTPUT_FOLDER + '/all')

    pipelines = {'data_3': ['python-libraries', 'weld-single', 'weld-par', 'julia', 'java', 'R'],
                 'data_6': ['python-libraries', 'weld-single', 'weld-par', 'julia', 'java', 'R'],
                 'data_12': ['python-libraries', 'weld-single', 'weld-par', 'julia', 'java', 'R'],
                 'data_25': ['python-libraries', 'julia', 'java', 'R']}
    subplots = [221, 222, 223, 224]
    plt.figure(figsize=(12, 8))
    for input_, subpl in zip(pipelines.keys(), subplots):
        plt.subplot(subpl)

        plot_time_bars_single(input_, pipelines[input_])

    plt.subplots_adjust(hspace=0.3, bottom=0.08, top=0.95, left=0.05, right=0.98)

    if args.save:
        plt.savefig(OUTPUT_FOLDER + '/all/time_bars_rest.png')
    else:
        plt.show()


def plot_time_inputs_single(pipeline, inputs):
    dfs = [read_time_pipelines(input_, pipeline) for input_ in inputs]

    # combine into 1 df
    df = dfs[0]
    for d in dfs[1:]:
        df = df.append(d)

    df = df.reset_index()[['real_mean', 'real_diff']]

    plt.bar(df.index.values, df['real_mean'] / 60, yerr=df['real_diff'] / 60, color='c', capsize=4)
    plt.ylabel('Time (minutes)', fontsize=12)
    plt.title(pipeline, fontsize=15)
    plt.xticks(df.index.values, inputs)

    # plot a power curve
    x = df.index.values
    y = [df['real_mean'][0] / 60] * len(x)
    for i in range(1, len(x)):
        y[i] = y[i - 1] * 2
    plt.plot(df.index.values, y, color='r')


def plot_time_inputs(inputs):
    # make sure directory exists
    os.system('mkdir -p ' + OUTPUT_FOLDER + '/all')

    pipelines = {'python-libraries': ['data_0', 'data_1', 'data_3', 'data_6', 'data_12', 'data_25', 'data_50'],
                 'java': ['data_0', 'data_1', 'data_3', 'data_6', 'data_12', 'data_25', 'data_50'],
                 'julia': ['data_0', 'data_1', 'data_3', 'data_6', 'data_12', 'data_25'],
                 'R': ['data_0', 'data_1', 'data_3', 'data_6', 'data_12', 'data_25'],
                 'weld-single': ['data_0', 'data_1', 'data_3', 'data_6', 'data_12'],
                 'weld-par': ['data_0', 'data_1', 'data_3', 'data_6', 'data_12']}
    subplots = [321, 322, 323, 324, 325, 326]
    plt.figure(figsize=(10, 10))

    for pipeline, subpl in zip(pipelines.keys(), subplots):
        plt.subplot(subpl)

        plot_time_inputs_single(pipeline, pipelines[pipeline])

    plt.subplots_adjust(hspace=0.25, bottom=0.05, top=0.95, left=0.07, right=0.98)

    if args.save:
        plt.savefig(OUTPUT_FOLDER + '/all/time_inputs.png')
    else:
        plt.show()


def plot_mem_inputs_single(pipeline, inputs):
    input_sizes = {'data_0': 2.171, 'data_1': 4.342, 'data_3': 8.677, 'data_6': 17.361,
              'data_12': 34.722, 'data_25': 69.457, 'data_50': 138.908}
    dfs = [read_time_pipelines(input_, pipeline) for input_ in inputs]

    # combine into 1 df
    df = dfs[0]
    for d in dfs[1:]:
        df = df.append(d)

    df = df.reset_index()[['mem_mean', 'mem_diff']]

    plt.bar(df.index.values, df['mem_mean'] / 1000000, yerr=df['mem_diff'] / 1000000, color='y')
    plt.ylabel('Memory (GB)', fontsize=12)
    plt.title(pipeline, fontsize=15)
    plt.xticks(df.index.values, inputs)

    # # plot a power curve
    # x = df.index.values
    # y = [df['mem_mean'][0] / 1000000] * len(x)
    # for i in range(1, len(x)):
    #     y[i] = y[i - 1] * 2
    plt.plot(df.index.values, [input_sizes[inp] for inp in inputs], color='r')


def plot_mem_inputs(inputs):
    # make sure directory exists
    os.system('mkdir -p ' + OUTPUT_FOLDER + '/all')

    pipelines = {'python-libraries': ['data_0', 'data_1', 'data_3', 'data_6', 'data_12', 'data_25', 'data_50'],
                 'java': ['data_0', 'data_1', 'data_3', 'data_6', 'data_12', 'data_25', 'data_50'],
                 'julia': ['data_0', 'data_1', 'data_3', 'data_6', 'data_12', 'data_25'],
                 'R': ['data_0', 'data_1', 'data_3', 'data_6', 'data_12', 'data_25'],
                 'weld-single': ['data_0', 'data_1', 'data_3', 'data_6', 'data_12'],
                 'weld-par': ['data_0', 'data_1', 'data_3', 'data_6', 'data_12']}
    subplots = [321, 322, 323, 324, 325, 326]
    plt.figure(figsize=(10, 10))

    for pipeline, subpl in zip(pipelines.keys(), subplots):
        plt.subplot(subpl)

        plot_mem_inputs_single(pipeline, pipelines[pipeline])

    plt.subplots_adjust(hspace=0.25, bottom=0.05, top=0.95, left=0.07, right=0.98)

    if args.save:
        plt.savefig(OUTPUT_FOLDER + '/all/mem_inputs.png')
    else:
        plt.show()


def plot_profile_scatter_single(input_, pipelines):
    dfs = {pipeline: read_collectl_pipelines_mem(input_, pipeline) for pipeline in pipelines}
    dfs['python'] = dfs['python-libraries']
    del dfs['python-libraries']

    [plt.plot(df.index.values, df['mem_mean'], '-', label=name, color=pipelines_colors[name])
     for name, df in dfs.items()]
    [plt.fill_between(df.index.values, df['mem_mean'] - df['mem_diff'], df['mem_mean'] + df['mem_diff'], alpha=0.3, facecolor=pipelines_colors[name])
     for name, df in dfs.items()]
    plt.ylabel('Memory (GB)', fontsize=10)
    plt.xlabel('Time (s)', fontsize=10)
    plt.title(input_, fontsize=14)
    plt.legend(fontsize=10, labelspacing=0.05)


def plot_profile_scatter(inputs):
    # make sure directory exists
    os.system('mkdir -p ' + OUTPUT_FOLDER + '/all')

    pipelines = {'data_0': ['python-libraries', 'weld-single', 'weld-par', 'julia', 'java', 'R'],
                 'data_1': ['python-libraries', 'weld-single', 'weld-par', 'julia', 'java', 'R'],
                 'data_3': ['python-libraries', 'weld-single', 'weld-par', 'julia', 'java', 'R'],
                 'data_6': ['python-libraries', 'weld-single', 'weld-par', 'julia', 'java', 'R'],
                 'data_12': ['python-libraries', 'weld-single', 'weld-par', 'julia', 'java', 'R'],
                 'data_25': ['python-libraries', 'julia', 'java', 'R']}
    subplots = [321, 322, 323, 324, 325, 326]
    plt.figure(figsize=(12, 12))

    for input_, subpl in zip(pipelines.keys(), subplots):
        plt.subplot(subpl)

        plot_profile_scatter_single(input_, pipelines[input_])

    plt.subplots_adjust(hspace=0.3, bottom=0.05, top=0.95, left=0.05, right=0.98)

    if args.save:
        plt.savefig(OUTPUT_FOLDER + '/all/profile_scatter.png')
    else:
        plt.show()


def plot_profile_scatter_single_cpu(input_, pipelines):
    dfs = {pipeline: read_collectl_pipelines_cpu(input_, pipeline) for pipeline in pipelines}
    dfs['python'] = dfs['python-libraries']
    del dfs['python-libraries']

    max_x = max([df.index.values[-1] for df in dfs.values()])
    [plt.plot(df.index.values, df['cpu_mean'], '-', label=name, color=pipelines_colors[name])
     for name, df in dfs.items()]
    [plt.fill_between(df.index.values, df['cpu_mean'] - df['cpu_diff'], df['cpu_mean'] + df['cpu_diff'], alpha=0.3, facecolor=pipelines_colors[name])
     for name, df in dfs.items()]
    plt.ylabel('CPU (%)', fontsize=12)
    plt.xlabel('Time (s)', fontsize=12)
    plt.title(input_, fontsize=12)
    plt.legend(fontsize=10, labelspacing=0.05, loc='upper right')

    plt.axhline(y=100, color='k', linewidth=0.5)
    plt.text(max_x - 3, 130, '100')


def plot_profile_scatter_cpu(inputs):
    # make sure directory exists
    os.system('mkdir -p ' + OUTPUT_FOLDER + '/all')

    pipelines = {'data_0': ['python-libraries', 'weld-single', 'weld-par', 'julia', 'java', 'R'],
                 'data_1': ['python-libraries', 'weld-single', 'weld-par', 'julia', 'java', 'R'],
                 'data_3': ['python-libraries', 'weld-single', 'weld-par', 'julia', 'java', 'R'],
                 'data_6': ['python-libraries', 'weld-single', 'weld-par', 'julia', 'java', 'R']}
    subplots = [221, 222, 223, 224]
    plt.figure(figsize=(10, 6))

    for input_, subpl in zip(pipelines.keys(), subplots):
        plt.subplot(subpl)

        plot_profile_scatter_single_cpu(input_, pipelines[input_])

    plt.subplots_adjust(hspace=0.4, wspace=0.3, bottom=0.1, top=0.95, left=0.1, right=0.95)

    if args.save:
        plt.savefig(OUTPUT_FOLDER + '/all/profile_scatter_cpu.png')
    else:
        plt.show()


def plot_profile_scatter_markers_single(input_, pipeline):
    df = read_collectl_pipelines_time_single(input_, pipeline, 2)
    df_markers = read_markers_pipelines_single(input_, pipeline, 2)

    # plt.figure(figsize=(10, 4))
    plt.plot(df.index.values, df['[MEM]Used'] / 1000000)
    plt.ylabel('Memory (GB)', fontsize=10)
    plt.xlabel('Time (s)', fontsize=10)
    plt.title(pipeline, fontsize=12)

    # find max y
    max_y = df['[MEM]Used'].max()

    # plot markers
    time_index = pd.Index(df['Time'])
    previous_time = "0"
    for m, t in zip(df_markers['marker'], df_markers['time']):
        x_val = time_index.get_loc(t)
        if t != previous_time:
            plt.axvline(x=x_val, color='red')
            plt.text(x_val + 0.005 * df.index.values[-1], max_y / 1000000, m, rotation=90, fontsize=8)
        previous_time = t


def plot_profile_markers(inputs):
    # make sure directory exists
    os.system('mkdir -p ' + OUTPUT_FOLDER + '/all')

    subplot_h = 6
    subplot_v = 2
    subplot_index = 1

    plt.figure(figsize=(12, 16))

    pipelines = ['python-libraries', 'weld-single', 'weld-par', 'julia', 'R', 'java']
    for pipeline in pipelines:
        for input_ in ['data_1', 'data_12']:
            plt.subplot(subplot_h, subplot_v, subplot_index)
            subplot_index += 1

            plot_profile_scatter_markers_single(input_, pipeline)

    plt.subplots_adjust(hspace=0.6, bottom=0.05, top=0.95, left=0.05, right=0.95)

    if args.save:
        plt.savefig(OUTPUT_FOLDER + '/all/profile_markers.png')
    else:
        plt.show()


def plot_profile_markers_spark(inputs):
    # make sure directory exists
    os.system('mkdir -p ' + OUTPUT_FOLDER + '/all')

    subplots = [211, 212]

    plt.figure(figsize=(6, 5))

    pipelines = ['spark-single', 'spark-par']
    for pipeline, subpl in zip(pipelines, subplots):
        plt.subplot(subpl)

        plot_profile_scatter_markers_single('data_1', pipeline)

    plt.subplots_adjust(hspace=0.6)

    if args.save:
        plt.savefig(OUTPUT_FOLDER + '/all/profile_markers_spark.png')
    else:
        plt.show()


def plot_weld_time_bars_single(input_, experiment):
    df = read_weld_time(input_, experiment)
    df_compile = read_weld_compile(input_, experiment)
    df = pd.merge(df, df_compile)

    plt.subplots(figsize=(5, 5))
    p1 = plt.bar(df.index.values, df['Weld_mean'], width=0.6)
    p2 = plt.bar(df.index.values, df['Weld compile time_mean'], bottom=df['Weld_mean'], width=0.6)
    p3 = plt.bar(df.index.values, df['total_convert_mean'],
                 bottom=df['Weld compile time_mean'] + df['Weld_mean'], width=0.6)
    p4 = plt.bar(df.index.values, df['real_mean'] - df['Weld compile time_mean'] - df['total_convert_mean'] - df['Weld_mean'],
                 bottom=df['total_convert_mean'] + df['Weld compile time_mean'] + df['Weld_mean'],
                 yerr=df['real_diff'], capsize=4, width=0.6)
    plt.ylabel('Time (s)')
    plt.title('Time to run pipeline for input={}'.format(input_))
    plt.ylim(ymax=(df['real_mean']).max() + 10)
    plt.xticks([])
    plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Weld', 'compile', 'en-/decode', 'python'))

    # add celltext
    cell_text = []
    for rect1, rect2, rect3, rect4 in zip(p1, p2, p3, p4):
        h1 = rect1.get_height()
        h2 = rect2.get_height()
        h3 = rect3.get_height()
        h4 = rect4.get_height()
        # total numbers
        total = '{0:.2f}'.format(h1 + h2 + h3 + h4)
        # ax.text(rect2.get_x() + rect2.get_width() / 3.5, h1 + h2 + h3 + h4 + 2, total)
        cell_text.append(['{0:.2f}'.format(h1), '{0:.2f}'.format(h2), '{0:.2f}'.format(h3), '{0:.2f}'.format(h4), total])

    # transpose
    cell_text = list(map(list, zip(*cell_text)))

    # build table with weld compile time
    table = plt.table(cellText=cell_text,
                      rowLabels=['Weld', 'compile', 'en-/decode', 'python', 'total'],
                      colLabels=df['pipeline'],
                      cellLoc='center',
                      loc='bottom',
                      colWidths=[0.6] * 2)
    table.scale(1, 2)
    plt.subplots_adjust(left=0.3, bottom=0.32)

    if args.save:
        plt.savefig(OUTPUT_FOLDER + '/' + input_ + '/' + experiment + '/time_bars.png')
    else:
        plt.show()


def plot_weld_time_bars(inputs):
    for input_ in inputs:
        # make sure directory exists
        os.system('mkdir -p ' + OUTPUT_FOLDER + '/' + input_)

        for experiment in all_weld_experiments.keys():
            # make sure directory exists
            os.system('mkdir -p ' + OUTPUT_FOLDER + '/' + input_ + '/' + experiment)

            plot_weld_time_bars_single(input_, experiment)


def plot_weld_time_lazy_single(input_):
    df = read_weld_time(input_, 'lazy')
    df_compile = read_weld_compile(input_, 'lazy')
    df1 = pd.merge(df, df_compile)
    df = read_weld_time(input_, 'cache')
    df_compile = read_weld_compile(input_, 'cache')
    df2 = pd.merge(df, df_compile)
    df = pd.concat([df1, df2])
    df = df.loc[df['pipeline'] != 'No-cache'].reset_index().drop(columns="index")

    plt.subplots(figsize=(7, 5))
    p1 = plt.bar(df.index.values, df['Weld_mean'], width=0.7)
    p2 = plt.bar(df.index.values, df['Weld compile time_mean'], bottom=df['Weld_mean'], width=0.7)
    p3 = plt.bar(df.index.values, df['total_convert_mean'],
                 bottom=df['Weld compile time_mean'] + df['Weld_mean'], width=0.7)
    p4 = plt.bar(df.index.values, df['real_mean'] - df['Weld compile time_mean'] - df['total_convert_mean'] - df['Weld_mean'],
                 bottom=df['total_convert_mean'] + df['Weld compile time_mean'] + df['Weld_mean'],
                 yerr=df['real_diff'], capsize=4, width=0.7)
    plt.ylabel('Time (s)')
    plt.title('Time to run pipeline for input={}'.format(input_))
    plt.ylim(ymax=(df['real_mean']).max() + 10)
    plt.xticks([])
    plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Weld', 'compile', 'en-/decode', 'python'))

    # add celltext
    cell_text = []
    for rect1, rect2, rect3, rect4 in zip(p1, p2, p3, p4):
        h1 = rect1.get_height()
        h2 = rect2.get_height()
        h3 = rect3.get_height()
        h4 = rect4.get_height()
        # total numbers
        total = '{0:.2f}'.format(h1 + h2 + h3 + h4)
        # ax.text(rect2.get_x() + rect2.get_width() / 3.5, h1 + h2 + h3 + h4 + 2, total)
        cell_text.append(['{0:.2f}'.format(h1), '{0:.2f}'.format(h2), '{0:.2f}'.format(h3), '{0:.2f}'.format(h4), total])

    # transpose
    cell_text = list(map(list, zip(*cell_text)))

    # build table with weld compile time
    table = plt.table(cellText=cell_text,
                      rowLabels=['Weld', 'compile', 'en-/decode', 'python', 'total'],
                      colLabels=['Eager', 'Lazy', 'Lazy w/ Cache'],
                      cellLoc='center',
                      loc='bottom',
                      colWidths=[0.333333] * 3)
    table.scale(1, 2)
    plt.subplots_adjust(left=0.21, bottom=0.32)

    if args.save:
        plt.savefig(OUTPUT_FOLDER + '/' + input_ + '/lazy_time_bars.png')
    else:
        plt.show()


def plot_weld_time_lazy(inputs):
    for input_ in inputs:
        # make sure directory exists
        os.system('mkdir -p ' + OUTPUT_FOLDER + '/' + input_)

        plot_weld_time_lazy_single(input_)


def plot_weld_profile_scatter_single(input_, experiment):
    experiment_factors = all_weld_experiments[experiment]
    dfs = {experiment_factor: read_weld_collectl_single(input_, experiment, experiment_factor, 2)
           for experiment_factor in experiment_factors}
    markers = {experiment_factor: read_weld_markers_single(input_, experiment, experiment_factor, 2)
               for experiment_factor in experiment_factors}

    plt.figure(figsize=(6, 5))
    plt.subplots_adjust(hspace=0.6, wspace=0.4)
    # plt.suptitle('Weld memory usage over time', fontsize=16)
    # find max y
    max_y = max([df['[MEM]Used'].max() for df in dfs.values()])
    # find max x
    max_x = max([df.index.values[-1] for df in dfs.values()])
    # scale factor for B -> GB
    scale = 1000000

    subplots = [211, 212]
    actual_names = [k.capitalize() for k in dfs.keys()]
    for name, df, subplot, act_name in zip(dfs.keys(), dfs.values(), subplots, actual_names):
        plt.subplot(subplot)
        # change to GB
        plt.plot(df.index.values, df['[MEM]Used'] / scale, label=name)
        plt.ylim(0, max_y / scale + 5)
        plt.xlim(xmin=-2, xmax=max_x * 1.1)
        # plot markers
        time_index = pd.Index(df['Time'])
        previous_time = "0"
        for m, t in zip(markers[name]['marker'], markers[name]['time']):
            x_val = time_index.get_loc(t)
            if t != previous_time:
                plt.axvline(x=x_val, color='red')
                plt.text(x_val + 1, max_y / scale - 5, str(x_val) + ' ' + m, rotation=90, fontsize=8)
            previous_time = t

        plt.ylabel('Memory (GB)')
        plt.xlabel('Time (s)')
        plt.title(act_name)

    if args.save:
        plt.savefig(OUTPUT_FOLDER + '/' + input_ + '/' + experiment + '/profile_scatter.png')
    else:
        plt.show()


def plot_weld_profile_scatter(inputs):
    for input_ in inputs:
        # make sure directory exists
        os.system('mkdir -p ' + OUTPUT_FOLDER + '/' + input_)

        for experiment in all_weld_experiments.keys():
            # make sure directory exists
            os.system('mkdir -p ' + OUTPUT_FOLDER + '/' + input_ + '/' + experiment)

            plot_weld_profile_scatter_single(input_, experiment)


def plot_weld_scatter_single(input_):
    experiment_factors = all_weld_experiments['lazy']
    dfs = {experiment_factor: read_weld_collectl_single(input_, 'lazy', experiment_factor, 2)
           for experiment_factor in experiment_factors}
    markers = {experiment_factor: read_weld_markers_single(input_, 'lazy', experiment_factor, 2)
               for experiment_factor in experiment_factors}
    dfs['Lazy w/ Cache'] = read_weld_collectl_single(input_, 'cache', 'cache', 2)
    markers['Lazy w/ Cache'] = read_weld_markers_single(input_, 'cache', 'cache', 2)

    plt.figure(figsize=(7, 7))
    # plt.suptitle('Weld memory usage over time', fontsize=16)
    # find max y
    max_y = max([df['[MEM]Used'].max() for df in dfs.values()])
    # find max x
    max_x = max([df.index.values[-1] for df in dfs.values()])
    # scale factor for B -> GB
    scale = 1000000

    # remove some negative gradients end of curve
    dfs['lazy'] = dfs['lazy'][:len(dfs['lazy']) - 1]
    dfs['Lazy w/ Cache'] = dfs['Lazy w/ Cache'][:len(dfs['Lazy w/ Cache']) - 1]

    subplots = [311, 312, 313]
    actual_names = [k.capitalize() for k in dfs.keys()]
    for name, df, subplot, act_name in zip(dfs.keys(), dfs.values(), subplots, actual_names):
        plt.subplot(subplot)
        # change to GB
        plt.plot(df.index.values, df['[MEM]Used'] / scale, label=name)
        plt.ylim(0, max_y / scale + 5)
        plt.xlim(xmin=-2, xmax=max_x * 1.1)
        # plot markers
        time_index = pd.Index(df['Time'])
        previous_time = "0"
        for m, t in zip(markers[name]['marker'], markers[name]['time']):
            x_val = time_index.get_loc(t)
            if t != previous_time:
                plt.axvline(x=x_val, color='red')
                plt.text(x_val + 1, max_y / scale - 8, str(x_val) + ' ' + m, rotation=90, fontsize=8)
            previous_time = t

        plt.ylabel('Memory (GB)')
        plt.xlabel('Time (s)')
        plt.title(act_name)

    plt.subplots_adjust(hspace=0.6, wspace=0.6)

    if args.save:
        plt.savefig(OUTPUT_FOLDER + '/' + input_ + '/lazy_profile_scatter.png')
    else:
        plt.show()


def plot_weld_scatter_lazy(inputs):
    for input_ in inputs:
        # make sure directory exists
        os.system('mkdir -p ' + OUTPUT_FOLDER + '/' + input_)

        plot_weld_scatter_single(input_)


def plot_weld_io_bars_single(input_):
    df1 = read_weld_time(input_, 'lazy')
    df2 = read_weld_time(input_, 'cache')
    df = pd.concat([df1, df2])
    df = df.loc[df['pipeline'] != 'No-cache'].reset_index().drop(columns='index')

    df['pipeline'] = ['Eager', 'Lazy', 'Lazy w/ Cache']

    plt.figure(figsize=(4, 5))
    plt.bar(df['pipeline'], df['input_mean'] / 1000, color='m')
    plt.ylabel('File system inputs (K)')
    plt.title('File system inputs for input={}'.format(input_))
    plt.subplots_adjust(left=0.2)

    if args.save:
        plt.savefig(OUTPUT_FOLDER + '/' + input_ + '/io_bars.png')
    else:
        plt.show()


def plot_weld_io_bars(inputs):
    for input_ in inputs:
        # make sure directory exists
        os.system('mkdir -p ' + OUTPUT_FOLDER + '/' + input_)

        plot_weld_io_bars_single(input_)


def plot_weld_final_single(input_):
    df = read_weld_time_single(input_, 'lazy', 'eager')
    df = df.append(read_weld_time_single(input_, 'cache', 'cache')).reset_index().drop(columns='index')

    # fix scales
    df['real_mean'] = df['real_mean'] / 60
    df['real_diff'] = df['real_diff'] / 60
    df['mem_max(K)_mean'] = df['mem_max(K)_mean'] / 1000000
    df['mem_max(K)_diff'] = df['mem_max(K)_diff'] / 1000000
    df['input_mean'] = df['input_mean'] / 1000
    df['input_diff'] = df['input_diff'] / 1000

    plt.figure(figsize=(8, 4))
    subplots = [131, 132, 133]
    columns = ['real', 'mem_max(K)', 'input']
    ylabels = ['Mean real time (min)', 'Mean max memory (GB)', 'File system inputs (K)']
    colors = ['b', 'g', 'm']
    df['pipeline'] = ['Eager', 'Lazy']
    for subplot, column, ylabel, color in zip(subplots, columns, ylabels, colors):
        plt.subplot(subplot)
        plt.bar(df['pipeline'], df[column + '_mean'], yerr=df[column + '_diff'], color=color, width=0.7)
        plt.ylabel(ylabel)

    plt.tight_layout()

    if args.save:
        plt.savefig(OUTPUT_FOLDER + '/' + input_ + '/final_bars.png')
    else:
        plt.show()


def plot_weld_final(inputs):
    # total mem, total io, total time
    for input_ in inputs:
        # make sure directory exists
        os.system('mkdir -p ' + OUTPUT_FOLDER + '/' + input_)

        plot_weld_final_single(input_)


# all available plots
all_plots = {'time_bars_full': plot_time_bars_full,
             'time_bars_rest': plot_time_bars,
             'profile_scatter': plot_profile_scatter,
             'profile_scatter_cpu': plot_profile_scatter_cpu,
             'profile_markers': plot_profile_markers,
             'profile_markers_spark': plot_profile_markers_spark,
             'time_inputs': plot_time_inputs,
             'mem_inputs': plot_mem_inputs,
             'weld_time_bars': plot_weld_time_bars,
             'weld_scatter': plot_weld_profile_scatter,
             'weld_time_lazy': plot_weld_time_lazy,
             'weld_scatter_lazy': plot_weld_scatter_lazy,
             'weld_io': plot_weld_io_bars,
             'weld_final': plot_weld_final}

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
