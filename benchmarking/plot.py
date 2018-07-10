import argparse

import matplotlib.pyplot as plt
import os
import pandas as pd

HOME2 = os.environ.get('HOME2')
if HOME2 is None:
    raise RuntimeError('Cannot find HOME2 environment variable')

OUTPUT_FOLDER = HOME2 + '/results/graphs'

all_pipelines = ['python-libraries', 'weld-single', 'weld-par', 'julia', 'java', 'R', 'spark-single', 'spark-par']
all_inputs = ['data_0', 'data_1', 'data_3', 'data_6', 'data_12', 'data_25', 'data_50', 'data_100']
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
    first_df_path = HOME2 + '/results/pipelines/' + input_ + '/time/' + pipeline + '/0_time.csv'
    df = pd.read_csv(first_df_path)
    for run in range(1, 5):
        input_path = HOME2 + '/results/pipelines/' + input_ + '/time/' + pipeline + '/' + str(run) + '_time.csv'
        df = df.append(pd.read_csv(input_path))

    res = pd.DataFrame({'real_mean': [df['real'].mean()],
                        'real_diff': [(df['real'].max() - df['real'].min()) / 2],
                        'pipeline': [pipeline]})

    return res


def plot_time_bars_single(input_):
    pipelines = ['python-libraries', 'weld-single', 'weld-par', 'julia', 'java', 'R', 'spark-single', 'spark-par']
    dfs = [read_time_input_df(input_, pipeline) for pipeline in pipelines]

    # combine into 1 df
    df = dfs[0]
    for d in dfs[1:]:
        df = df.append(d)

    # fix index
    df = df.reset_index().drop(columns='index').sort_values(by='real_mean').reset_index()

    # fix spark-single bar
    spark_single_val = df['real_mean'][7]
    df.loc[df['pipeline'] == 'spark-single', 'real_mean'] = df['real_mean'][6] + 10
    df.loc[df['pipeline'] == 'spark-single', 'real_diff'] = 0

    # remove error bars if too small?
    # df['real_diff'] = df['real_diff'].map(lambda x: 0 if x < 10 else x)

    plt.figure(figsize=(12, 8))
    bar = plt.bar(df.index.values, df['real_mean'] / 60, yerr=df['real_diff'] / 60)  # capsize=6
    plt.ylabel('Time (minutes)')
    plt.title('Mean real time to run pipeline for input={}'.format(input_))
    plt.xticks(df.index.values, df['pipeline'])
    # add value of spark-single
    plt.text(bar[-1].get_x() + bar[-1].get_width() / 3, 0.1, '{0:.1f}'.format(spark_single_val / 60))
    plt.scatter(df.index.values[-1], df['real_mean'][7] / 60 + 0.1, c='black', s=1)
    plt.scatter(df.index.values[-1], df['real_mean'][7] / 60 + 0.2, c='black', s=1)

    if args.save:
        plt.savefig(OUTPUT_FOLDER + '/' + input_ + '/time_bars.png')
    else:
        plt.show()


def plot_time_bars(inputs):
    for input_ in inputs:
        # make sure directory exists
        os.system('mkdir -p ' + OUTPUT_FOLDER + '/' + input_)

        plot_time_bars_single(input_)


def plot_time_inputs_single(pipeline):
    inputs = ['data_0', 'data_1', 'data_3', 'data_6', 'data_12']
    dfs = [read_time_input_df(input_, pipeline) for input_ in inputs]

    # combine into 1 df
    df = dfs[0]
    for d in dfs[1:]:
        df = df.append(d)

    df = df.reset_index()[['real_mean', 'real_diff']]

    plt.figure(figsize=(12, 8))
    plt.bar(df.index.values, df['real_mean'] / 60, yerr=df['real_diff'] / 60, color='y', capsize=4)
    plt.ylabel('Time (minutes)')
    plt.xlabel('Datasets')
    plt.title('Mean real time to run pipeline={} for all inputs'.format(pipeline))
    plt.xticks(df.index.values, inputs)

    # plot a power curve
    x = df.index.values
    y = [df['real_mean'][0] / 60] * len(x)
    for i in range(1, len(x)):
        y[i] = y[i - 1] * 2
    plt.plot(df.index.values, y, color='r')

    if args.save:
        plt.savefig(OUTPUT_FOLDER + '/all/' + 'time_inputs_' + str(pipeline) + '.png')
    else:
        plt.show()


def plot_time_inputs(inputs):
    pipelines = ['python-libraries', 'weld-single', 'weld-par', 'julia', 'java', 'R']
    for pipeline in pipelines:
        # make sure directory exists
        os.system('mkdir -p ' + OUTPUT_FOLDER + '/all')

        plot_time_inputs_single(pipeline)


def read_mem_input_df(input_, pipeline):
    first_df_path = HOME2 + '/results/pipelines/' + input_ + '/time/' + pipeline + '/0_time.csv'
    df = pd.read_csv(first_df_path)
    for run in range(1, 5):
        input_path = HOME2 + '/results/pipelines/' + input_ + '/time/' + pipeline + '/' + str(run) + '_time.csv'
        df = df.append(pd.read_csv(input_path))

    res = pd.DataFrame({'mem_mean': [df['mem_max(K)'].mean()],
                        'mem_diff': [(df['mem_max(K)'].max() - df['mem_max(K)'].min()) / 2],
                        'pipeline': [pipeline]})

    return res


def plot_mem_inputs_single(pipeline):
    inputs = ['data_0', 'data_1', 'data_3', 'data_6', 'data_12']
    input_sizes = [0.362, 0.724, 1.447, 2.895, 5.790]#, 11.582, 23.163, 46.328]
    dfs = [read_mem_input_df(input_, pipeline) for input_ in inputs]

    # combine into 1 df
    df = dfs[0]
    for d in dfs[1:]:
        df = df.append(d)

    df = df.reset_index()[['mem_mean', 'mem_diff']]

    plt.figure(figsize=(12, 8))
    plt.bar(df.index.values, df['mem_mean'] / 1000000, yerr=df['mem_diff'] / 1000000, color='y')
    plt.ylabel('Memory (GB)')
    plt.xlabel('Datasets')
    plt.title('Mean total memory to run pipeline={} for all inputs'.format(pipeline))
    plt.xticks(df.index.values, inputs)

    # plot a power curve
    plt.plot(df.index.values, input_sizes, color='r')

    if args.save:
        plt.savefig(OUTPUT_FOLDER + '/all/' + 'mem_inputs_' + str(pipeline) + '.png')
    else:
        plt.show()


def plot_mem_inputs(inputs):
    pipelines = ['python-libraries', 'weld-single', 'weld-par', 'julia', 'java', 'R']
    for pipeline in pipelines:
        # make sure directory exists
        os.system('mkdir -p ' + OUTPUT_FOLDER + '/all')

        plot_mem_inputs_single(pipeline)


def read_collectl_input_df(input_, pipeline):
    first_df_path = HOME2 + '/results/pipelines/' + input_ + '/profile/' + pipeline + '/0_profile.csv'
    df = pd.DataFrame({'real_0': pd.read_csv(first_df_path, skiprows=range(0, 15), usecols=['[MEM]Used'])['[MEM]Used']})
    for run in range(1, 5):
        input_path = HOME2 + '/results/pipelines/' + input_ + '/profile/' + pipeline + '/' + str(run) + '_profile.csv'
        df['real_' + str(run)] = pd.read_csv(input_path, skiprows=range(0, 15), usecols=['[MEM]Used'])['[MEM]Used']

    series_mean = df.mean(axis=1) / 1000000
    df['mem_diff'] = (df.max(axis=1) / 1000000 - df.min(axis=1) / 1000000) / 2
    df['mem_mean'] = series_mean
    df = df[['mem_mean', 'mem_diff']]

    # keep track of which pipeline the data refers to
    df['pipeline'] = pipeline
    # remove the last 2 values
    df = df[0:len(df) - 2]

    return df


def plot_profile_scatter_single(input_):
    pipelines = ['python-libraries', 'weld-single', 'weld-par', 'julia', 'java', 'R']
    dfs = {pipeline: read_collectl_input_df(input_, pipeline) for pipeline in pipelines}

    plt.figure(figsize=(12, 8))
    [plt.plot(df.index.values, df['mem_mean'], '-', label=name) for name, df in dfs.items()]
    [plt.fill_between(df.index.values, df['mem_mean'] - df['mem_diff'], df['mem_mean'] + df['mem_diff'], alpha=0.3) for df in dfs.values()]
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


def read_collectl_input_df_cpu(input_, pipeline):
    first_df_path = HOME2 + '/results/pipelines/' + input_ + '/profile/' + pipeline + '/0_profile.csv'
    df = pd.DataFrame({'real_0': pd.read_csv(first_df_path, skiprows=range(0, 15), usecols=['[CPU]Totl%'])['[CPU]Totl%'] * 32})
    for run in range(1, 5):
        input_path = HOME2 + '/results/pipelines/' + input_ + '/profile/' + pipeline + '/' + str(run) + '_profile.csv'
        df['real_' + str(run)] = pd.read_csv(input_path, skiprows=range(0, 15), usecols=['[CPU]Totl%'])['[CPU]Totl%'] * 32

    series_mean = df.mean(axis=1)
    df['cpu_diff'] = (df.max(axis=1) - df.min(axis=1)) / 2
    df['cpu_mean'] = series_mean
    df = df[['cpu_mean', 'cpu_diff']]

    # keep track of which pipeline the data refers to
    df['pipeline'] = pipeline
    # remove the last 2 values
    df = df[0:len(df) - 2]

    return df


def plot_profile_scatter_single_cpu(input_):
    pipelines = ['python-libraries', 'weld-single', 'weld-par', 'julia', 'java', 'R']#, 'spark-single', 'spark-par']
    dfs = {pipeline: read_collectl_input_df_cpu(input_, pipeline) for pipeline in pipelines}

    max_x = max([df.index.values[-1] for df in dfs.values()])
    plt.figure(figsize=(12, 8))
    [plt.plot(df.index.values, df['cpu_mean'], '-', label=name) for name, df in dfs.items()]
    [plt.fill_between(df.index.values, df['cpu_mean'] - df['cpu_diff'], df['cpu_mean'] + df['cpu_diff'], alpha=0.3) for df in dfs.values()]
    plt.ylabel('CPU (%)')
    plt.xlabel('Time (s)')
    plt.title('CPU Total usage over input={}'.format(input_))
    plt.legend()

    plt.axhline(y=100, color='k')
    plt.text(max_x - 3, 130, '100')

    if args.save:
        plt.savefig(OUTPUT_FOLDER + '/' + input_ + '/profile_scatter_cpu.png')
    else:
        plt.show()


def plot_profile_scatter_cpu(inputs):
    for input_ in inputs:
        # make sure directory exists
        os.system('mkdir -p ' + OUTPUT_FOLDER + '/' + input_)

        plot_profile_scatter_single_cpu(input_)


def parse_markers_csv_time_bars(input_, pipeline, run):
    # read the compile times
    compile_input_path = HOME2 + '/results/weld/' + input_ + '/compile_' + pipeline + '_' + str(run) + '.txt'
    df_compile = pd.read_csv(compile_input_path, header=None, names=['type'])
    # filter out the markers
    df_compile = df_compile[df_compile['type'].str[0] != '#']
    # split into 2 columns
    df_compile['type'], df_compile['time'] = df_compile['type'].str.split(':', 1).str
    df_compile['time'] = df_compile['time'].apply(pd.to_numeric)
    df_compile = df_compile.groupby('type').sum().transpose().reset_index()
    df_compile['total_convert'] = df_compile['Python->Weld'] + df_compile['Weld->Python']

    return df_compile


def read_weld_time_input_df(input_, pipeline):
    input_path = HOME2 + '/results/weld/' + input_ + '/time_' + pipeline + '_' + str(0) + '.csv'
    df = pd.read_csv(input_path)
    df_compile = parse_markers_csv_time_bars(input_, pipeline, 0)
    for i in range(1, 5):
        df2 = pd.read_csv(HOME2 + '/results/weld/' + input_ + '/time_' + pipeline + '_' + str(i) + '.csv')
        df = df.append(df2)
        df2 = parse_markers_csv_time_bars(input_, pipeline, i)
        df_compile = df_compile.append(df2)

    df = pd.concat([df, df_compile], axis=1)[['real', 'Weld compile time', 'total_convert', 'Weld']]

    series_mean = df.mean(axis=0)
    df_mean = pd.DataFrame({series_mean.index[i] + '_mean': [series_mean[i]] for i in range(len(series_mean))})
    series_diff = (df.max(axis=0) - df.min(axis=0)) / 2
    df_diff = pd.DataFrame({series_diff.index[i] + '_diff': [series_diff[i]] for i in range(len(series_diff))})

    df = pd.concat([df_mean, df_diff], axis=1)
    df['pipeline'] = pipeline

    return df


def plot_weld_time_bars_single(input_):
    dfs = [read_weld_time_input_df(input_, pipeline) for pipeline in all_weld_types]
    actual_names = ['Eager', 'Lazy', 'Eager & Cache', 'Lazy & Cache']

    # combine into 1 df
    df = dfs[0]
    for d in dfs[1:]:
        df = df.append(d)

    # fix index
    df = df.reset_index().drop(columns='index')
    df['pipeline'] = actual_names

    plt.subplots(figsize=(12, 8))
    p1 = plt.bar(df.index.values, df['Weld_mean'])
    p2 = plt.bar(df.index.values, df['Weld compile time_mean'], bottom=df['Weld_mean'])
    p3 = plt.bar(df.index.values, df['total_convert_mean'],
                 bottom=df['Weld compile time_mean'] + df['Weld_mean'])
    p4 = plt.bar(df.index.values, df['real_mean'] - df['Weld compile time_mean'] - df['total_convert_mean'] - df['Weld_mean'],
                 bottom=df['total_convert_mean'] + df['Weld compile time_mean'] + df['Weld_mean'],
                 yerr=df['real_diff'], capsize=4)
    plt.ylabel('Time (s)')
    plt.title('Time to run pipeline for input={}'.format(input_))
    plt.ylim(ymax=(df['real_mean']).max() + 10)
    plt.xticks([])
    plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Weld', 'compile', 'en-/decode', 'python'))

    # add values as numbers over the bars and celltext
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
                      loc='bottom')
    table.scale(1, 2)
    plt.subplots_adjust(left=0.2, bottom=0.3)

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
    input_path = HOME2 + '/results/weld/' + input_ + '/profile_' + pipeline + '_0.csv'
    df = pd.read_csv(input_path, skiprows=range(0, 15))
    # keep track of which pipeline the data refers to
    df['pipeline'] = pipeline

    return df


def read_weld_markers(input_, pipeline):
    input_path = HOME2 + '/results/weld/' + input_ + '/compile_' + pipeline + '_0.txt'
    df = pd.read_csv(input_path, header=None, names=['data'])
    # filter out the markers
    df = df[df['data'].str[0] == '#']
    # split the data
    df['time'], df['marker'] = df['data'].str.split('-', 1).str
    df['time'] = df['time'].apply(lambda x: x[1:])
    del df['data']
    # keep track of which pipeline the data refers to
    df['pipeline'] = pipeline

    return df


def plot_weld_profile_scatter_single(input_):
    dfs = {pipeline: read_weld_collectl_input_df(input_, pipeline) for pipeline in all_weld_types}
    markers = {pipeline: read_weld_markers(input_, pipeline) for pipeline in all_weld_types}

    dfs['no-lazy-no-cache'] = dfs['no-lazy-no-cache'][:len(dfs['no-lazy-no-cache']) - 1]
    dfs['no-cache'] = dfs['no-cache'][:len(dfs['no-cache']) - 1]

    plt.figure(figsize=(12, 8))
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
    max_xs = [max_x_other, max_x_other, max_x_other, max_x_other]
    actual_names = ['Eager', 'Lazy', 'Eager & Cache', 'Lazy & Cache']
    for name, df, subplot, max_x, act_name in zip(dfs.keys(), dfs.values(), subplots, max_xs, actual_names):
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
                plt.text(x_val + 1, max_y / scale - 5, str(x_val) + ' ' + m, rotation=90)
            previous_time = t

        plt.ylabel('Memory (GB)')
        plt.xlabel('Time (s)')
        plt.title(act_name)

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
             'profile_scatter_cpu': plot_profile_scatter_cpu,
             'time_inputs': plot_time_inputs,
             'mem_inputs': plot_mem_inputs,
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
