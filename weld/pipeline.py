from __future__ import print_function
from datetime import datetime

import argparse
import sys

import os
from weld.weldobject import WeldObject

import pandas_weld as pdw

parser = argparse.ArgumentParser(description='Weld Pipeline')
parser.add_argument('-i', '--input', required=True, help='Path to folder containing input files')
parser.add_argument('-s', '--slice', required=True, help='Start and stop of a subset of the data')
parser.add_argument('-o', '--output', help='Path to output folder')
# TODO: remove; redundant; implied by --output
parser.add_argument('-c', '--check', action='store_true', default=False,
                    help='If passed, create output to check correctness of the pipeline, so output is saved '
                         'to csv files in --output folder. Otherwise, prints to stdout')
parser.add_argument('-e', '--eager', action='store_true')
args = parser.parse_args()

if args.check and args.output is None:
    raise RuntimeError('--check requires an output folder path')

PATH1 = args.input + 'data1.nc'
PATH2 = args.input + 'data2.nc'

if args.eager:
    df1 = pdw.read_netcdf4_eager(PATH1)
    df2 = pdw.read_netcdf4_eager(PATH2)
else:
    df1 = pdw.read_netcdf4(PATH1)
    df2 = pdw.read_netcdf4(PATH2)


def print_event(name):
    print('#{}-{}'.format(str(datetime.now()), name))


print_event('data_read')


# PIPELINE
# 1. join the 2 dataframes
df = df1.merge(df2)

print_event('data_merged')

# 2. quick preview on the data
df_head = df.head(10)
if args.check:
    df_head.to_csv(args.output + 'head' + '.csv')
else:
    print(df_head, file=sys.stderr)

print_event('done_head')

# 3. want a subset of the data
slice_ = [int(x) for x in args.slice.split(':')]
df = df[slice_[0]:slice_[1]]

# 4. drop rows with null values
# could use ~np.isnan(column) (?)
df = df[(df['tg'] != -99.99) & (df['pp'] != -999.9) & (df['rr'] != -999.9)]

# 5. drop pp_err and rr_err columns
df = df.drop(columns=['pp_stderr', 'rr_stderr'])


# 6. UDF 1: compute absolute difference between max and min
# this could alternatively be implemented in numpy_weld but really results in the same thing
def compute_abs_maxmin(series_max, series_min):
    weld_template = "map(zip(%(self)s, %(other)s), |e| if(e.$0 > e.$1, e.$0 - e.$1, e.$1 - e.$0))"
    mapping = {'other': pdw.get_expression_or_raw(series_min)}

    return series_max.map(weld_template, mapping)


df['abs_diff'] = compute_abs_maxmin(df['tx'], df['tn'])

# 7. explore the data through aggregations
df_agg = df.agg(['min', 'max', 'mean', 'std'])\
    .reset_index()\
    .rename(columns={'Index': 'agg'})\
    .evaluate()
if args.check:
    df_agg.to_csv(args.output + 'agg' + '.csv', index=False)
else:
    print(df_agg, file=sys.stderr)

print_event('done_agg')

# 8. compute mean per month
# need index as columns
df = df.reset_index()


# UDF 2: compute custom year+month format, now through C UDF
def compute_year_month(time):
    WeldObject.load_binary(os.getcwd() + '/udf_yearmonth.so')
    weld_template = "cudf[udf_yearmonth, vec[vec[i8]]](%(self)s)"

    return time.map(weld_template, {})


df['year_month'] = compute_year_month(df['time'])
# group by month and rename columns appropriately
df_grouped = df[['latitude', 'longitude', 'year_month', 'tg', 'tn', 'tx', 'pp', 'rr']] \
    .groupby(['latitude', 'longitude', 'year_month']) \
    .mean() \
    .rename(columns={'tg': 'tg_mean', 'tn': 'tn_mean', 'tx': 'tx_mean', 'pp': 'pp_mean', 'rr': 'rr_mean'}) \
    .reset_index()\
    .evaluate()

# 9. EVALUATE
if args.check:
    df.evaluate().to_csv(args.output + 'result' + '.csv', index=False)
else:
    print(df.evaluate(), file=sys.stderr)

print_event('done_evaluate')
