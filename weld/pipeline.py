import argparse

import pandas_weld as pdw

parser = argparse.ArgumentParser(description='Weld Pipeline')
parser.add_argument('-i', '--input', required=True, help='Path to folder containing input files')
parser.add_argument('-o', '--output', help='Path to output folder')
parser.add_argument('-c', '--check', action='store_true', default=False,
                    help='If passed, create output to check correctness of the pipeline, so output is saved '
                         'to csv files in --output folder. Otherwise, prints to stdout')
parser.add_argument('-e', '--eager', default='nope')
args = parser.parse_args()

if args.check and args.output is None:
    raise RuntimeError('--check requires an output folder path')

PATH1 = args.input + 'data1.nc'
PATH2 = args.input + 'data2.nc'

if args.eager is 'nope':
    df1 = pdw.read_netcdf4(PATH1)
    df2 = pdw.read_netcdf4(PATH2)
else:
    df1 = pdw.read_netcdf4_eager(PATH1)
    df2 = pdw.read_netcdf4_eager(PATH2)


# PIPELINE
# 1. join the 2 dataframes
df = df1.merge(df2)

# 2. quick preview on the data
df_head = df.head(10)
if args.check:
    df_head.to_csv(args.output + 'head' + '.csv')
else:
    print(df_head)

# 3. want a subset of the data, here only latitude >= 42.25 & <= 60.25 (~ mainland Europe)
# not a filter because we want to showcase the selection of a subset of rows within the dataset;
# might as well be 123456:987654 but the values in that slice don't make much sense for this dataset
df = df[709920:1482480]  # TODO: update values for larger datasets

# 4. drop rows with null values
# could use ~np.isnan(column) (?)
df = df[(df['tg'] != -99.99) & (df['pp'] != -999.9) & (df['rr'] != -999.9)]

# 5. drop pp_err and rr_err columns
df = df.drop(columns=['pp_err', 'rr_err'])


# 6. UDF 1: compute absolute difference between max and min
# this could alternatively be implemented in numpy_weld but really results in the same thing
def compute_abs_maxmin(series_max, series_min):
    weld_template = "map(zip(%(self)s, %(other)s), |e| if(e.$0 > e.$1, e.$0 - e.$1, e.$1 - e.$0))"
    mapping = {'other': pdw.get_expression_or_raw(series_min)}

    return series_max.map(weld_template, mapping)


df['abs_diff'] = compute_abs_maxmin(df['tx'], df['tn'])

# 7. explore the data through aggregations
df_agg = df.agg(['min', 'max', 'mean', 'std'])\
    .evaluate()\
    .reset_index()\
    .rename(columns={'Index': 'agg'})
if args.check:
    df_agg.to_csv(args.output + 'agg' + '.csv', index=False)  # EVALUATE STEP
else:
    print(df_agg)

# TODO: this needs update from Weld
# # 8. compute std per month
# # need index as columns
# df = df.reset_index()
# # UDF 2: compute custom year+month format
# df['year_month'] = df['time'].map(lambda x: x.year * 100 + x.month)
# # group by month and rename columns appropriately
# df_grouped = df[['latitude', 'longitude', 'year_month', 'tg', 'tn', 'tx', 'pp', 'rr']] \
#     .groupby(['latitude', 'longitude', 'year_month']) \
#     .mean() \
#     .rename(columns={'tg': 'tg_mean', 'tn': 'tn_mean', 'tx': 'tx_mean', 'pp': 'pp_mean', 'rr': 'rr_mean'}) \
#     .reset_index()
# # merge the results; TODO: this will probably be another EVALUATE STEP to avoid the merge
# df = df.merge(df_grouped, on=['latitude', 'longitude', 'year_month'], how='inner')
# # clean up
# del df_grouped
# df = df.drop('year_month', axis=1)

# 9. EVALUATE
if args.check:
    df.to_csv(args.output + 'result' + '.csv', index=False)  # EVALUATE STEP
else:
    print(df)
