import numpy as np
import os
from netcdf_parser import to_dataframe

# /export/scratch1/radujica/datasets/ECAD/original/small_sample/
PATH_HOME = os.getenv('HOME2') + '/datasets/ECAD/original/small_sample/'
df1 = to_dataframe(PATH_HOME + 'data1.nc')
df2 = to_dataframe(PATH_HOME + 'data2.nc')


"PIPELINE"
# 1. join the 2 dataframes
df = df1.join(df2, how='inner', sort=False)

# 2. quick preview on the data
print(df.head(10))

# 3. want a subset of the data, here only latitude >= 42.25 & <= 60.25 (~ mainland Europe)
# not a filter because we want to showcase the selection of a subset of rows within the dataset;
# might as well be 123456:987654 but the values in that slice don't make much sense for this dataset
df = df[709920:1482480]  # TODO: update values for larger datasets

# 4. drop rows with null values
# could use ~np.isnan(column) (?)
df = df[(df['tg'].notna()) & (df['pp'].notna()) & (df['rr'].notna())]

# 5. drop pp_err and rr_err columns
df = df.drop(columns=['pp_err', 'rr_err'])


# 6. UDF 1: compute absolute difference between max and min
def compute_abs_maxmin(series_max, series_min):
    return np.abs(np.subtract(series_max, series_min))


df['abs_diff'] = compute_abs_maxmin(df['tx'], df['tn'])

# 7. explore the data through aggregations
print(df.agg(['min', 'max', 'mean', 'std']))    # EVALUATE STEP

# 8. compute std per month
# need index as columns
df = df.reset_index()
# UDF 2: compute custom year+month format
df['year_month'] = df['time'].map(lambda x: x.year * 100 + x.month)
# group by month and rename columns appropriately
df_grouped = df[['latitude', 'longitude', 'year_month', 'tg', 'tn', 'tx', 'pp', 'rr']] \
    .groupby(['latitude', 'longitude', 'year_month']) \
    .mean() \
    .rename(columns={'tg': 'tg_mean', 'tn': 'tn_mean', 'tx': 'tx_mean', 'pp': 'pp_mean', 'rr': 'rr_mean'}) \
    .reset_index()
# merge the results; TODO: this will probably be another EVALUATE STEP to avoid the merge
df = df.merge(df_grouped, on=['latitude', 'longitude', 'year_month'], how='inner')
# clean up
del df_grouped
df = df.drop('year_month', axis=1)

# 9. EVALUATE
print(df)
