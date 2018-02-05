import xarray as xr
import numpy as np
import os

PATH_HOME = os.getenv('HOME', '/')
PATH_DATASETS_ROOT = '/datasets/ECAD/original/small_sample/'
FILES = ['tg', 'tg_err', 'tn', 'tn_err', 'tx', 'tx_err', 'pp', 'pp_err', 'rr', 'rr_err']
PATH_EXTENSION = '.nc'


def read_file_as_df(path):
    """
    :param path: str
        Path to netCDF file
    :return: Pandas DataFrame of :param path:
    """
    return xr.open_dataset(path).to_dataframe()


# 1. read all data into 1 dataframe; note it is naively done before filtering nulls
df = read_file_as_df(PATH_HOME + PATH_DATASETS_ROOT + FILES[0] + PATH_EXTENSION)

for file_name in FILES[1:]:
    raw_df = read_file_as_df(PATH_HOME + PATH_DATASETS_ROOT + file_name + PATH_EXTENSION)

    # the _err files have the same variable name as the non _err files, so fix it
    # e.g. tg -> tg_err
    if FILES.index(file_name) % 2 == 1:
        raw_df = raw_df.rename(columns={FILES[FILES.index(file_name) - 1]: file_name})

    # avoiding right join for performance; join = merge on index by default
    df = df.join(raw_df, how='inner', sort=False)


# 2. drop rows with null values
df = df.dropna()


# 3. drop pp_err and rr_err columns
df = df.drop(columns=['pp_err', 'rr_err'])


# 4. explore the data through aggregations
print(df.describe())


# 5. compute absolute difference between max and min values;
# absolute assuming it might be the (unexpected) case than min > max
def compute_abs_maxmin(series_min, series_max):
    return np.abs(np.subtract(series_max, series_min))


df['abs_diff'] = compute_abs_maxmin(df['tx'], df['tn'])


# 6. compute std per month
# need values as columns
df = df.reset_index()
# the actual udf to convert from date to custom year+month format
df['year_month'] = df['time'].map(lambda x: x.year * 100 + x.month)
# group by month and rename columns appropriately
df_grouped = df[['latitude', 'longitude', 'year_month', 'tg', 'tn', 'tx', 'pp', 'rr']]\
    .groupby(['latitude', 'longitude', 'year_month'])\
    .std()\
    .rename(columns={'tg': 'tg_std', 'tn': 'tn_std', 'tx': 'tx_std', 'pp': 'pp_std', 'rr': 'rr_std'})\
    .reset_index()
# merge the results
df = df.merge(df_grouped, on=['latitude', 'longitude', 'year_month'], how='inner')
# clean up
del df_grouped
df = df.drop('year_month', axis=1)
df = df.set_index(['latitude', 'longitude', 'time'])
