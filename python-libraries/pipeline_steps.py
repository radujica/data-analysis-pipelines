import xarray as xr
import numpy as np
import os

HOME = os.getenv('HOME2')
PATH_DATASETS_ROOT = HOME + '/datasets/ECAD/original/small_sample/'
FILES = ['tg', 'tg_err', 'tn', 'tn_err', 'tx', 'tx_err', 'pp', 'pp_err', 'rr', 'rr_err']
PATH_EXTENSION = '.nc'


# 1
def read_datasets():
    for file_name in FILES:
        xr.open_dataset(PATH_DATASETS_ROOT + file_name + PATH_EXTENSION)


# TODO: input data probably needs to be materialized first before timing starts
def _read_datasets():
    for file_name in FILES:
        yield xr.open_dataset(PATH_DATASETS_ROOT + file_name + PATH_EXTENSION)


# 2
def convert_to_dataframe(datasets):
    for ds in datasets:
        ds.to_dataframe()


def _read_dataframes():
    for ds in _read_datasets():
        yield ds.to_dataframe()


# 3. read all data into 1 dataframe; note it is naively done before filtering
def join_dataframes(dfs):
    df = dfs.__next__()
    count = 1

    for raw_df in dfs:
        # the _err files have the same variable name as the non _err files, so fix it
        # e.g. tg -> tg_err
        if count % 2 == 1:
            raw_df = raw_df.rename(columns={FILES[count - 1]: FILES[count]})

        count += 1

        # avoiding right join for performance; join = merge on index by default
        df = df.join(raw_df, how='inner', sort=False)

    return df


# 4 this should bring big gains with weld if these columns are just not read
def drop_columns(df):
    return df.drop(columns=['pp_err', 'rr_err'])


# 5 this should bring big gains with weld if the operations are done purely on this subset;
# note that this would restrict the entire pipeline to this subset
def head(df):
    return df.head(100)


# 5 filter based on value
def drop_null_rows(df):
    return df.dropna()


# 6 aggregations ~ evaluate step
def describe(df):
    return df.describe()


def _compute_abs_maxmin(series_min, series_max):
    return np.abs(np.subtract(series_max, series_min))


# 7 np udf
def add_maxmin_diff(df):
    df['diff'] = _compute_abs_maxmin(df['tx'], df['tn'])

    return df


# 8
def reset_index(df):
    return df.reset_index()


# 9
def set_index(df, index_name_list):
    return df.set_index(index_name_list)


# 10 lambda udf; note index reset is done purely to access an index as a regular column; maybe do outside function? ^
def udf(df):
    # need values as columns
    df = df.reset_index()
    # the actual udf to convert from date to custom year+month format
    df['year_month'] = df['time'].map(lambda x: x.year * 100 + x.month)
    # revert to original index
    df = df.set_index(['latitude', 'longitude', 'time'])

    return df


# 11 just the groupby; requires udf^
def group_by(df):
    return df[['latitude', 'longitude', 'year_month', 'tg', 'tn', 'tx', 'pp', 'rr']]\
        .groupby(['latitude', 'longitude', 'year_month'])


# 12 groupby ~ evaluate step; should time each step; requires udf^
def std_per_month(df):
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
    df = df.drop('year_month', axis=1)\
           .set_index(['latitude', 'longitude', 'time'])

    return df
