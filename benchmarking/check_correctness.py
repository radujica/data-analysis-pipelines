import argparse

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Check pipeline output correctness')
parser.add_argument('files', nargs='+')
args = parser.parse_args()

# the first one shall be the ground truth to compare against, i.e. the python implementation
file_name = args.files[0].split('/')[-1].split('_')[-1]


# make sure dataframes are sorted some way
def sort_df(df, file_name):
    if file_name == 'agg.csv':
        df = df.sort_values(by='agg')
    elif file_name == 'grouped.csv':
        df = df.sort_values(by='column')

    return df


def compare_column(column_truth, other_column):
    error_message = 'Columns not equal.\nTruth={}:{}\nOther={}:{}'.format(column_truth.name,
                                                                          column_truth.values,
                                                                          other_column.name,
                                                                          other_column.values)
    # strings ~ the agg column in agg.csv and column in grouped.csv
    if column_truth.dtype == np.dtype(np.object):
        np.testing.assert_array_equal(column_truth.values, other_column.values, error_message, verbose=False)
    else:
        np.testing.assert_allclose(column_truth.values, other_column.values, rtol=1, err_msg=error_message, verbose=False)


def compare_df(df_truth, other_df):
    len_truth = len(df_truth)
    len_other = len(other_df)
    print('Checking={}'.format(other_df.name))
    np.testing.assert_equal(len_truth, len_truth,
                            'DataFrame lengths not equal.\nTruth={}\nOther={}'.format(len_truth, len_other))

    for column in df_truth:
        compare_column(df_truth[column], other_df[column])


df_truth = sort_df(pd.read_csv(args.files[0]), file_name)

for file in args.files[1:]:
    other_df = sort_df(pd.read_csv(file), file_name)
    other_df.name = '/'.join(file.split('/')[-2:])
    compare_df(df_truth, other_df)
