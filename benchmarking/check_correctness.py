import argparse

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Check pipeline output correctness')
parser.add_argument('files', nargs='+')
args = parser.parse_args()

# the first one shall be the ground truth to compare against, i.e. the python implementation
df_truth = pd.read_csv(args.files[0])


def compare_column(column_truth, other_column):
    error_message = 'Columns not equal. Truth={}:{}, Other={}:{}'.format(column_truth.name,
                                                                         column_truth.values,
                                                                         other_column.name,
                                                                         other_column.values)
    if column_truth.dtype == np.dtype(np.object):
        np.testing.assert_array_equal(column_truth.values, other_column.values, error_message)

    else:
        np.testing.assert_array_almost_equal(column_truth.values, other_column.values, decimal=3, err_msg=error_message)


def compare_df(df_truth, other_df):
    len_truth = len(df_truth)
    len_other = len(other_df)
    np.testing.assert_equal(len_truth, len_truth,
                            'DataFrame lengths not equal. Truth={}, Other={}'.format(len_truth, len_other))

    for column in df_truth:
        compare_column(df_truth[column], other_df[column])


for file in args.files[1:]:
    other_df = pd.read_csv(file)
    compare_df(df_truth, other_df)
