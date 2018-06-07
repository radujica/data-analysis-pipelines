import pandas as pd
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Check pipeline output correctness')
parser.add_argument('files', nargs='+')
args = parser.parse_args()

# the first one shall be the ground truth to compare against, i.e. the python implementation
df_truth = pd.read_csv(args.files[0])


def compare_column(column_truth, other_column):
    if column_truth.dtype == np.dtype(np.object):
        np.testing.assert_array_equal(column_truth.values, other_column.values)
    else:
        np.testing.assert_array_almost_equal(column_truth.values, other_column.values, decimal=3)


def compare_df(df_truth, other_df):
    for column in df_truth:
        compare_column(df_truth[column], other_df[column])


for file in args.files[1:]:
    other_df = pd.read_csv(file)
    compare_df(df_truth, other_df)
