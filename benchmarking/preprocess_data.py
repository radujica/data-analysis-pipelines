import argparse

import xarray as xr

"""
Join and convert the NETCDF3_CLASSIC files into a large NETCDF4 file (with full HDF5 API)
"""

parser = argparse.ArgumentParser(description='Combine data')
parser.add_argument('-i', '--input', required=True, help='Path to folder containing input files; also output folder')
args = parser.parse_args()


def combine(path, files, output):
    file_extension = '.nc'

    # first dataset
    result_ds = xr.open_dataset(path + files[0] + file_extension)

    # merge all into result_ds
    for file_name in files[1:]:
        raw_ds = xr.open_dataset(path + file_name + file_extension)

        # the _err files have the same variable name as the non _err files, so fix it
        # e.g. tg -> tg_err
        if files.index(file_name) % 2 == 1:
            raw_ds.rename({files[files.index(file_name) - 1]: file_name}, inplace=True)

        result_ds = xr.merge([result_ds, raw_ds], join='inner')

    result_ds.to_netcdf(path=path + output + file_extension,
                        format='NETCDF4',
                        engine='netcdf4')


combine(args.input, ['tg', 'tg_stderr', 'pp', 'pp_stderr', 'rr', 'rr_stderr'], 'data1')
combine(args.input, ['tn', 'tn_stderr', 'tx', 'tx_stderr'], 'data2')
