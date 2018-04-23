import xarray as xr
import os

"""
Join and convert the NETCDF3_CLASSIC files into a large NETCDF4 file (with full HDF5 API)

Note: this uses the same dependencies as in python-libraries
"""

PATH_HOME = os.getenv('HOME2')
PATH_DATASETS_ROOT = '/datasets/ECAD/original/small_sample/'


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


combine(PATH_HOME + PATH_DATASETS_ROOT, ['tg', 'tg_err', 'pp', 'pp_err', 'rr', 'rr_err'], 'data1')
combine(PATH_HOME + PATH_DATASETS_ROOT, ['tn', 'tn_err', 'tx', 'tx_err'], 'data2')
