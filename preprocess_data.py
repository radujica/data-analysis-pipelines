import xarray as xr
import os

"""
Join and convert the NETCDF3_CLASSIC files into a large NETCDF4 file (with full HDF5 API)

Note: this uses the same dependencies as in python-libraries
"""

PATH_HOME = os.getenv('HOME', '/')
PATH_DATASETS_ROOT = '/datasets/ECAD/original/sample/'
FILES = ['tg', 'tg_err', 'tn', 'tn_err', 'tx', 'tx_err', 'pp', 'pp_err', 'rr', 'rr_err']
PATH_EXTENSION = '.nc/data'
OUTPUT_NAME = 'combined.nc'

# first dataset
result_ds = xr.open_dataset(PATH_HOME + PATH_DATASETS_ROOT + FILES[0] + PATH_EXTENSION)

# merge all into result_ds
for file_name in FILES[1:]:
    raw_ds = xr.open_dataset(PATH_HOME + PATH_DATASETS_ROOT + file_name + PATH_EXTENSION)

    # the _err files have the same variable name as the non _err files, so fix it
    # e.g. tg -> tg_err
    if FILES.index(file_name) % 2 == 1:
        raw_ds.rename({FILES[FILES.index(file_name) - 1]: file_name}, inplace=True)

    result_ds = xr.merge([result_ds, raw_ds], join='inner')

result_ds.to_netcdf(path=PATH_HOME + PATH_DATASETS_ROOT + OUTPUT_NAME,
                    format='NETCDF4',
                    engine='netcdf4')
