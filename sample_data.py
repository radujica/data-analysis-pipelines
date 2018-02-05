import xarray as xr
import os

"""
Create samples of the files

Note: this uses the same dependencies as in python-libraries
"""

PATH_HOME = os.getenv('HOME', '/')
PATH_DATASETS_ROOT = '/datasets/ECAD/original/sample/'
PATH_DATASETS_OUTPUT = '/datasets/ECAD/original/small_sample/'
FILES = ['tg', 'tg_err', 'tn', 'tn_err', 'tx', 'tx_err', 'pp', 'pp_err', 'rr', 'rr_err']
PATH_EXTENSION = '.nc/data'

for file_name in FILES:
    raw_ds = xr.open_dataset(PATH_HOME + PATH_DATASETS_ROOT + file_name + PATH_EXTENSION)

    ds = raw_ds.sel(time=slice('1950-01-01', '1950-03-31'))

    ds.to_netcdf(path=PATH_HOME + PATH_DATASETS_OUTPUT + file_name + '.nc',
                 format='NETCDF3_CLASSIC',
                 engine='netcdf4')
