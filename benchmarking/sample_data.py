import argparse

import xarray as xr

"""
Create samples of the files
"""

parser = argparse.ArgumentParser(description='Sample Data')
parser.add_argument('-i', '--input', required=True, help='Path to folder containing input files')
parser.add_argument('-o', '--output', required=True, help='Path to output folder')
parser.add_argument('--start', required=True, help='Starting date, e.g. 1950-01-01')
parser.add_argument('--stop', required=True, help='Ending date, e.g. 1950-03-31')
args = parser.parse_args()

FILES = ['tg', 'tg_stderr', 'tn', 'tn_stderr', 'tx', 'tx_stderr', 'pp', 'pp_stderr', 'rr', 'rr_stderr']
PATH_EXTENSION = '.nc'

for file_name in FILES:
    raw_ds = xr.open_dataset(args.input + file_name + PATH_EXTENSION)

    ds = raw_ds.sel(time=slice(args.start, args.stop))

    ds.to_netcdf(path=args.output + file_name + '.nc',
                 format='NETCDF3_CLASSIC',
                 engine='netcdf4')
