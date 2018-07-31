import argparse

import pandas_weld as pdw

parser = argparse.ArgumentParser(description='Weld Pipeline')
parser.add_argument('-i', '--input', required=True, help='Path to folder containing input files')
parser.add_argument('-s', '--slice', required=True, help='Start and stop of a subset of the data')
parser.add_argument('-o', '--output', required=True, help='Path to output folder')
parser.add_argument('-e', '--eager', action='store_true')
parser.add_argument('-c', '--csv', action='store_true')
args = parser.parse_args()


PATH = args.input + 'data1.nc'

if args.eager:
    df = pdw.read_netcdf4_eager(PATH)
else:
    df = pdw.read_netcdf4(PATH)

slice_ = [int(x) for x in args.slice.split(':')]
df = df[slice_[0]:slice_[1]]

df = df.drop(columns=['tx', 'tx_stderr'])

df.sum().to_frame().to_csv(args.output + 'output' + '.csv')
