"""
Convert netcdf to csv
"""
import argparse
import pandas_weld as pdw

# pipenv run python convert_to_csv.py --input=<path>

parser = argparse.ArgumentParser(description='Combine data')
parser.add_argument('-i', '--input', required=True, help='Path to folder containing input files; also output folder')
args = parser.parse_args()

pdw.read_netcdf4_eager(args.input + 'data1.nc').evaluate().to_csv(args.input + 'data1.csv')
pdw.read_netcdf4_eager(args.input + 'data2.nc').to_csv(args.input + 'data2.csv')
