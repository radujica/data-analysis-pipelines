import netCDF4
from collections import OrderedDict
import pandas as pd
import numpy as np

"""
Convert NETCDF3_CLASSIC file from netCDF4 to pandas DataFrame.
Based on how xarray does it.

TODO: 
- generalize datetime format conversion, i.e. the =='time' check
- xarray seems to round(?) or just convert from float32 to float64
"""

PATH = '/export/scratch1/radujica/datasets/ECAD/original/small_sample/tg.nc'


def to_dataframe(path):
    ds = netCDF4.Dataset(path, format='NECDF3_CLASSIC')
    columns = [k for k in ds.variables if k not in ds.dimensions]
    ordered_dims = dict(map(lambda kv: (kv[0], kv[1].size), OrderedDict(sorted(ds.dimensions.items())).items()))

    data = []
    for k in columns:
        variable_dims = ds.variables[k].dimensions
        expanded_dims = tuple(d for d in ordered_dims if d not in set(variable_dims)) + variable_dims
        raw_data = ds.variables[k][0:]
        expanded_data = np.broadcast_to(raw_data.filled(np.nan), tuple(ordered_dims[d] for d in expanded_dims))
        axes = tuple(expanded_dims.index(d) for d in ordered_dims)
        transposed_data = np.transpose(expanded_data, axes)
        data.append(transposed_data.reshape(-1))

    def convert_datetime(variable):
        return pd.to_datetime(netCDF4.num2date(variable[0:], variable.units, calendar=variable.calendar))

    names = ordered_dims
    indexes = [convert_datetime(ds.variables[k]) if k == u'time' else ds.variables[k][0:] for k in names]
    index = pd.MultiIndex.from_product(indexes, names=names)
    return pd.DataFrame(OrderedDict(zip(columns, data)), index=index)


print(to_dataframe(PATH))
