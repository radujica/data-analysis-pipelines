import netCDF4
from collections import OrderedDict
import pandas as pd
import numpy as np

"""
Used to convert my relevant netCDF3_CLASSIC datasets to pandas dataframes.
"""


def to_dataframe(ds):
    """ Convert netCDF4.Dataset to pandas.DataFrame

    Based on how xarray does it.

    TODO:

    - further generalize datetime format conversion; can calendar attr be in non-datetime data?

    - xarray seems to round(?) or just convert from float32 to float64

    Parameters
    ----------
    ds : netCDF4.Dataset
        to convert from

    Returns
    -------
    df : pandas.DataFrame
        the resulting dataframe

    """
    
    columns = [k for k in ds.variables if k not in ds.dimensions]
    ordered_dimensions = dict(map(lambda kv: (kv[0], kv[1].size), OrderedDict(sorted(ds.dimensions.items())).items()))

    data = []
    for k in columns:
        variable_dims = ds.variables[k].dimensions
        raw_data = ds.variables[k][0:]
        shape = tuple(ordered_dimensions[d] for d in variable_dims)
        expanded_data = np.broadcast_to(raw_data.filled(np.nan), shape)
        axes = tuple(variable_dims.index(d) for d in ordered_dimensions)
        transposed_data = np.transpose(expanded_data, axes)
        data.append(transposed_data.reshape(-1))

    def convert_datetime(variable):
        return pd.to_datetime(netCDF4.num2date(variable[0:], variable.units, calendar=variable.calendar))

    indexes = [convert_datetime(ds.variables[k]) if hasattr(ds.variables[k], 'calendar')
               else ds.variables[k][0:] for k in ordered_dimensions]
    index = pd.MultiIndex.from_product(indexes, names=ordered_dimensions)

    return pd.DataFrame(OrderedDict(zip(columns, data)), index=index)
