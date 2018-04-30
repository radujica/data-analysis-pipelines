import netCDF4
import pandas as pd
from collections import OrderedDict


# need to go from netcdf to pandas DataFrame without transposing like xarray does.
# could either fork xarray and disable transposing or use the method below
def to_dataframe(path):
    """ NetCDF4.Dataset to pandas.DataFrame without transpose

    Parameters
    ----------
    path : str
        path to netcdf4 file to convert

    Returns
    -------
    pandas.DataFrame
    """
    ds = netCDF4.Dataset(path)

    columns = [k for k in ds.variables if k not in ds.dimensions]
    ordered_dimensions = OrderedDict(map(lambda kv: (kv[0], kv[1].size), ds.dimensions.items()))

    data = [ds.variables[k][:].reshape(-1) for k in columns]

    def convert_datetime(variable):
        return pd.to_datetime(netCDF4.num2date(variable[:], variable.units, calendar=variable.calendar))

    indexes = [convert_datetime(ds.variables[k]) if hasattr(ds.variables[k], 'calendar')
               else ds.variables[k][:] for k in ordered_dimensions]
    index = pd.MultiIndex.from_product(indexes, names=ordered_dimensions)

    return pd.DataFrame(OrderedDict(zip(columns, data)), index=index)
