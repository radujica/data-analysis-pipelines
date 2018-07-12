from collections import OrderedDict

import numpy as np
from grizzly.encoders import numpy_to_weld_type

import csv_weld
import netCDF4_weld
import netCDF4_weld_eager
from lazy_result import LazyResult
from pandas_weld import MultiIndex, DataFrame, Index
from pandas_weld.weld import weld_range


def read_netcdf4_eager(path):
    """ Read eagerly a netcdf4 file as a DataFrame

    Parameters
    ----------
    path : str
        path of the file

    Returns
    -------
    DataFrame

    """
    # This is how it should look like but currently the data is corrupted somehow
    # import netCDF4
    # import pandas as pd
    # ds = netCDF4.Dataset(path)
    #
    # [ds.variables[k].set_auto_mask(False) for k in ds.variables]
    #
    # columns = [k for k in ds.variables if k not in ds.dimensions]
    # ordered_dimensions = OrderedDict(map(lambda kv: (kv[0], kv[1].size), OrderedDict(ds.dimensions.items()).items()))
    #
    # data = [ds.variables[k][:].reshape(-1) for k in columns]
    #
    # def convert_datetime(variable):
    #     return np.array([str(pd.Timestamp(d).date()) for d in netCDF4.num2date(variable[:],
    #                                                                            variable.units,
    #                                                                            calendar=variable.calendar)],
    #                     dtype=np.str)
    #
    # indexes = [convert_datetime(ds.variables[k]) if hasattr(ds.variables[k], 'calendar')
    #            else ds.variables[k][:] for k in ordered_dimensions]
    # index = MultiIndex.from_product(indexes, names=list(ordered_dimensions.keys()))
    #
    # return DataFrame(dict(zip(columns, data)), index)

    ds = netCDF4_weld_eager.Dataset(path)

    columns = [k for k in ds.variables if k not in ds.dimensions]
    dimensions = OrderedDict(map(lambda kv: (kv[0], kv[1]),
                                 OrderedDict(ds.dimensions.items()).items()))

    # columns data, either LazyResult or raw
    data = [ds.variables[k] for k in columns]
    # the dimensions
    indexes = [ds.variables[k] for k in dimensions]

    index = MultiIndex.from_product(indexes, list(dimensions.keys()))

    return DataFrame(dict(zip(columns, data)), index)


def read_netcdf4(path):
    """ Read a netcdf4 file as a DataFrame

    Parameters
    ----------
    path : str
        path of the file

    Returns
    -------
    DataFrame

    """
    ds = netCDF4_weld.Dataset(path)

    columns = [k for k in ds.variables if k not in ds.dimensions]
    dimensions = OrderedDict(map(lambda kv: (kv[0], kv[1]),
                                 OrderedDict(ds.dimensions.items()).items()))

    # columns data, either LazyResult or raw
    data = [ds.variables[k] for k in columns]
    # the dimensions
    indexes = [ds.variables[k] for k in dimensions]

    index = MultiIndex.from_product(indexes, list(dimensions.keys()))

    return DataFrame(dict(zip(columns, data)), index)


def read_csv(path):
    """ Read a csv file as a DataFrame

    Parameters
    ----------
    path : str
        path of the file

    Returns
    -------
    DataFrame

    """
    table = csv_weld.Table(path)

    new_columns = {}
    for column_name in table.columns:
        column = table.columns[column_name]
        weld_obj = LazyResult.generate_placeholder_weld_object(column.data_id, column.encoder, column.decoder)
        new_columns[column_name] = LazyResult(weld_obj, numpy_to_weld_type(column.dtype), 1)

    random_column = new_columns[new_columns.keys()[0]]
    index_weld_obj = weld_range(0, 'len({})'.format(random_column.expr.weld_code), 1)
    index_weld_obj.update(random_column.expr)

    return DataFrame(new_columns, Index(index_weld_obj, np.dtype(np.int64)))
