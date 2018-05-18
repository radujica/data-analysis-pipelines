from collections import OrderedDict

import numpy as np
from grizzly.encoders import numpy_to_weld_type

import csv_weld
import netCDF4_weld
from lazy_result import LazyResult
from pandas_weld import MultiIndex, DataFrame, Index
from pandas_weld.weld import weld_range


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
