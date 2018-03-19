import netCDF4
import netCDF4_weld
from collections import OrderedDict
from pandas_weld import MultiIndex, DataFrame


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

    def read_dataset():
        return netCDF4.Dataset(path)

    ds = netCDF4_weld.Dataset(read_dataset)

    columns = [k for k in ds.variables if k not in ds.dimensions]
    dimensions = OrderedDict(map(lambda kv: (kv[0], kv[1]),
                                 OrderedDict(ds.dimensions.items()).items()))

    # columns data, either LazyData or raw
    data = [ds.process_column(k) for k in columns]
    # the dimensions
    indexes = [ds.process_dimension(k) for k in dimensions]

    index = MultiIndex.from_product(indexes, list(dimensions.keys()))

    return DataFrame(dict(zip(columns, data)), index)
