import netCDF4
import numpy as np
import pandas as pd
from grizzly.encoders import NumPyEncoder, NumPyDecoder, numpy_to_weld_type
from weld.weldobject import WeldObject

from lazy_data import LazyData
from lazy_result import LazyResult
from netCDF4_weld.utils import convert_row_to_nd_slices, replace_slice_defaults


class Variable(LazyData, LazyResult):
    """ Weld-ed netCDF4.Variable.

    Functionality is currently (very) restricted to an example operation, printing, and evaluating.

    Parameters
    ----------
    file_id : str
        generated by Dataset from FileMapping
    column_name : str
        the variable name in the dataset
    dimensions : tuple
        same as netCDF4.Variable.dimensions
    shape : tuple
        same as netCDF4.Variable.shape
    attributes : OrderedDict
        all Variable metadata
    expression : str or WeldObject
        str if created by netCDF4_weld.Dataset, else WeldObject tracking the computations created by operations
        on this variable; note that expression must be == column_name if created by Dataset!
    dtype : np.dtype
        type of the elements in this variable

    See also
    --------
    netCDF4.Variable

    """
    encoder = NumPyEncoder()
    decoder = NumPyDecoder()

    def __init__(self, file_id, column_name, dimensions, shape, attributes, expression, dtype):
        inferred_dtype = self._infer_dtype(dtype, attributes)
        weld_type = numpy_to_weld_type(inferred_dtype)
        LazyResult.__init__(self, expression, weld_type, 1)

        self.file_id = file_id
        self.column_name = column_name
        self.dimensions = dimensions
        self.shape = shape
        self.attributes = attributes
        # when reading data with netCDF4, the values are multiplied by the scale_factor if it exists,
        # which means that even if data is of type int, the scale factor is often float making the result a float
        self.dtype = inferred_dtype

        # same as [:]
        # the param used to lazy_slice_rows
        self.tuple_slices = slice(None)
        self._slice = None

    @staticmethod
    def _infer_dtype(dtype, attributes):
        # TODO: can it be float64?
        if 'scale_factor' in attributes:
            return np.dtype(np.float32)
        # calendar is stored as int in netCDF4, but we want the datetime format later which is encoded as a str(?)
        if 'calendar' in attributes:
            return np.dtype(np.str)
        else:
            return dtype

    def eager_read(self, slice_=None):
        ds = LazyResult.retrieve_file(self.file_id)

        # implemented like this to allow re-use of this method from eager_head
        if slice_ is None:
            slice_ = self.tuple_slices

        # want just np.array, no MaskedArray; let netCDF4 do the work of replacing missing values
        ds.variables[self.column_name].set_auto_mask(False)
        # the actual read from file call
        data = ds.variables[self.column_name][slice_]

        # TODO: transpose might be required when data variables have dimensions in a different order than the
        # dimensions declarations

        # want dimension = 1
        data = data.reshape(-1)

        attributes = ds.variables[self.column_name].__dict__
        # xarray creates a pandas DatetimeIndex with Timestamps (as it should); to save time however,
        # a shortcut is taken to convert netCDF4 python date -> pandas timestamp -> py datetime
        # TODO: weld pandas DatetimeIndex & Timestamp
        if 'calendar' in attributes:
            data = np.array([str(pd.Timestamp(k).date()) for k in netCDF4.num2date(data, attributes['units'],
                                                                                   calendar=attributes['calendar'])],
                            dtype=np.str)

        if self._slice is not None and self.column_name not in self.dimensions:
            return data[self._slice]
        else:
            return data

    def eager_head(self, n=10):
        tuple_slices = convert_row_to_nd_slices(slice(0, n, 1), self.shape)

        # bypass the cache and call directly
        return self.eager_read(slice_=tuple_slices)

    def lazy_skip_columns(self, columns):
        # nothing to do since netcdf is able to read specific columns only
        pass

    def lazy_slice_rows(self, slice_):
        # user wants a slice of rows, so convert to netCDF4 slices for all dimensions
        if isinstance(slice_, slice):
            slice_ = replace_slice_defaults(slice_)
            self._slice = slice_
            # self.tuple_slices = convert_row_to_nd_slices(slice_, self.shape)
        elif isinstance(slice_, tuple):  # assumed correct
            # self.tuple_slices = slice_
            pass
        else:
            raise TypeError('expected either slice or tuple of slices')

    def __repr__(self):
        return "{}(column_name={}, dtype={}, dimensions={}, attributes={})".format(self.__class__.__name__,
                                                                                   self.column_name,
                                                                                   self.dtype,
                                                                                   repr(self.dimensions),
                                                                                   repr(self.attributes))

    def __str__(self):
        return str(self.expr)

    # this and add are to show that one could also implement/do Weld operations at this level, not just in pandas
    def _element_wise_op(self, array, value, operation):
        weld_obj = WeldObject(Variable.encoder, Variable.decoder)

        array_var = weld_obj.update(array)

        if isinstance(array, WeldObject):
            array_var = array.obj_id
            weld_obj.dependencies[array_var] = array

        weld_template = """
        result(
            for(%(array)s, 
                appender[%(type)s], 
                |b: appender[%(type)s], i: i64, n: %(type)s| 
                    merge(b, n %(operation)s %(value)s)
            )
        )"""

        weld_obj.weld_code = weld_template % {'array': array_var,
                                              'value': value,
                                              'operation': operation,
                                              'type': numpy_to_weld_type(self.dtype)}

        return weld_obj

    def __add__(self, value):
        return Variable(self.file_id,
                        self.column_name,
                        self.shape,
                        self.dimensions,
                        self.attributes,
                        self._element_wise_op(self.expr, value, '+'),
                        self.dtype)