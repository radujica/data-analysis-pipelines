from grizzly.encoders import NumPyEncoder, NumPyDecoder, numpy_to_weld_type
from lazy_data import LazyData
from weld.weldobject import WeldObject
from netCDF4_weld.utils import convert_row_to_nd_slices
import numpy as np
import pandas as pd
import netCDF4


class Variable(LazyData):
    """ Weld-ed netCDF4.Variable.

    Functionality is currently (very) restricted to an example operation, printing, and evaluating.

    Parameters
    ----------
    read_file_func : netCDF4.Dataset
        the Dataset from which this variable originates
    column_name : str
        the variable name in the dataset
    dimensions : tuple
        same as netCDF4.Variable.dimensions
    attributes : OrderedDict
        all Variable metadata
    expression : str / WeldObject
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

    def __init__(self, read_file_func, column_name, data_id, dimensions, attributes, expression, dtype):
        inferred_dtype = self._infer_dtype(dtype, attributes)
        weld_type = numpy_to_weld_type(inferred_dtype)
        LazyData.__init__(self, expression, weld_type, 1, data_id,
                          self.read_data, (read_file_func, column_name))

        self.read_file_func = read_file_func
        self.column_name = column_name
        self.dimensions = dimensions
        self.attributes = attributes
        # when reading data with netCDF4, the values are multiplied by the scale_factor if it exists,
        # which means that even if data is of type int, the scale factor is often float making the result a float
        self.dtype = inferred_dtype

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

    @staticmethod
    def read_data(read_file_func, variable_name, tuple_slices=None):
        """ Reads data from file

        Once data is read, the result is stored. On subsequent reads, this stored data is returned.

        Parameters
        ----------
        read_file_func : func
            when called, it shall read the file; in this case, returning a netCDF4.Dataset from which 
            variable_name can be read and returned
        variable_name : str
            the name of the variable
        tuple_slices : ()
            of slices for selecting data

        Returns
        -------
        np.array
            raw data

        See also
        --------
        Python slicing

        """
        ds = read_file_func()

        if tuple_slices is not None:
            if not isinstance(tuple_slices, tuple):
                raise ValueError('expected a tuple of slices')

            for elem in tuple_slices:
                if not isinstance(elem, slice):
                    raise ValueError('expected slice in tuple_slices')

            # user wants a slice of rows, so convert to netCDF4 slices for all dimensions
            if len(tuple_slices) == 1:
                tuple_slices = convert_row_to_nd_slices(tuple_slices, ds.variables[variable_name].shape)
        else:
            # same as [:]
            tuple_slices = slice(None)

        # want just np.array, no MaskedArray; let netCDF4 do the work of replacing missing values
        ds.variables[variable_name].set_auto_mask(False)
        # the actual read from file call
        data = ds.variables[variable_name][tuple_slices]

        # TODO: transpose might be required when data variables have dimensions in a different order than the
        # dimensions declarations

        # want dimension = 1
        data = data.reshape(-1)

        attributes = ds.variables[variable_name].__dict__
        # xarray creates a pandas DatetimeIndex with Timestamps (as it should); to save time however,
        # a shortcut is taken to convert netCDF4 python date -> pandas timestamp -> py datetime
        # TODO: weld pandas DatetimeIndex & Timestamp
        if 'calendar' in attributes:
            data = np.array([str(pd.Timestamp(k).date()) for k in netCDF4.num2date(data, attributes['units'],
                                                                                   calendar=attributes['calendar'])],
                            dtype=np.str)

        return data

    def head(self, n=10):
        """ Read first n values eagerly

        Note this data is not cached

        Parameters
        ----------
        n : int
            how many values to read

        Returns
        -------
        np.array
            raw data

        """
        return self.read_data(self.read_file_func, self.column_name, (slice(0, n, 1),))

    # TODO: reduce printed expression context if materialized; can be looooooooooooooooong
    def __repr__(self):
        """ Only a descriptive representation; no data is read """
        string_representation = """variable: %(column_name)s, dtype: %(dtype)s
    dimensions: %(dimensions)s
    attributes: %(attributes)s
    expression: %(expression)s"""

        return string_representation % {'column_name': self.column_name,
                                        'dtype': self.dtype,
                                        'dimensions': self.dimensions,
                                        'attributes': self.attributes,
                                        'expression': self.expr}

    # this and add are for learning/testing purposes
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

    def add(self, value):
        return Variable(self.read_file_func,
                        self.column_name,
                        self.data_id,
                        self.dimensions,
                        self.attributes,
                        self._element_wise_op(self.expr, value, '+'),
                        self.dtype)
