from grizzly.encoders import NumPyEncoder, NumPyDecoder, numpy_to_weld_type_mapping
from numpy.ma import MaskedArray
from lazy_data import LazyData
from weld.weldobject import WeldObject
import numpy as np
import netCDF4


class Variable(LazyData):
    """ Weld-ed netCDF4.Variable.

    Functionality is currently (very) restricted to an example operation, printing, and evaluating.

    Parameters
    ----------
    read_file_func : netCDF4.Dataset
        the Dataset from which this variable originates
    ds_id : int
        links this variable to a specific Dataset based on the Dataset's id
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
    _encoder = NumPyEncoder()
    _decoder = NumPyDecoder()

    def __init__(self, read_file_func, ds_id, column_name, dimensions, attributes, expression, dtype):
        inferred_dtype = self._infer_dtype(dtype, attributes)
        # IMO this should be WeldVec(...) HERE and not enforced during evaluate()
        weld_type = numpy_to_weld_type_mapping[str(inferred_dtype)]
        data_id = self._create_data_id(ds_id, column_name)
        LazyData.__init__(self, expression, weld_type, 1, data_id, self.read_data, (read_file_func, column_name))

        self.read_file_func = read_file_func
        self.ds_id = ds_id
        self.column_name = column_name
        self.dimensions = dimensions
        self.attributes = attributes
        # when reading data with netCDF4, the values are multiplied by the scale_factor if it exists,
        # which means that even if data is of type int, the scale factor is often float making the result a float
        self.dtype = inferred_dtype

    # TODO: see LazyData TODO about moving id generation
    @staticmethod
    def _create_data_id(ds_id, column_name):
        return ds_id + '_' + column_name

    # TODO: this doesn't seem robust; perhaps float64 is also possible?
    @staticmethod
    def _infer_dtype(dtype, attributes):
        if 'scale_factor' in attributes:
            return np.dtype(np.float32)
        else:
            return dtype

    # TODO: flatten/reshape(-1)-ing always might not be the best idea
    @staticmethod
    def read_data(read_file_func, variable_name, start=0, end=None, stride=None):
        """ Reads data from file

        Once data is read, the result is stored. On subsequent reads, this stored data is returned.

        Parameters
        ----------
        read_file_func : func
            when called, it shall read the file; in this case, returning a netCDF4.Dataset from which 
            variable_name can be read and returned
        variable_name : str
            the name of the variable
        start : int
            index to start reading data from
        end : int
            where to stop
        stride : int
            how much to jump between each read

        Returns
        -------
        np.array
            raw data

        See also
        --------
        Python slicing

        """
        ds = read_file_func()

        if end is None and stride is None:
            data = ds.variables[variable_name][start:]
        else:
            data = ds.variables[variable_name][start:end:stride]
        # remove MaskedArrays, just np.array please
        if isinstance(data, MaskedArray):
            data = data.filled(np.nan)
        # want dimension = 1
        data = data.reshape(-1)
        # if a datetime variable, want python's datetime
        attributes = ds.variables[variable_name].__dict__
        if 'calendar' in attributes:
            data = netCDF4.num2date(data, attributes['units'], calendar=attributes['calendar'])

        return data

    # TODO: this should encode start, end, stride in weld code ~ iter
    def __getitem__(self, item):
        pass

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
        return self.read_data(self.read_file_func, self.column_name, 0, n, 1)

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
        weld_obj = WeldObject(self._encoder, self._decoder)

        array_var = weld_obj.update(array)

        if isinstance(array, WeldObject):
            array_var = array.obj_id
            weld_obj.dependencies[array_var] = array

        # TODO: remove result(?); result should only be called once in a pipeline iirc
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
                                              'type': numpy_to_weld_type_mapping[str(self.dtype)]}

        return weld_obj

    def add(self, value):
        return Variable(self.read_file_func,
                        self.ds_id,
                        self.column_name,
                        self.dimensions,
                        self.attributes,
                        self._element_wise_op(self.expr, value, '+'),
                        self.dtype)
