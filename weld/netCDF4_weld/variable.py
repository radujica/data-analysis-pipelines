from grizzly.encoders import NumPyEncoder, NumPyDecoder, numpy_to_weld_type_mapping
from grizzly.lazy_op import LazyOpResult
from netCDF4 import num2date
from numpy.ma import MaskedArray
from weld.weldobject import WeldObject
import numpy as np


class Variable(LazyOpResult):
    """ Weld-ed netCDF4.Variable.

    Functionality is currently (very) restricted to an example operation, printing, and evaluating.

    Parameters
    ----------
    ds : netCDF4.Dataset
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
    # keep track of variable id -> Weld's _inpX assigned name
    _input_mapping = {}
    # cache the materialized, i.e. read, data; variable id -> data
    _materialized_columns = {}

    def __init__(self, ds, ds_id, column_name, dimensions, attributes, expression, dtype):
        inferred_dtype = self._infer_dtype(dtype, attributes)
        # IMO this should be WeldVec(...) HERE and not enforced during evaluate()
        weld_type = numpy_to_weld_type_mapping[str(inferred_dtype)]
        LazyOpResult.__init__(self, expression, weld_type, 1)

        self.ds = ds
        self.ds_id = ds_id
        # maybe worth it to make it a 'true' id
        self._id = ds_id + column_name
        self.column_name = column_name
        self.dimensions = dimensions
        self.attributes = attributes
        # when reading data with netCDF4, the values are multiplied by the scale_factor if it exists
        self.dtype = inferred_dtype

    # TODO: this doesn't seem robust; perhaps float64 is also possible?
    @staticmethod
    def _infer_dtype(dtype, attributes):
        if 'scale_factor' in attributes:
            return np.dtype(np.float32)
        else:
            return dtype

    # TODO: flatten/reshape(-1)-ing always might not be the best idea
    def _read_data(self, start=0, end=None, stride=None):
        """ Reads data from file

        Once data is read, the result is stored. On subsequent reads, this stored data is returned.

        Parameters
        ----------
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
        if self._id in self._materialized_columns:
            return self._materialized_columns[self._id]

        if end is None and stride is None:
            data = self.ds.variables[self.column_name][start:]
        else:
            data = self.ds.variables[self.column_name][start:end:stride]
        # remove MaskedArrays, just np.array please
        if isinstance(data, MaskedArray):
            data = data.filled(np.nan)
        # want dimension = 1
        data = data.reshape(-1)
        # if a datetime variable, want python's datetime
        if 'calendar' in self.attributes:
            data = num2date(data, self.attributes['units'], calendar=self.attributes['calendar'])
        # cache the data for further reads
        self._materialized_columns[self._id] = data

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
        return self._read_data(0, n, 1)

    # TODO: _read_data should take params!
    def evaluate(self, verbose=True, decode=True, passes=None, num_threads=1,
                 apply_experimental_transforms=False):
        """ Evaluate the expression on this variable and read the data necessar

        Returns
        -------
        np.array
            result of all the tracked computations

        See also
        --------
        WeldObject.evaluate

        """
        data = self._read_data()
        if isinstance(self.expr, WeldObject):
            self.expr.context[self._input_mapping[self._id]] = data
            return super(Variable, self).evaluate(verbose, decode, passes, num_threads, apply_experimental_transforms)
        else:
            return data

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
        # this means an _inpX id was returned for this column/variable; keep track of it
        if array_var is not None:
            self._input_mapping[self._id] = array_var

        if isinstance(array, WeldObject):
            array_var = array.obj_id
            weld_obj.dependencies[array_var] = array

        # TODO: remove result(?); result should only be called once in a pipeline iirc
        weld_template = """
        result(
            for(%(array)s, 
                appender[%(type)s], 
                |b: appender[%(type)s], i: i64, n: %(type)s| merge(b, n %(operation)s %(value)s)
            )
        )"""

        weld_obj.weld_code = weld_template % {'array': array_var,
                                              'value': value,
                                              'operation': operation,
                                              'type': numpy_to_weld_type_mapping[str(self.dtype)]}

        return weld_obj

    def add(self, value):
        return Variable(self.ds,
                        self.ds_id,
                        self.column_name,
                        self.dimensions,
                        self.attributes,
                        self._element_wise_op(self.expr, value, '+'),
                        self.dtype)
