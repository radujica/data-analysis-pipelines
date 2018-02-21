from grizzly.encoders import NumPyEncoder, NumPyDecoder, numpy_to_weld_type_mapping
from grizzly.lazy_op import LazyOpResult
from weld.types import WeldVec
from weld.weldobject import WeldObject
import numpy as np


class Variable(LazyOpResult):
    """ Weld-ed netCDF4.Variable.

    Functionality is currently (very) restricted to an example operation, printing, and evaluating.

    Parameters
    ----------
    ds : netCDF4.Dataset
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
    _encoder = NumPyEncoder()
    _decoder = NumPyDecoder()
    # _input_counter and _obj_input_mapping are used to keep track of which variables come from which raw data
    _input_counter = 0
    _obj_input_mapping = {}

    def __init__(self, ds, column_name, dimensions, attributes, expression, dtype):
        weld_type = WeldVec(numpy_to_weld_type_mapping[str(dtype)])
        super(Variable, self).__init__(expression, weld_type)

        self.ds = ds
        self.column_name = column_name
        self.dimensions = dimensions
        self.attributes = attributes
        self.dtype = dtype

        if not isinstance(expression, WeldObject):
            Variable._obj_input_mapping[Variable._input_counter] = expression
            Variable._input_counter += 1

    # TODO: this could be better; look at what netcdf4 does
    # TODO: look at scale_factor and update accordingly (like in netcdf4)
    def _read_data(self, start=0, end=None, stride=None):
        """ Eagerly reads data from file

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
        if end is None and stride is None:
            return self.ds.variables[self.column_name][start:]
        else:
            return self.ds.variables[self.column_name][start:end:stride]

    # TODO: this should encode start, end, stride in weld code ~ iter
    def __getitem__(self, item):
        pass

    # TODO: _read_data should take params!
    # TODO: maybe flatten can be avoided?
    def evaluate(self, verbose=True, decode=True, passes=None, num_threads=1,
                 apply_experimental_transforms=False):
        """ Evaluate the expression on this variable and read the data necessary

        Note that after 1 evaluate, the result is already materialized in context

        Returns
        -------
        np.array
            result of all the tracked computations

        See also
        --------
        WeldObject.evaluate

        """
        if isinstance(self.expression, WeldObject):
            for key in self.expression.context:
                data = self._read_data().flatten()
                # TODO: fix this HACK ~ about np masked array f32
                if self.dtype is np.dtype(np.int16):
                    data = data.astype(np.int16)
                self.expression.context[key] = data
            return super(Variable, self).evaluate(verbose, decode, passes, num_threads, apply_experimental_transforms)
        else:
            data = self._read_data().flatten()
            # if self.dtype is np.dtype(np.int32):
            #     data = data.astype(np.int32)
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
                                        'expression': self.expression}

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
                        self.column_name,
                        self.dimensions,
                        self.attributes,
                        self._element_wise_op(self.expression, value, '+'),
                        self.dtype)
