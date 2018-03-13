from grizzly.encoders import NumPyEncoder, NumPyDecoder
from weld.weldobject import WeldObject
from lazy_data import LazyData
from utils import subset, replace_slice_defaults


class Series(LazyData):
    """ Weld-ed pandas Series

    Parameters
    ----------
    data : np.ndarray / WeldObject
        what shall be evaluated
    weld_type : WeldType
        of the elements
    data_id : str
        generated only by parsers to record the existence of new data from file; needs to be passed on
        to other LazyData children objects, e.g. when creating a pandas_weld.Series from netCDF4_weld.Variable

    See also
    --------
    pandas.Series

    """
    _encoder = NumPyEncoder()
    _decoder = NumPyDecoder()

    # TODO: should accept and store dtype instead; also see TODO @LazyData
    def __init__(self, data, weld_type, data_id=None):
        super(Series, self).__init__(data, weld_type, 1, data_id)

    def __getitem__(self, item):
        if not isinstance(item, slice):
            raise ValueError('expected a slice in Series.__getitem__')

        item = replace_slice_defaults(item)

        # update func_args so that less data is read from file
        if isinstance(self, LazyData) and self.data_id is not None:
            index = self.input_mapping.data_ids.index(self.data_id)
            old_args = self.input_mapping.input_function_args[index]
            slice_as_tuple = (slice(item.start, item.stop, item.step),)
            new_args = old_args + (slice_as_tuple,)
            self.input_mapping.update_input_function_args(index, new_args)

        return Series(subset(self, item).expr,
                      self.weld_type,
                      self.data_id)

    @staticmethod
    def _aggregate(array, operation, weld_type):
        """ Returns operation on the elements in the array.

        Arguments
        ---------
        array : WeldObject / np.ndarray
            input array
        operation : {'+'}
            operation to apply
        weld_type : WeldType
            type of each element in the input array

        Returns
        -------
        WeldObject
            representation of this computation

        """
        weld_obj = WeldObject(Series._encoder, Series._decoder)

        array_var = weld_obj.update(array)
        if isinstance(array, WeldObject):
            array_var = array.obj_id
            weld_obj.dependencies[array_var] = array

        weld_template = """
        result(
            for(
                %(array)s,
                merger[%(type)s,%(operation)s],
                |b, i, e| 
                    merge(b, e)
            )
        )"""

        weld_obj.weld_code = weld_template % {"array": array_var,
                                              "type": weld_type,
                                              "operation": operation}

        return weld_obj

    def sum(self):
        """ Sums all the elements

        Returns
        -------
        Series

        """
        return LazyData(
            Series._aggregate(self.expr,
                              "+",
                              self.weld_type),
            self.weld_type,
            0)
