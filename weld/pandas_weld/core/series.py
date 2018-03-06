from grizzly.encoders import NumPyEncoder, NumPyDecoder
from weld.weldobject import WeldObject
from netCDF4_weld.lazy_data import LazyData


class Series(LazyData):
    """ Weld-ed pandas Series

    Parameters
    ----------
    expression : np.ndarray / WeldObject
        what shall be evaluated
    weld_type : WeldType
        of the elements
    dimension : int
        dimensionality of data
    data_id : str
        generated only by parsers to record the existence of new data from file; needs to be passed on
        to other LazyData children objects, e.g. when creating a pandas_weld.Series from netCDF4_weld.Variable

    """
    _encoder = NumPyEncoder()
    _decoder = NumPyDecoder()

    def __init__(self, expression, weld_type, dimension, data_id=None):
        super(Series, self).__init__(expression, weld_type, dimension, data_id)

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
