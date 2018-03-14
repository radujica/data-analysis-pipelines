from grizzly.encoders import NumPyEncoder, NumPyDecoder
from weld.weldobject import WeldObject

_encoder = NumPyEncoder()
_decoder = NumPyDecoder()


def weld_aggregate(array, operation, weld_type):
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
    weld_obj = WeldObject(_encoder, _decoder)

    array_var = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_var = array.obj_id
        weld_obj.dependencies[array_var] = array

    weld_template = """
    result(
        for(
            %(array)s,
            merger[%(type)s, %(operation)s],
            |b, i, e| 
                merge(b, e)
        )
    )"""

    weld_obj.weld_code = weld_template % {"array": array_var,
                                          "type": weld_type,
                                          "operation": operation}

    return weld_obj


def weld_subset(array, slice_, weld_type):
    """ Return a subset of the input array

    Parameters
    ----------
    array : np.array / WeldObject
        1-dimensional array
    slice_ : slice
        subset to return
    weld_type : WeldType
        type of each element in the array

    Returns
    -------
    WeldObject
        representation of this computation

    """
    weld_obj = WeldObject(_encoder, _decoder)

    array_var = weld_obj.update(array)

    if isinstance(array, WeldObject):
        array_var = array.obj_id
        weld_obj.dependencies[array_var] = array

    weld_template = """
        result(
            for(
                iter(%(array)s, %(slice_start)s, %(slice_stop)s, %(slice_step)s),
                appender[%(type)s],
                |b: appender[%(type)s], i: i64, n: %(type)s| 
                    merge(b, n)
            )  
        )"""

    weld_obj.weld_code = weld_template % {'array': array_var,
                                          'type': weld_type,
                                          'slice_start': '%sL' % slice_.start,
                                          'slice_stop': '%sL' % slice_.stop,
                                          'slice_step': '%sL' % slice_.step}

    return weld_obj
