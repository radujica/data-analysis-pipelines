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
    operation : {'+', '*', 'min', 'max'}
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

    weld_obj.weld_code = weld_template % {'array': array_var,
                                          'type': weld_type,
                                          'operation': operation}

    return weld_obj


# TODO: replace with slice? apparently it exists
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


def weld_filter(array, bool_array, weld_type):
    """ Returns a new array only with the elements with a corresponding
    True in bool_array

    Parameters
    ----------
    array : np.ndarray / WeldObject
        input array
    bool_array : np.ndarray / WeldObject
        array of bool with True for elements in array desired in the result array
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

    bool_array_var = weld_obj.update(bool_array)
    if isinstance(bool_array, WeldObject):
        bool_array_var = bool_array.obj_id
        weld_obj.dependencies[bool_array_var] = bool_array

    weld_template = """
    result(
        for(
            zip(%(array)s, %(bool_array)s),
            appender[%(type)s],
            |b: appender[%(type)s], i: i64, e: {%(type)s, bool}| 
                if (e.$1, 
                    merge(b, e.$0), 
                    b)
        )
    )"""

    weld_obj.weld_code = weld_template % {'array': array_var,
                                          'bool_array': bool_array_var,
                                          'type': weld_type}

    return weld_obj


def weld_compare(array, scalar, operation, weld_type):
    """ Applies comparison operation between each element in the array with scalar

    Parameters
    ----------
    array : np.ndarray / WeldObject
        input array
    scalar : str or scalar type
        value to compare with; must be same type as the values in the array. If not a str,
        it is casted to weld_type (allowing one to write e.g. native Python int)
    operation : {<, <=, ==, !=, >=, >}
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

    # this means input is a native python literal, therefore need to cast to weld one
    if not isinstance(scalar, str):
        scalar = "%s(%s)" % (weld_type, str(scalar))

    weld_template = """
    map(
        %(array)s,
        |a: %(type)s| 
            a %(operation)s %(scalar)s
    )"""

    weld_obj.weld_code = weld_template % {'array': array_var,
                                          'scalar': scalar,
                                          'operation': operation,
                                          'type': weld_type}

    return weld_obj


# only int64 supported atm
def weld_range(start, stop, step):
    """ Create a vector for the range parameters above

    Parameters
    ----------
    start : int
    stop : int
    step : int

    Returns
    -------
    WeldObject
        representation of this computation

    """
    weld_obj = WeldObject(_encoder, _decoder)

    weld_template = """
    result(
        for(
            rangeiter(%(start)s, %(stop)s, %(step)s),
            appender[i64],
            |b: appender[i64], i: i64, e: i64| 
                merge(b, e)
        )
    )"""

    weld_obj.weld_code = weld_template % {'start': 'i64(%s)' % start,
                                          'stop': 'i64(%s)' % stop,
                                          'step': 'i64(%s)' % step}

    return weld_obj


def weld_element_wise_op(array, scalar, operation, weld_type):
    """ Applies operation to each element in the array with scalar

    Parameters
    ----------
    array : np.ndarray / WeldObject
        input array
    scalar : str or scalar type
        value to compare with; must be same type as the values in the array. If not a str,
        it is casted to weld_type (allowing one to write e.g. native Python int)
    operation : {+, -, *, /}
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

    # this means input is a native python literal, therefore need to cast to weld one
    if not isinstance(scalar, str):
        scalar = "%s(%s)" % (weld_type, str(scalar))

    weld_template = """
        result(
            for(%(array)s, 
                appender[%(type)s], 
                |b: appender[%(type)s], i: i64, n: %(type)s| 
                    merge(b, n %(operation)s %(value)s)
            )
        )"""

    weld_obj.weld_code = weld_template % {'array': array_var,
                                          'value': scalar,
                                          'operation': operation,
                                          'type': weld_type}

    return weld_obj


def weld_count(array):
    """ Returns the length of the array

    Parameters
    ----------
    array : np.ndarray / WeldObject
        input array

    Returns
    -------
        representation of this computation
    """
    weld_obj = WeldObject(_encoder, _decoder)

    array_var = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_var = array.obj_id
        weld_obj.dependencies[array_var] = array

    weld_template = """
    len(
        %(array)s
    )"""

    weld_obj.weld_code = weld_template % {"array": array_var}

    return weld_obj
