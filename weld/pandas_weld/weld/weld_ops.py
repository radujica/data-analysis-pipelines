from grizzly.encoders import NumPyEncoder, NumPyDecoder
from weld.weldobject import WeldObject

_encoder = NumPyEncoder()
_decoder = NumPyDecoder()


# TODO all: types can be inferred with '?'; is the cost of doing it high?
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
    WeldObject
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

    weld_obj.weld_code = weld_template % {'array': array_var}

    return weld_obj


def weld_mean(array, weld_type):
    """ Returns the mean of the array

    Parameters
    ----------
    array : np.ndarray / WeldObject
        input array
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
    let sum = 
        for(
            %(array)s,
            merger[%(type)s, +],
            |b, i: i64, n: %(type)s|
                merge(b, n)
        );
    f64(result(sum)) / f64(len(%(array)s))"""

    weld_obj.weld_code = weld_template % {'array': array_var,
                                          'type': weld_type}

    return weld_obj


def weld_standard_deviation(array, weld_type):
    """ Returns the standard deviation of the array

    Parameters
    ----------
    array : np.ndarray / WeldObject
        input array
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

    # obtain the mean
    mean_obj = weld_mean(array, weld_type)
    # we know it's a registered WeldObject, no need to check
    weld_obj.update(mean_obj)
    mean_var = mean_obj.obj_id
    weld_obj.dependencies[mean_var] = mean_obj

    weld_template = """
    let numer = 
        for(
            %(array)s,
            merger[f64, +],
            |b, i, n|
                merge(b, pow(f64(n) - %(mean)s, 2.0))
        );
    let denom = len(%(array)s) - 1L;
    sqrt(result(numer) / f64(denom))"""

    weld_obj.weld_code = weld_template % {'array': array_var,
                                          'type': weld_type,
                                          'mean': mean_var}

    return weld_obj


# does NOT work correctly with duplicate elements; indexes MUST be sorted
def weld_merge_single_index(indexes):
    """ Returns bool arrays for which indexes shall be kept

    Parameters
    ----------
    indexes : list of np.array / WeldObject
        input array

    Returns
    -------
    [WeldObject]
        representation of the computations

    Examples
    -------
    >>> index1 = np.array([1, 3, 4, 5, 6])
    >>> index2 = np.array([2, 3, 5])
    >>> result = weld_merge_single_index([index1, index2])
    >>> LazyData(result[0], WeldBit(), 1).evaluate(verbose=False)
    [False True False True False]
    >>> LazyData(result[1], WeldBit(), 1).evaluate(verbose=False)
    [False True True]

    """
    weld_objects = []
    weld_ids = []

    for i in xrange(len(indexes)):
        weld_obj = WeldObject(_encoder, _decoder)

        array_var = weld_obj.update(indexes[i])
        if isinstance(indexes[i], WeldObject):
            array_var = indexes[i].obj_id
            weld_obj.dependencies[array_var] = indexes[i]

        weld_objects.append(weld_obj)
        weld_ids.append(array_var)

    weld_objects[0].update(weld_objects[1])
    weld_objects[0].dependencies[weld_ids[1]] = weld_objects[1]
    weld_objects[1].update(weld_objects[0])
    weld_objects[1].dependencies[weld_ids[0]] = weld_objects[0]

    weld_template = """
    let len1 = len(%(array1)s);
    let len2 = len(%(array2)s);
    let res = iterate({0L, 0L, appender[bool], appender[bool]},
            |p|
                let val1 = lookup(%(array1)s, p.$0);
                let val2 = lookup(%(array2)s, p.$1);
                {
                    if(val1 == val2,
                 
                        {p.$0 + 1L, p.$1 + 1L, merge(p.$2, true), merge(p.$3, true)},
                        
                        if(val1 < val2,  
                            {p.$0 + 1L, p.$1, merge(p.$2, false), p.$3},
                            {p.$0, p.$1 + 1L, p.$2, merge(p.$3, false)}
                        )
                    ),
                    if(val1 == val2, p.$0 + 1L, if(val1 < val2, p.$0 + 1L, p.$0)) < len1 && 
                    if(val1 == val2, p.$1 + 1L, if(val1 < val2, p.$1, p.$1 + 1L)) < len2
                }
    );
    # iterate over remaining un-checked elements in both arrays
    let res = if (res.$0 < len1, iterate(res,
            |p|
                {
                    {p.$0 + 1L, p.$1, merge(p.$2, false), p.$3},
                    p.$0 + 1L < len1
                }
    ), res);
    let res = if (res.$1 < len2, iterate(res,
            |p|
                {
                    {p.$0, p.$1 + 1L, p.$2, merge(p.$3, false)},
                    p.$1 + 1L < len2
                }
    ), res);
    res"""

    weld_objects[0].weld_code = 'result(' + weld_template % {'array1': weld_ids[0],
                                                             'array2': weld_ids[1]} + '.$2)'

    weld_objects[1].weld_code = 'result(' + weld_template % {'array1': weld_ids[0],
                                                             'array2': weld_ids[1]} + '.$3)'

    return weld_objects
