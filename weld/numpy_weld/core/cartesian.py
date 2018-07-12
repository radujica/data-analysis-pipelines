import numpy as np
from grizzly.encoders import NumPyEncoder, NumPyDecoder, numpy_to_weld_type
from weld.types import WeldLong
from weld.weldobject import WeldObject

from lazy_result import LazyResult

# the methods are only intended to work with numpy, so have a single encoder/decoder
_encoder = NumPyEncoder()
_decoder = NumPyDecoder()

# TODO: could generalize to return either values or indices


def _duplicate_elements_indices(array, n, weld_type, cartesian=False):
    weld_obj = WeldObject(_encoder, _decoder)

    array_var = weld_obj.update(array)

    if isinstance(array, WeldObject):
        array_var = array.obj_id
        weld_obj.dependencies[array_var] = array

    if isinstance(n, WeldObject):
        weld_obj.update(n)
        weld_obj.dependencies[n.obj_id] = n
        n = 'len(%s)' % n.obj_id
    elif isinstance(n, np.ndarray):
        array_var = weld_obj.update(n)
        n = 'len(%s)' % array_var

    weld_template = """
    result(
        for(
            %(array)s,
            appender[i64],
            |b: appender[i64], i: i64, n: %(type)s| 
                iterate(
                    {b, %(index_or_value)s, 1L}, 
                    |p| 
                        {{merge(b, p.$1), p.$1, p.$2 + 1L}, 
                        p.$2 < %(n)s}
                ).$0
        )  
    )"""

    weld_obj.weld_code = weld_template % {'array': array_var,
                                          'n': 'i64(%s)' % n,
                                          'index_or_value': 'n' if cartesian else 'i',
                                          'type': 'i64' if cartesian else weld_type}

    return weld_obj


# TODO: duplicate elements don't work!
def duplicate_elements_indices(array, n, cartesian=False):
    """ Expands array by multiplying each element n times

    Parameters
    ----------
    array : np.ndarray or LazyResult
        the source data
    n : long or LazyResult
        how many times to repeat each element; if LazyResult, will use its length
    cartesian : bool
        True if used internally by cartesian_product to signify the operation
        has been done once already and hence must behave slightly different by using the number
        in the array instead of the index of that number (since at this point the array already contains indexes)

    Returns
    -------
    LazyResult
        the expanded array containing the indices, not the elements

    Examples
    --------
    >>> duplicate_elements_indices(np.array([1, 2, 3]), 2)
    [0, 0, 1, 1, 2, 2]

    """
    if isinstance(array, LazyResult):
        weld_type = array.weld_type
        array = array.expr
    elif isinstance(array, np.ndarray):
        weld_type = numpy_to_weld_type(array.dtype)
    else:
        raise NotImplementedError

    if isinstance(n, LazyResult):
        n = n.expr
    elif isinstance(n, np.ndarray):
        n = len(n)
    elif not isinstance(n, long):
        raise TypeError('expected either a long value or a LazyResult to use its length')

    return LazyResult(_duplicate_elements_indices(array, n, weld_type, cartesian),
                      WeldLong(),
                      1)


def _duplicate_array_indices(array, n, weld_type, cartesian=False):
    weld_obj = WeldObject(_encoder, _decoder)

    array_var = weld_obj.update(array)

    if isinstance(array, WeldObject):
        array_var = array.obj_id
        weld_obj.dependencies[array_var] = array

    if isinstance(n, WeldObject):
        weld_obj.update(n)
        weld_obj.dependencies[n.obj_id] = n
        n = 'len(%s)' % n.obj_id
    elif isinstance(n, np.ndarray):
        array_var = weld_obj.update(n)
        n = 'len(%s)' % array_var

    weld_template = """   
    result(
        iterate(
            {appender[i64], %(array)s, 1L},
            |p| 
                {{for(
                    p.$1,
                    p.$0,
                    |b: appender[i64], i: i64, n: %(type)s|
                        merge(b, %(index_or_value)s)
                ), p.$1, p.$2 + 1L},
                p.$2 < %(n)s}
        ).$0
    )"""

    weld_obj.weld_code = weld_template % {'array': array_var,
                                          'n': 'i64(%s)' % n,
                                          'index_or_value': 'n' if cartesian else 'i',
                                          'type': 'i64' if cartesian else weld_type}

    return weld_obj


# TODO: duplicate elements don't work!
def duplicate_array_indices(array, n, cartesian=False):
    """ Duplicate array n times

    Parameters
    ----------
    array : np.ndarray or LazyResult
        the source data
    n : long or LazyResult
        how many times to repeat the source array; if LazyResult, will use its length
    cartesian : bool
        True if used internally by cartesian_product to signify the operation
        has been done once already and hence must behave slightly different by using the number
        in the array instead of the index of that number (since at this point the array already contains indexes)

    Returns
    -------
    LazyResult
        the expanded array containing the indices, not the elements

    Examples
    --------
    >>> duplicate_array_indices(np.array([1, 2, 3]), 2)
    [0, 1, 2, 0, 1, 2]

    """
    if isinstance(array, LazyResult):
        weld_type = array.weld_type
        array = array.expr
    elif isinstance(array, np.ndarray):
        weld_type = numpy_to_weld_type(array.dtype)
    else:
        raise NotImplementedError

    if isinstance(n, LazyResult):
        n = n.expr
    elif isinstance(n, np.ndarray):
        n = len(n)
    elif not isinstance(n, long):
        raise TypeError('expected either a long value or a LazyResult to use its length')

    return LazyResult(_duplicate_array_indices(array, n, weld_type, cartesian),
                      WeldLong(),
                      1)


def _cartesian_product_indices(arrays):
    # compute (lazily) the x resulting columns
    results = [0] * len(arrays)
    results[0] = duplicate_elements_indices(arrays[0], arrays[1])
    results[1] = duplicate_array_indices(arrays[1], arrays[0])

    for i in range(2, len(arrays)):
        for j in range(0, i):
            results[j] = duplicate_elements_indices(results[j], arrays[i], cartesian=True)

        results[i] = duplicate_array_indices(arrays[i], arrays[0])

        for j in range(1, i):
            results[i] = duplicate_array_indices(results[i], arrays[j], cartesian=True)

    # final object
    weld_obj = WeldObject(_encoder, _decoder)
    # add the columns as dependencies to the final output
    for result in results:
        weld_obj.update(result.expr)
        weld_obj.dependencies[result.expr.obj_id] = result.expr

    # construct the template for a single vec[vec[i64]] which will result in a np.ndarray of ndim=2
    weld_template = 'let res = {%s};\n' % ', '.join([res.expr.obj_id for res in results])
    for i in range(len(results) + 1):
        line = 'let a_%s = ' % str(i)
        if i == 0:
            line += 'appender[vec[i64]];\n'
        else:
            index = str(i - 1)
            line += 'merge(a_%s, res.$%s);\n' % (index, index)
        weld_template += line
    weld_template += 'result(a_%s)\n' % str(len(results))

    # no other replacements needed
    weld_obj.weld_code = weld_template

    return weld_obj


# TODO: duplicate elements don't work!
def cartesian_product_indices(arrays, cache=True):
    """ Performs cartesian product between all arrays

    Returns the indices instead of the actual values

    Parameters
    ----------
    arrays : list of (np.ndarray or LazyResult)
        list containing arrays that need to be in the product
    cache : bool, optional
        flag to indicate whether to cache result as intermediate result

    Returns
    -------
    list of LazyResult

    Examples
    --------
    >>> cartesian_product_indices([np.array([1, 2]), np.array([3, 4])])
    [[0, 0, 1, 1], [0, 1, 0, 1]]

    See also
    --------
    pandas.MultiIndex

    """
    if len(arrays) < 2:
        raise ValueError('expected at least 2 arrays')

    weld_object = _cartesian_product_indices(arrays)
    # this now contains the entire np.ndarray with all results of cartesian product
    result = LazyResult(weld_object, WeldLong(), 2)

    # construct the actual weld_objects corresponding to single result columns/arrays
    weld_objects = []
    weld_ids = []
    if cache:
        id_ = LazyResult.generate_intermediate_id('cartesian_product')
        weld_input_name = WeldObject.generate_input_name(id_)
        LazyResult.register_intermediate_result(weld_input_name, result)

        for i in range(len(arrays)):
            weld_obj = WeldObject(_encoder, _decoder)

            result_var = weld_obj.update(id_)
            assert result_var is not None

            weld_objects.append(weld_obj)
            weld_ids.append(result_var)
    else:
        for i in range(len(arrays)):
            weld_obj = WeldObject(_encoder, _decoder)

            result_var = weld_obj.update(result.expr)
            assert result_var is None
            result_var = result.expr.obj_id
            weld_obj.dependencies[result_var] = result.expr

            weld_objects.append(weld_obj)
            weld_ids.append(result_var)

    weld_template = """lookup(%(array)s, %(i)sL)"""
    for i in range(len(arrays)):
        weld_objects[i].weld_code = weld_template % {'array': weld_ids[i],
                                                     'i': str(i)}

    return [LazyResult(obj, WeldLong(), 1) for obj in weld_objects]


def _array_to_labels(array, levels, levels_type):
    weld_obj = WeldObject(_encoder, _decoder)

    array_var = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_var = array.obj_id
        weld_obj.dependencies[array_var] = array

    levels_var = weld_obj.update(levels)
    if isinstance(levels, WeldObject):
        levels_var = levels.obj_id
        weld_obj.dependencies[levels_var] = levels

    weld_template = """
    let indices = result(
        for(%(levels)s,
            appender[i64],
            |b, i, e|
                merge(b, i)
        )
    );
    let indices_dict = result(
        for(zip(%(levels)s, indices),
            dictmerger[%(type)s, i64, +],
            |b, i, e|
                merge(b, {e.$0, e.$1})        
        )
    );
    result(   
        for(
            %(array)s,
            appender[i64],
            |b, i, e|
                merge(b, lookup(indices_dict, e))
        )
    )"""

    weld_obj.weld_code = weld_template % {'array': array_var,
                                          'levels': levels_var,
                                          'type': levels_type}

    return weld_obj


# TODO: does NOT work with duplicates!
# TODO: this doesn't really belong here
def array_to_labels(array, levels, levels_type):
    """ Extracts the indices of the values in the array

    Parameters
    ----------
    array : np.ndarray or LazyResult
        the source data
    levels : np.ndarray or LazyResult
        the unique items from the array, currently sorted by default (see TODOs)
    levels_type : WeldType
        of the levels

    Returns
    -------
    LazyResult
        the labels for MultiIndex

    """
    if isinstance(array, LazyResult):
        array = array.expr

    if isinstance(levels, LazyResult):
        levels = levels.expr

    return LazyResult(_array_to_labels(array, levels, levels_type),
                      WeldLong(),
                      1)
