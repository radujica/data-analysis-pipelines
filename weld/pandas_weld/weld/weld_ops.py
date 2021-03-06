import numpy as np
from grizzly.encoders import NumPyEncoder, NumPyDecoder
from weld.types import WeldBit
from weld.weldobject import WeldObject

from lazy_result import LazyResult

_encoder = NumPyEncoder()
_decoder = NumPyDecoder()


def weld_aggregate(array, operation, weld_type):
    """ Returns operation on the elements in the array.

    Arguments
    ---------
    array : WeldObject or np.ndarray
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


def weld_filter(array, bool_array):
    """ Returns a new array only with the elements with a corresponding
    True in bool_array

    Parameters
    ----------
    array : np.ndarray or WeldObject
        input array
    bool_array : np.ndarray / WeldObject
        array of bool with True for elements in array desired in the result array

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
            appender,
            |b, i, e| 
                if (e.$1, 
                    merge(b, e.$0), 
                    b)
        )
    )"""

    weld_obj.weld_code = weld_template % {'array': array_var,
                                          'bool_array': bool_array_var}

    return weld_obj


def weld_compare(array, scalar, operation, weld_type):
    """ Applies comparison operation between each element in the array with scalar

    Parameters
    ----------
    array : np.ndarray or WeldObject
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
    array : np.ndarray or WeldObject
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
            appender, 
            |b, i, n| 
                merge(b, n %(operation)s %(scalar)s)
        )
    )"""

    weld_obj.weld_code = weld_template % {'array': array_var,
                                          'scalar': scalar,
                                          'operation': operation}

    return weld_obj


def weld_array_op(array1, array2, operation):
    """ Applies operation to each element in the array with scalar

    Their lengths and types are assumed to be the same.
    TODO: what happens if not?

    Parameters
    ----------
    array1 : np.ndarray or WeldObject
        input array
    array2 : np.ndarray or WeldObject
        second input array
    operation : {+, -, *, /, &&, ||}

    Returns
    -------
    WeldObject
        representation of this computation

    """
    weld_obj = WeldObject(_encoder, _decoder)

    array1_var = weld_obj.update(array1)
    if isinstance(array1, WeldObject):
        array1_var = array1.obj_id
        weld_obj.dependencies[array1_var] = array1

    array2_var = weld_obj.update(array2)
    if isinstance(array2, WeldObject):
        array2_var = array2.obj_id
        weld_obj.dependencies[array2_var] = array2

    weld_template = """
    result(
        for(zip(%(array1)s, %(array2)s), 
            appender, 
            |b, i, n| 
                merge(b, n.$0 %(operation)s n.$1)
        )
    )"""

    weld_obj.weld_code = weld_template % {'array1': array1_var,
                                          'array2': array2_var,
                                          'operation': operation}

    return weld_obj


def weld_count(array):
    """ Returns the length of the array

    Parameters
    ----------
    array : np.ndarray or WeldObject
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
    array : np.ndarray or WeldObject
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
    f64(
        result(
            for(
                %(array)s,
                merger[%(type)s, +],
                |b, i, n|
                    merge(b, n)
            )
        )
    ) / f64(len(%(array)s))"""

    weld_obj.weld_code = weld_template % {'array': array_var,
                                          'type': weld_type}

    return weld_obj


def weld_standard_deviation(array, weld_type):
    """ Returns the standard deviation of the array

    Parameters
    ----------
    array : np.ndarray or WeldObject
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
    sqrt(
        result(
            for(
                %(array)s,
                merger[f64, +],
                |b, i, n|
                    merge(b, pow(f64(n) - %(mean)s, 2.0))
            )
        ) / f64(len(%(array)s) - 1L)
    )"""

    weld_obj.weld_code = weld_template % {'array': array_var,
                                          'type': weld_type,
                                          'mean': mean_var}

    return weld_obj


# does NOT work correctly with duplicate elements; indexes MUST be sorted
def weld_merge_single_index(indexes, cache=True):
    """ Returns bool arrays for which indexes shall be kept

    Parameters
    ----------
    indexes : list of np.array or WeldObject
        input array
    cache : bool
        flag to indicate whether to cache result as intermediate result

    Returns
    -------
    list of WeldObject
        representation of the computations

    Examples
    -------
    >>> index1 = np.array([1, 3, 4, 5, 6])
    >>> index2 = np.array([2, 3, 5])
    >>> result = weld_merge_single_index([index1, index2])
    >>> LazyResult(result[0], WeldBit(), 1).evaluate(verbose=False)
    [False True False True False]
    >>> LazyResult(result[1], WeldBit(), 1).evaluate(verbose=False)
    [False True True]

    """
    weld_obj = WeldObject(_encoder, _decoder)
    weld_ids = []
    for array in indexes:
        array_var = weld_obj.update(array)
        if isinstance(array, WeldObject):
            array_var = array.obj_id
            weld_obj.dependencies[array_var] = array
        weld_ids.append(array_var)

    weld_template = """
    let len1 = len(%(array1)s);
    let len2 = len(%(array2)s);
    # bool arrays shall be padded until maxLen so that result can be cached as np.ndarray of ndim=2
    let maxlen = if(len1 > len2, len1, len2);
    let res = iterate({0L, 0L, appender[bool], appender[bool]},
            |p|
                let val1 = lookup(%(array1)s, p.$0);
                let val2 = lookup(%(array2)s, p.$1);
                let iter_output = 
                    if(val1 == val2,
                        {p.$0 + 1L, p.$1 + 1L, merge(p.$2, true), merge(p.$3, true)},
                        if(val1 < val2,  
                            {p.$0 + 1L, p.$1, merge(p.$2, false), p.$3},
                            {p.$0, p.$1 + 1L, p.$2, merge(p.$3, false)}
                        )
                    );
                    
                {
                    iter_output,
                    iter_output.$0 < len1 && 
                    iter_output.$1 < len2
                }
    );
    # iterate over remaining un-checked elements in both arrays
    let res = if (res.$0 < maxlen, iterate(res,
            |p|
                {
                    {p.$0 + 1L, p.$1, merge(p.$2, false), p.$3},
                    p.$0 + 1L < maxlen
                }
    ), res);
    let res = if (res.$1 < maxlen, iterate(res,
            |p|
                {
                    {p.$0, p.$1 + 1L, p.$2, merge(p.$3, false)},
                    p.$1 + 1L < maxlen
                }
    ), res);
    let b = appender[vec[bool]];
    let c = merge(b, result(res.$2));
    result(merge(c, result(res.$3)))"""

    weld_obj.weld_code = weld_template % {'array1': weld_ids[0],
                                          'array2': weld_ids[1]}
    # this has both required bool arrays into 1 ndarray; note that arrays have been padded with False until of same len
    # TODO: this could still be a single vec/array with the arrays concatenated instead to avoid decoder with ndim=2 mallocs
    result = LazyResult(weld_obj, WeldBit(), 2)

    # creating the actual results to return
    weld_objects = []
    weld_ids = []
    weld_col_ids = []

    if cache:
        id_ = LazyResult.generate_intermediate_id('sindex_merge')
        weld_input_id = WeldObject.generate_input_name(id_)
        LazyResult.register_intermediate_result(weld_input_id, result)

        for i in range(2):
            weld_obj = WeldObject(_encoder, _decoder)

            result_var = weld_obj.update(id_)
            assert result_var is not None

            weld_objects.append(weld_obj)
            weld_ids.append(result_var)
    else:
        for i in range(2):
            weld_obj = WeldObject(_encoder, _decoder)

            result_var = weld_obj.update(result.expr)
            assert result_var is None
            result_var = result.expr.obj_id
            weld_obj.dependencies[result_var] = result.expr

            weld_objects.append(weld_obj)
            weld_ids.append(result_var)

    # need 1 array from each resulting tables to get actual length
    for i in range(2):
        array_var = weld_objects[i].update(indexes[i])
        if isinstance(indexes[i], WeldObject):
            array_var = indexes[i].obj_id
            weld_objects[i].dependencies[array_var] = indexes[i]
        weld_col_ids.append(array_var)

    weld_templ = """slice(lookup(%(array)s, %(i)s), 0L, len(%(col)s))"""

    for i in range(2):
        weld_objects[i].weld_code = weld_templ % {'array': weld_ids[i],
                                                  'i': str(i) + 'L',
                                                  'col': weld_col_ids[i]}

    return weld_objects


def weld_index_to_values(levels, labels):
    """ Construct the actual index from levels ('values') and labels ('indices')

    Parameters
    ----------
    levels : np.array or WeldObject
        the possible values
    labels : np.array or WeldObject
        the indices to the levels for the actual index values

    Returns
    -------
    WeldObject
        representation of the computation

    Examples
    --------
    >>> levels = np.array([1.0, 2.5, 3.0])
    >>> labels = np.array([0, 0, 1, 2])
    >>> print(LazyResult(weld_index_to_values(levels, labels), WeldDouble(), 1).evaluate(verbose=False))
    [1. 1. 2.5 3.]

    """
    # TODO: fix this temporary hack
    if isinstance(levels, np.ndarray) and levels.dtype == 'object':
        levels = levels.astype(np.str)

    weld_obj = WeldObject(_encoder, _decoder)

    levels_var = weld_obj.update(levels)
    if isinstance(levels, WeldObject):
        levels_var = levels.obj_id
        weld_obj.dependencies[levels_var] = levels

    labels_var = weld_obj.update(labels)
    if isinstance(labels, WeldObject):
        labels_var = labels.obj_id
        weld_obj.dependencies[labels_var] = labels

    weld_template = """
    result(
        for(
            %(labels)s,
            appender,
            |b, i, n|
                merge(b, lookup(%(levels)s, n))
        )
    )"""

    weld_obj.weld_code = weld_template % {'labels': labels_var,
                                          'levels': levels_var}

    return weld_obj


# TODO: generify this
def weld_merge_triple_index(indexes, cache=True):
    """ Returns bool arrays for which indexes shall be kept

    Note it does NOT work correctly with duplicate elements; indexes MUST be already sorted

    Parameters
    ----------
    indexes : list of list
        of np.array or WeldObject
        list of len 2 with first and second elements being the labels in a list
        for the first and second DataFrame MultiIndex, respectively
    cache : bool
        flag to indicate whether to cache result as intermediate result

    Returns
    -------
    list of WeldObject
        representation of the computations, one for each DataFrame

    """
    assert len(indexes) == 2
    assert len(indexes[0]) == len(indexes[1]) == 3

    # flatten the list
    indexes = [elem for sublist in indexes for elem in sublist]

    # create final weld objects of what will be the bool arrays
    # also save the weld_ids for the inputs
    weld_obj = WeldObject(_encoder, _decoder)
    weld_ids = []

    for array in indexes:
        array_var = weld_obj.update(array)
        if isinstance(array, WeldObject):
            array_var = array.obj_id
            weld_obj.dependencies[array_var] = array

        weld_ids.append(array_var)

    weld_template = """
    let len1 = len(%(array1)s);
    let len2 = len(%(array4)s);
    # bool arrays shall be padded until maxLen so that result can be cached as np.ndarray of ndim=2
    let maxlen = if(len1 > len2, len1, len2);
    let indexes1 = {%(array1)s, %(array2)s, %(array3)s};
    let indexes2 = {%(array4)s, %(array5)s, %(array6)s};
    let res = if(len1 > 0L && len2 > 0L,
                iterate({0L, 0L, appender[bool], appender[bool]},
                |p|
                    let val1 = {lookup(indexes1.$0, p.$0), lookup(indexes1.$1, p.$0), lookup(indexes1.$2, p.$0)};
                    let val2 = {lookup(indexes2.$0, p.$1), lookup(indexes2.$1, p.$1), lookup(indexes2.$2, p.$1)};
                    
                    let iter_output = 
                        if(val1.$0 == val2.$0,
                            if(val1.$1 == val2.$1,
                                if(val1.$2 == val2.$2,
                                    {p.$0 + 1L, p.$1 + 1L, merge(p.$2, true), merge(p.$3, true)},
                                    if(val1.$2 < val2.$2,
                                        {p.$0 + 1L, p.$1, merge(p.$2, false), p.$3},
                                        {p.$0, p.$1 + 1L, p.$2, merge(p.$3, false)}
                                    )
                                ),
                                if(val1.$1 < val2.$1,
                                    {p.$0 + 1L, p.$1, merge(p.$2, false), p.$3},
                                    {p.$0, p.$1 + 1L, p.$2, merge(p.$3, false)}
                                )
                            ),
                            if(val1.$0 < val2.$0,
                                {p.$0 + 1L, p.$1, merge(p.$2, false), p.$3},
                                {p.$0, p.$1 + 1L, p.$2, merge(p.$3, false)}
                            )
                        );
                    {
                        iter_output,
                        iter_output.$0 < len1 && 
                        iter_output.$1 < len2
                    }
                ),
                {0L, 0L, appender[bool], appender[bool]}
    );
    # iterate over remaining un-checked elements in both arrays and append False until maxLen
    let res = if(res.$0 < maxlen, iterate(res,
            |p|
                {
                    {p.$0 + 1L, p.$1, merge(p.$2, false), p.$3},
                    p.$0 + 1L < maxlen
                }
    ), res);
    let res = if(res.$1 < maxlen, iterate(res,
            |p|
                {
                    {p.$0, p.$1 + 1L, p.$2, merge(p.$3, false)},
                    p.$1 + 1L < maxlen
                }
    ), res);
    let b = appender[vec[bool]];
    let c = merge(b, result(res.$2));
    result(merge(c, result(res.$3)))"""

    weld_obj.weld_code = weld_template % {'array1': weld_ids[0],
                                          'array2': weld_ids[1],
                                          'array3': weld_ids[2],
                                          'array4': weld_ids[3],
                                          'array5': weld_ids[4],
                                          'array6': weld_ids[5]}

    result = LazyResult(weld_obj, WeldBit(), 2)

    weld_objects = []
    weld_ids = []
    weld_col_ids = []

    if cache:
        id_ = LazyResult.generate_intermediate_id('mindex_merge')
        weld_input_name = WeldObject.generate_input_name(id_)
        LazyResult.register_intermediate_result(weld_input_name, result)

        for i in range(2):
            weld_obj = WeldObject(_encoder, _decoder)

            result_var = weld_obj.update(id_)
            assert result_var is not None

            weld_objects.append(weld_obj)
            weld_ids.append(result_var)
    else:
        for i in range(2):
            weld_obj = WeldObject(_encoder, _decoder)

            result_var = weld_obj.update(result.expr)
            assert result_var is None
            result_var = result.expr.obj_id
            weld_obj.dependencies[result_var] = result.expr

            weld_objects.append(weld_obj)
            weld_ids.append(result_var)

    # need 1 array from each resulting tables to get actual length
    for i in range(2):
        array_var = weld_objects[i].update(indexes[i * 3])
        if isinstance(indexes[i * 3], WeldObject):
            array_var = indexes[i * 3].obj_id
            weld_objects[i].dependencies[array_var] = indexes[i * 3]
        weld_col_ids.append(array_var)

    weld_templ = """slice(lookup(%(array)s, %(i)s), 0L, len(%(col)s))"""

    for i in range(2):
        weld_objects[i].weld_code = weld_templ % {'array': weld_ids[i],
                                                  'i': str(i) + 'L',
                                                  'col': weld_col_ids[i]}

    return weld_objects


def weld_groupby(by, by_types, columns, columns_types):
    """ Groups by the columns in by

    Parameters
    ----------
    by : list of np.ndarray or list of WeldObject
        the data to group by
    by_types : list of WeldType
        corresponding to by
    columns : list of np.ndarray or list of WeldObject
        the data to group
    columns_types : list of WeldType
        corresponding to columns

    Returns
    -------
    WeldObject
        representation of the computation

    """
    weld_obj = WeldObject(_encoder, _decoder)

    by_list_var = []
    for by_elem in by:
        by_var = weld_obj.update(by_elem)
        if isinstance(by_elem, WeldObject):
            by_var = by_elem.obj_id
            weld_obj.dependencies[by_var] = by_elem
        by_list_var.append(by_var)

    columns_list_var = []
    for column in columns:
        column_var = weld_obj.update(column)
        if isinstance(column, WeldObject):
            column_var = column.obj_id
            weld_obj.dependencies[column_var] = column
        columns_list_var.append(column_var)

    weld_template = """
    let columns = result(
            for(
                %(columns)s,
                appender,
                |b, i, e|
                    merge(b, e)
            )
    );
    tovec(
        result(
            for(
                zip(%(by)s, columns),
                groupmerger[%(by_types)s, %(columns_types)s],
                |b, i, e|
                    merge(b, {%(to_merge_keys)s, %(to_merge_values)s})
            )
        )
    )"""

    by = ', '.join(by_list_var) if len(by_list_var) > 1 else '%s' % by_list_var[0]
    columns = 'zip(%s)' % ', '.join(columns_list_var) if len(columns_list_var) > 1 else '%s' % columns_list_var[0]
    to_merge_keys = '{%s}' % ', '.join(['e.$%s' % str(k) for k in range(len(by_types))])
    to_merge_values = '{%s}' % ', '.join(['e.$%s.$%s' % (str(len(by_types)), str(k)) for k in range(len(columns_types))]) if len(columns_types) > 1 else '{e.$%s}' % str(len(by_types))
    by_types = '{%s}' % ', '.join([str(k) for k in by_types])
    columns_types = '{%s}' % ', '.join([str(k) for k in columns_types]) if len(columns_types) > 1 else '{%s}' % columns_types[0]

    weld_obj.weld_code = weld_template % {'by': by,
                                          'columns': columns,
                                          'by_types': by_types,
                                          'to_merge_keys': to_merge_keys,
                                          'to_merge_values': to_merge_values,
                                          'columns_types': columns_types}

    return weld_obj


def weld_groupby_aggregate(grouped_df, by_types, columns_types, operation):
    """ Groups by the columns in by

    Parameters
    ----------
    grouped_df : WeldObject
        DataFrame which has been grouped through weld_groupby
    by_types : list of WeldType
        corresponding to by
    columns_types : list of WeldType
        corresponding to columns
    operation : {'+', '*', 'min', 'max'}
        what operation to apply to grouped rows

    Returns
    -------
    WeldObject
        representation of the computation

    """
    weld_obj = WeldObject(_encoder, _decoder)

    grouped_df_var = weld_obj.update(grouped_df)

    assert grouped_df_var is None

    grouped_df_var = grouped_df.obj_id
    weld_obj.dependencies[grouped_df_var] = grouped_df

    weld_template = """
    tovec(
        result(
            for(
                %(grouped_df)s,
                dictmerger[%(by_types)s, %(columns_types)s, +],
                |b, i, e|
                    let group_res = for(e.$1,
                        %(mergers)s,
                        |c, j, f|
                            %(merger_ops)s
                    );
                    
                    merge(b, {e.$0, %(merger_res)s})
            )
        )
    )"""

    """ should be this but unsupported by Weld
    let merged = 
                        result(
                            for(e.$1,
                                merger[%(columns_types)s, %(operation)s],
                                |c, j, f|
                                    merge(c, f)
                            )
                    );
    """

    by_types = '{%s}' % ', '.join([str(k) for k in by_types])
    columns_typess = '{%s}' % ', '.join([str(k) for k in columns_types])
    mergers = '{%s}' % ', '.join(['merger[%s, %s]' % (str(k), operation) for k in columns_types])
    merger_ops = '{%s}' % ', '.join(['merge(c.$%s, f.$%s)' % (str(k), str(k)) for k in range(len(columns_types))])
    merger_res = '{%s}' % ', '.join(['result(group_res.$%s)' % str(k) for k in range(len(columns_types))])

    weld_obj.weld_code = weld_template % {'grouped_df': grouped_df_var,
                                          'operation': operation,
                                          'mergers': mergers,
                                          'merger_ops': merger_ops,
                                          'merger_res': merger_res,
                                          'by_types': by_types,
                                          'columns_types': columns_typess}

    return weld_obj


def weld_groupby_mean(grouped_df, by_types, columns_types):
    """ Groups by the columns in by

    Parameters
    ----------
    grouped_df : WeldObject
        DataFrame which has been grouped through weld_groupby
    by_types : list of WeldType
        corresponding to by
    columns_types : list of WeldType
        corresponding to columns

    Returns
    -------
    WeldObject
        representation of the computation

    """
    weld_obj = WeldObject(_encoder, _decoder)

    grouped_df_var = weld_obj.update(grouped_df)

    assert grouped_df_var is None

    grouped_df_var = grouped_df.obj_id
    weld_obj.dependencies[grouped_df_var] = grouped_df

    weld_template = """
    tovec(
        result(
            for(
                %(grouped_df)s,
                dictmerger[%(by_types)s, %(columns_types)s, +],
                |b, i, e|
                    let group_res = for(e.$1,
                        %(mergers)s,
                        |c, j, f|
                            %(merger_ops)s
                    );
                    
                    merge(b, {e.$0, %(merger_res)s})
            )
        )
    )"""

    """ should be this but unsupported by Weld
    let merged = 
                        result(
                            for(e.$1,
                                merger[%(columns_types)s, %(operation)s],
                                |c, j, f|
                                    merge(c, f)
                            )
                    );
    """

    by_types = '{%s}' % ', '.join([str(k) for k in by_types])
    columns_typess = '{%s}' % ', '.join(['f64' for k in columns_types])
    mergers = '{%s}' % ', '.join(['merger[%s, +]' % str(k) for k in columns_types])
    merger_ops = '{%s}' % ', '.join(['merge(c.$%s, f.$%s)' % (str(k), str(k)) for k in range(len(columns_types))])
    merger_res = '{%s}' % ', '.join(['f64(result(group_res.$%s)) / f64(len(e.$1))' % str(k) for k in range(len(columns_types))])

    weld_obj.weld_code = weld_template % {'grouped_df': grouped_df_var,
                                          'mergers': mergers,
                                          'merger_ops': merger_ops,
                                          'merger_res': merger_res,
                                          'by_types': by_types,
                                          'columns_types': columns_typess}

    return weld_obj


def weld_get_column(grouped_df, index, is_index=False):
    """ Gets the (index) column from the grouped DataFrame

    Parameters
    ----------
    grouped_df : WeldObject
        DataFrame which has been grouped through weld_groupby
    index : int
        index of the column; the mapping name-to-index is maintained by DataFrameGroupBy
    is_index : bool
        to signal if the requested column is in the index

    Returns
    -------
    WeldObject
        representation of the computation

    """
    weld_obj = WeldObject(_encoder, _decoder)

    grouped_df_var = weld_obj.update(grouped_df)

    assert grouped_df_var is None

    grouped_df_var = grouped_df.obj_id
    weld_obj.dependencies[grouped_df_var] = grouped_df

    weld_template = """
    map(
        %(grouped_df)s,
        |e|
            e.$%(index)s
    )"""

    weld_obj.weld_code = weld_template % {'grouped_df': grouped_df_var,
                                          'index': '0.$%s' % index if is_index else '1.$%s' % index}

    return weld_obj


# TODO: bugged! should be ordered dict
def weld_unique(array, type):
    """ Extract the unique elements in the array

    Parameters
    ----------
    array : np.ndarray or WeldObject
        input array
    type : WeldType
        of the input array

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
    sort(
        map(
            tovec(
                result(
                    for(
                        map(
                            %(array)s,
                            |e| 
                                {e, 0}
                        ),
                        dictmerger[%(type)s, i32, +],
                        |b, i, e| 
                            merge(b, e)
                    )
                )
            ),
            |e| 
                e.$0
        ),
        |x: %(type)s| 
            x
    )"""

    weld_obj.weld_code = weld_template % {'array': array_var,
                                          'type': type}

    return weld_obj


def weld_udf(weld_template, mapping):
    """ Apply weld_code given arrays and scalars as input

    Parameters
    ----------
    weld_template : str
        the code that will be recorded for execution
    mapping : dict
        maps placeholders to either arrays (np.array or WeldObject) or scalars

    Returns
    -------
        the result of the inputted weld_code computation

    Examples
    -------
    >>> array = np.array([1, 3, 4])
    >>> weld_template = "map(%(array)s, |e| e + %(scalar)s)"
    >>> mapping = {'array': array, 'scalar': '2L'}
    >>> result = weld_udf(weld_template, mapping)
    >>> LazyResult(result, WeldLong(), 1).evaluate()
    [3 5 6]

    """
    weld_obj = WeldObject(_encoder, _decoder)

    # update the mapping with the weld var's (array_var in other methods)
    for k, v in mapping.items():
        # does not need to be registered if not np.array or weldobject
        if not isinstance(v, (np.ndarray, WeldObject)):
            continue

        array_var = weld_obj.update(v)

        if isinstance(v, WeldObject):
            array_var = v.obj_id
            weld_obj.dependencies[array_var] = v

        mapping.update({k: array_var})

    weld_obj.weld_code = weld_template % mapping

    return weld_obj


# TODO: be able to sort a dataframe since the groupby output is expected to be sorted
def weld_sort():
    pass


def weld_describe(array, weld_type, aggregations):
    """ Aggregate during the same evaluation as opposed to separately as in Series.agg

    Parameters
    ----------
    array :  np.ndarray or WeldObject
        to aggregate on
    weld_type : WeldType
        of the array
    aggregations : list of str
        supported are = {'min', 'max', 'sum', 'prod', 'mean', 'std'}

    Returns
    -------
    WeldObject

    """
    assert isinstance(aggregations, list)
    assert len(aggregations) > 0

    weld_obj = WeldObject(_encoder, _decoder)

    array_var = weld_obj.update(array)
    if isinstance(array, WeldObject):
        array_var = array.obj_id
        weld_obj.dependencies[array_var] = array

    merger_chunk = """
    let agg_%(name)s = f64(
        result(
            for(
                %(array)s,
                merger[%(type)s, %(operation)s],
                |b, i, e| 
                    merge(b, e)
            )
        )
    );"""

    mean_chunk_solo = """
    let agg_mean = f64(
        result(
            for(
                %(array)s,
                merger[%(type)s, +],
                |b, i, n|
                    merge(b, n)
            )
        )
    ) / f64(len(%(array)s));"""

    mean_chunk_with_sum = """
    let agg_mean = agg_sum / f64(len(%(array)s));
    """

    std_chunk_solo = """
    %(mean)s
    let agg_std = sqrt(
        result(
            for(
                %(array)s,
                merger[f64, +],
                |b, i, n|
                    merge(b, pow(f64(n) - agg_mean, 2.0))
            )
        ) / f64(len(%(array)s) - 1L)
    );""".replace('%(mean)s', mean_chunk_with_sum if 'sum' in aggregations else mean_chunk_solo)

    std_chunk_with_mean = """
    let agg_std = sqrt(
        result(
            for(
                %(array)s,
                merger[f64, +],
                |b, i, n|
                    merge(b, pow(f64(n) - agg_mean, 2.0))
            )
        ) / f64(len(%(array)s) - 1L)
    );"""

    aggregations_dict = {'min': merger_chunk.replace('%(operation)s', 'min').replace('%(name)s', 'min'),
                         'max': merger_chunk.replace('%(operation)s', 'max').replace('%(name)s', 'max'),
                         'sum': merger_chunk.replace('%(operation)s', '+').replace('%(name)s', 'sum'),
                         'prod': merger_chunk.replace('%(operation)s', '*').replace('%(name)s', 'prod'),
                         'mean': mean_chunk_with_sum if 'sum' in aggregations else mean_chunk_solo,
                         'std': std_chunk_with_mean if 'mean' in aggregations else std_chunk_solo}

    weld_template = """
    %(chunks)s
    let agg_result = appender[f64];
    %(merges)s
    result(agg_result)
    """

    chunks = ''.join([aggregations_dict[agg] for agg in aggregations])
    merges = ''.join(['let agg_result = merge(agg_result, %s);\n\t' % ('agg_' + agg) for agg in aggregations])

    weld_obj.weld_code = weld_template % {'chunks': chunks, 'merges': merges} \
                                       % {'array': array_var, 'type': weld_type}

    return weld_obj
