from grizzly.encoders import NumPyEncoder, NumPyDecoder, numpy_to_weld_type_mapping
from weld.weldobject import WeldObject

_encoder = NumPyEncoder()
_decoder = NumPyDecoder()


# TODO: the helper methods should probably reside in the numpy package
def duplicate_elements(array, n, dtype, multiple=False):
    """ Expands array by multiplying each element n times

    Parameters
    ----------
    array : np.array / WeldObject
        the source data
    n : long
        how many times to repeat each element
    dtype : np.dtype
        type of the elements in the array
    multiple : boolean
        True if used internally by cartesian_product to signify the operation
        has been done once already and hence must behave slightly different

    Returns
    -------
    WeldObject

    Examples
    --------
    >>> duplicate_elements(np.array([1, 2, 3], dtype=np.int32), 2, np.dtype(np.int32))
    [1, 1, 2, 2, 3, 3]

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
                                          'n': n,
                                          'index_or_value': 'n' if multiple else 'i',
                                          'type': 'i64' if multiple else numpy_to_weld_type_mapping[str(dtype)]}

    return weld_obj


def duplicate_array(array, n, dtype, multiple=False):
    """ Duplicate array n times

    Parameters
    ----------
    array : np.array / WeldObject
        the source data
    n : int
        how many times to repeat the source array
    dtype : np.dtype
        type of the elements in the array
    multiple : boolean
        True if used internally by cartesian_product to signify the operation
        has been done once already and hence must behave slightly different

    Returns
    -------
    WeldObject

    Examples
    --------
    >>> duplicate_array(np.array([1, 2, 3], dtype=np.int32), 2, np.dtype(np.int32))
    [1, 2, 3, 1, 2, 3]

    """
    weld_obj = WeldObject(_encoder, _decoder)

    array_var = weld_obj.update(array)

    if isinstance(array, WeldObject):
        array_var = array.obj_id
        weld_obj.dependencies[array_var] = array

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
                                          'n': n,
                                          'index_or_value': 'n' if multiple else 'i',
                                          'type': 'i64' if multiple else numpy_to_weld_type_mapping[str(dtype)]}

    return weld_obj


# just helper class to name pair elements in cartesian_product_indices
class _WeldObjectIdPair(object):
    def __init__(self, weld_object, object_id):
        self.weld_object = weld_object
        self.object_id = object_id

    def __repr__(self):
        return self.weld_object + ', ' + self.object_id


# TODO: could generalize to return either values or indices
# TODO: properly test starting with WeldObjects too
def cartesian_product_indices(arrays, arrays_types):
    """ Performs cartesian product between all arrays

    Returns the indices instead of the actual values

    Parameters
    ----------
    arrays : list of np.array or WeldObject
        list containing arrays that need to be in the product
    arrays_types : list of the np.dtype's of the arrays
        type of the elements in the arrays

    Returns
    -------
    WeldObject

    Examples
    --------
    >>> cartesian_product_indices([np.array([1, 2], dtype=np.int32),
                                   np.array([3, 4], dtype=np.int32)], np.dtype(np.int32))
    [[0, 0, 1, 1], [0, 1, 0, 1]]

    See also
    --------
    pd.MultiIndex

    """
    number_of_arrays = len(arrays)

    if number_of_arrays < 2:
        raise ValueError('expected at least 2 arrays')

    # need to first register the objects since they rely on each other on the length
    # so build a list of [weld_obj, id] where the id will be used in len(id)
    # the weld_objects 'are' now the arrays
    weld_objects = []
    for i in xrange(number_of_arrays):
        weld_obj = WeldObject(_encoder, _decoder)
        array_var = weld_obj.update(arrays[i])

        if isinstance(arrays[i], WeldObject):
            array_var = arrays[i].obj_id
            weld_obj.dependencies[array_var] = arrays[i]

        weld_obj.weld_code = array_var

        weld_objects.append(_WeldObjectIdPair(weld_obj, array_var))

    # update the context of each weld object with the others
    for i in xrange(number_of_arrays):
        for j in xrange(0, i):
            weld_objects[i].weld_object.context[weld_objects[j].object_id] = arrays[j]
        for j in xrange(i + 1, number_of_arrays):
            weld_objects[i].weld_object.context[weld_objects[j].object_id] = arrays[j]

    # first 2 arrays are cartesian-produced by default
    weld_objects[0].weld_object = \
        duplicate_elements(weld_objects[0].weld_object,
                           'len(%s)' % weld_objects[1].object_id,
                           arrays_types[0])
    weld_objects[1].weld_object = \
        duplicate_array(weld_objects[1].weld_object,
                        'len(%s)' % weld_objects[0].object_id,
                        arrays_types[1])

    # handle the remaining arrays, i.e. for index > 2
    for i in xrange(2, number_of_arrays):
        for j in xrange(0, i):
            weld_objects[j].weld_object = \
                duplicate_elements(weld_objects[j].weld_object,
                                   'len(%s)' % weld_objects[i].object_id,
                                   arrays_types[j],
                                   multiple=True)

        weld_objects[i].weld_object = \
            duplicate_array(weld_objects[i].weld_object,
                            'len(%s)' % weld_objects[0].object_id,
                            arrays_types[i])

        for j in xrange(1, i):
            weld_objects[i].weld_object = \
                duplicate_array(weld_objects[i].weld_object,
                                'len(%s)' % weld_objects[j].object_id,
                                arrays_types[i],
                                multiple=True)

    return [t.weld_object for t in weld_objects]


# maybe extend LazyOpResult?
# TODO: docs on this
class MultiIndex(object):
    def __init__(self, levels, labels, names):
        self.levels = levels
        self.labels = labels
        self.names = names

    @classmethod
    def from_product(cls, levels, levels_types, names):
        labels = cartesian_product_indices(levels, levels_types)

        return cls(levels, labels, names)
