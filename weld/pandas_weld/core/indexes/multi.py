from grizzly.encoders import NumPyEncoder, NumPyDecoder, numpy_to_weld_type_mapping
from weld.types import WeldLong
from weld.weldobject import WeldObject
from netCDF4_weld.lazy_data import LazyData
import numpy as np

# the methods are only intended to work with numpy, so have a single encoder/decoder
_encoder = NumPyEncoder()
_decoder = NumPyDecoder()


# TODO: the helper methods should probably reside in the numpy package
# TODO: could generalize to return either values or indices
# TODO: fix bug when inputs are objects

def _duplicate_elements_indices(array, n, weld_type, cartesian=False):
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
                                          'n': '%sL' % n if type(n) is long else n,
                                          'index_or_value': 'n' if cartesian else 'i',
                                          'type': 'i64' if cartesian else weld_type}

    return weld_obj


def duplicate_elements_indices(array, n, cartesian=False):
    """ Expands array by multiplying each element n times

    Parameters
    ----------
    array : np.array / WeldObject
        the source data
    n : long
        how many times to repeat each element
    cartesian : boolean
        True if used internally by cartesian_product to signify the operation
        has been done once already and hence must behave slightly different

    Returns
    -------
    LazyData
        the expanded array containing the indices, not the elements

    Examples
    --------
    >>> duplicate_elements_indices(np.array([1, 2, 3]), 2)
    [0, 0, 1, 1, 2, 2]

    """
    data_id = None
    read_func = None
    if isinstance(array, LazyData):
        weld_type = array.weld_type
        array = array.expr
        data_id = array.data_id
        read_func = array.read_func
    elif isinstance(array, np.ndarray):
        weld_type = numpy_to_weld_type_mapping[str(array.dtype)]
    else:
        raise NotImplementedError

    return LazyData(_duplicate_elements_indices(array, n, weld_type, cartesian),
                    WeldLong(),
                    1,
                    data_id,
                    read_func)


def _duplicate_array_indices(array, n, weld_type, cartesian=False):
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
                                          'n': '%sL' % n if type(n) is long else n,
                                          'index_or_value': 'n' if cartesian else 'i',
                                          'type': 'i64' if cartesian else weld_type}

    return weld_obj


def duplicate_array_indices(array, n, cartesian=False):
    """ Duplicate array n times

    Parameters
    ----------
    array : np.array / WeldObject
        the source data
    n : long
        how many times to repeat the source array
    cartesian : boolean
        True if used internally by cartesian_product to signify the operation
        has been done once already and hence must behave slightly different

    Returns
    -------
    LazyData
        the expanded array containing the indices, not the elements

    Examples
    --------
    >>> duplicate_array_indices(np.array([1, 2, 3]), 2)
    [0, 1, 2, 0, 1, 2]

    """
    data_id = None
    read_func = None
    if isinstance(array, LazyData):
        weld_type = array.weld_type
        array = array.expr
        data_id = array.data_id
        read_func = array.read_func
    elif isinstance(array, np.ndarray):
        weld_type = numpy_to_weld_type_mapping[str(array.dtype)]
    else:
        raise NotImplementedError

    return LazyData(_duplicate_array_indices(array, n, weld_type, cartesian),
                    WeldLong(),
                    1,
                    data_id,
                    read_func)


# helper class to name pair elements in cartesian_product_indices
class _WeldObjectIdPair(object):
    def __init__(self, weld_object, object_id):
        self.weld_object = weld_object
        self.object_id = object_id

    def __repr__(self):
        return self.weld_object + ', ' + self.object_id


def _cartesian_product_indices(arrays, arrays_types, number_of_arrays):
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
        _duplicate_elements_indices(weld_objects[0].weld_object,
                                    'len(%s)' % weld_objects[1].object_id,
                                    arrays_types[0])
    weld_objects[1].weld_object = \
        _duplicate_array_indices(weld_objects[1].weld_object,
                                 'len(%s)' % weld_objects[0].object_id,
                                 arrays_types[1])

    # handle the remaining arrays, i.e. for index > 2
    for i in xrange(2, number_of_arrays):
        for j in xrange(0, i):
            weld_objects[j].weld_object = \
                _duplicate_elements_indices(weld_objects[j].weld_object,
                                            'len(%s)' % weld_objects[i].object_id,
                                            arrays_types[j],
                                            cartesian=True)

        weld_objects[i].weld_object = \
            _duplicate_array_indices(weld_objects[i].weld_object,
                                     'len(%s)' % weld_objects[0].object_id,
                                     arrays_types[i])

        for j in xrange(1, i):
            weld_objects[i].weld_object = \
                _duplicate_array_indices(weld_objects[i].weld_object,
                                         'len(%s)' % weld_objects[j].object_id,
                                         arrays_types[i],
                                         cartesian=True)

    return [k.weld_object for k in weld_objects]


# TODO: properly test starting with WeldObjects too
def cartesian_product_indices(arrays):
    """ Performs cartesian product between all arrays

    Returns the indices instead of the actual values

    Parameters
    ----------
    arrays : list of np.array or WeldObject
        list containing arrays that need to be in the product

    Returns
    -------
    [LazyData]

    Examples
    --------
    >>> cartesian_product_indices([np.array([1, 2]), np.array([3, 4])])
    [[0, 0, 1, 1], [0, 1, 0, 1]]

    See also
    --------
    pd.MultiIndex

    """
    number_of_arrays = len(arrays)

    if number_of_arrays < 2:
        raise ValueError('expected at least 2 arrays')

    arrays_copied = arrays[:]
    weld_types = []

    for i in xrange(number_of_arrays):
        if isinstance(arrays_copied[i], LazyData):
            weld_type = arrays_copied[i].weld_type
            arrays_copied[i] = arrays_copied[i].expr
        elif isinstance(arrays_copied[i], np.ndarray):
            weld_type = numpy_to_weld_type_mapping[str(arrays_copied[i].dtype)]
        else:
            raise NotImplementedError

        weld_types.append(weld_type)

    weld_objects = _cartesian_product_indices(arrays_copied, weld_types, number_of_arrays)

    return [LazyData(weld_objects[k], WeldLong(), 1) for k in xrange(number_of_arrays)]


class MultiIndex(object):
    """ Create Weld-ed MultiIndex

    It is assumed that the parameter lists are in the same order,
    i.e. same indexes match

    Parameters
    ----------
    levels : list
        np.arrays or LazyData
    labels : list
        np.arrays or LazyData
    names : list
        str

    Returns
    -------
    MultiIndex

    """
    def __init__(self, levels, labels, names):
        self.levels = levels
        self.labels = labels
        self.names = names

    @classmethod
    def from_product(cls, levels, names):
        """ Create MultiIndex when no labels are available

        Parameters
        ----------
        levels : list
            np.arrays or WeldObjects
        names : list
            names of the levels

        Returns
        -------
        MultiIndex

        """
        labels = cartesian_product_indices(levels)

        return cls(levels, labels, names)

    # only names are (already) materialized
    def __repr__(self):
        return "index names:\n\t%s" % str(self.names)

    def evaluate_all(self, verbose=True, decode=True, passes=None, num_threads=1,
                     apply_experimental_transforms=False):
        materialized_levels = {}
        materialized_labels = {}

        for i in xrange(len(self.names)):
            materialized_levels[self.names[i]] = self.levels[i].evaluate(verbose, decode, passes,
                                                                         num_threads, apply_experimental_transforms)
            materialized_labels[self.names[i]] = self.labels[i].evaluate(verbose, decode, passes,
                                                                         num_threads, apply_experimental_transforms)

        string_representation = """%(repr)s\nlevels:\n\t%(levels)s\nlabels:\n\t%(labels)s"""

        return string_representation % {'repr': self.__repr__(),
                                        'levels': materialized_levels,
                                        'labels': materialized_labels}
