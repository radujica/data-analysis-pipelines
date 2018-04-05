from grizzly.encoders import numpy_to_weld_type
from weld.types import *
from lazy_data import LazyData
from pandas_weld.weld import weld_subset
import numpy as np


# assuming the keys are cached; could just replace with i32, etc
_weld_to_numpy_str_type_mapping = {
    str(WeldBit()): '?',
    str(WeldInt16()): 'h',
    str(WeldInt()): 'i',
    str(WeldLong()): 'l',
    str(WeldFloat()): 'f',
    str(WeldDouble()): 'd',
    str(WeldChar()): 'S'
}


def weld_to_numpy_type(weld_type):
    return np.dtype(_weld_to_numpy_str_type_mapping[str(weld_type)])


def subset(array, slice_):
    """ Return a subset of the input array

    Parameters
    ----------
    array : np.array / LazyData
        1-dimensional array
    slice_ : slice
        subset to return

    Returns
    -------
    LazyData

    """
    if not isinstance(slice_, slice):
        raise TypeError('expected a slice in subset')

    if isinstance(array, LazyData):
        weld_type = array.weld_type
        array = array.expr
    elif isinstance(array, np.ndarray):
        weld_type = numpy_to_weld_type(array.dtype)
    else:
        raise TypeError('expected array as LazyData or np.ndarray')

    return LazyData(weld_subset(array,
                                slice_,
                                weld_type),
                    weld_type,
                    1)


# to replace the None with default values for use in weld code where None is not acceptable
# TODO: make slicing more flexible
def replace_slice_defaults(slice_, default_start=0, default_step=1):
    if not isinstance(slice_, slice):
        raise TypeError('expected a slice in replace_slice_none')

    start = slice_.start
    stop = slice_.stop
    step = slice_.step

    if slice_.start is None:
        start = default_start

    if slice_.stop is None:
        raise ValueError('there must be a slice.stop')

    if slice_.step is None:
        step = default_step

    return slice(start, stop, step)


def get_expression_or_raw(data):
    if isinstance(data, LazyData):
        return data.expr
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise TypeError('expected LazyData or np.ndarray')


def get_weld_type(data):
    if isinstance(data, LazyData):
        return data.weld_type
    if isinstance(data, np.ndarray):
        return numpy_to_weld_type(data.dtype)
    else:
        raise TypeError('expected LazyData or np.ndarray')


def evaluate_or_raw(data, verbose, decode, passes,
                    num_threads, apply_experimental_transforms):
    if isinstance(data, LazyData):
        return data.evaluate(verbose, decode, passes,
                             num_threads, apply_experimental_transforms)
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise TypeError('expected LazyData or np.ndarray')
