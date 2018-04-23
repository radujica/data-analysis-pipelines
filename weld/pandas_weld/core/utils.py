from grizzly.encoders import numpy_to_weld_type
from weld.types import *
from lazy_result import LazyResult
import numpy as np


# assuming the keys are cached; could just replace with i32, etc
_weld_to_numpy_str_type_mapping = {
    str(WeldBit()): '?',
    str(WeldInt16()): 'h',
    str(WeldInt()): 'i',
    str(WeldLong()): 'l',
    str(WeldFloat()): 'f',
    str(WeldDouble()): 'd',
    str(WeldChar()): 'b',
    str(WeldVec(WeldChar())): 'S'
}


def weld_to_numpy_type(weld_type):
    return np.dtype(_weld_to_numpy_str_type_mapping[str(weld_type)])


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
    if isinstance(data, LazyResult):
        return data.expr
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise TypeError('expected LazyResult or np.ndarray')


def get_weld_type(data):
    if isinstance(data, LazyResult):
        return data.weld_type
    elif isinstance(data, np.ndarray):
        return numpy_to_weld_type(data.dtype)
    else:
        raise TypeError('expected LazyResult or np.ndarray')


def get_dtype(data):
    if isinstance(data, LazyResult):
        return weld_to_numpy_type(data.weld_type)
    elif isinstance(data, np.ndarray):
        return data.dtype
    else:
        raise TypeError('expected LazyResult or np.ndarray')


def get_weld_info(data, expression=False, weld_type=False, dtype=False):
    result = []
    if expression:
        result.append(get_expression_or_raw(data))
    if weld_type:
        result.append(get_weld_type(data))
    if dtype:
        result.append(get_dtype(data))

    return tuple(result)


def evaluate_or_raw(data, verbose, decode, passes,
                    num_threads, apply_experimental_transforms):
    if isinstance(data, LazyResult):
        return data.evaluate(verbose, decode, passes,
                             num_threads, apply_experimental_transforms)
    elif isinstance(data, np.ndarray):
        return data
    else:
        raise TypeError('expected LazyResult or np.ndarray')
