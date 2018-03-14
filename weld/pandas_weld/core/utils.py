from grizzly.encoders import numpy_to_weld_type_mapping
from lazy_data import LazyData
from pandas_weld.weld import weld_subset
import numpy as np


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
        weld_type = numpy_to_weld_type_mapping[str(array.dtype)]
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
