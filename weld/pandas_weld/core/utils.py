from grizzly.encoders import NumPyEncoder, NumPyDecoder, numpy_to_weld_type_mapping
from weld.weldobject import WeldObject
from lazy_data import LazyData
import numpy as np

_encoder = NumPyEncoder()
_decoder = NumPyDecoder()


def _subset(array, slice_, weld_type):
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
        raise ValueError('expected a slice in subset')

    if isinstance(array, LazyData):
        weld_type = array.weld_type
        array = array.expr
    elif isinstance(array, np.ndarray):
        weld_type = numpy_to_weld_type_mapping[str(array.dtype)]
    else:
        raise NotImplementedError

    return LazyData(_subset(array, slice_, weld_type),
                    weld_type,
                    1)


# to replace the None with default values for use in weld code where None is not acceptable
# TODO: make slicing more flexible
def replace_slice_defaults(slice_, default_start=0, default_step=1):
    if not isinstance(slice_, slice):
        raise ValueError('expected a slice in replace_slice_none')

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
