# TODO: handle different start and step, not only stop
def convert_row_to_nd_slices(slice_, shape):
    """ Convert from 1d slice to nd slices

    When first n rows are required, need to convert to
    closest suitable nd-slices that contain them.

    Parameters
    ----------
    slice_ : slice
        desired on the data
    shape : tuple
        the shape of the data

    Returns
    -------
    tuple
        of slices corresponding to the shape that are
        required to read the 1d slice of rows

    """
    if not isinstance(slice_, slice):
        raise ValueError('expected a tuple with a single slice')

    number_rows = slice_.stop
    slices_list = []
    cumulative_dimension = 1

    for dimension in reversed(shape):
        div = number_rows / cumulative_dimension
        if number_rows % cumulative_dimension == 0:
            new_stop = div
        else:
            new_stop = div + 1

        if new_stop > dimension:
            new_stop = dimension

        slices_list.append(slice(new_stop))
        cumulative_dimension *= dimension

    return tuple(reversed(slices_list))


def replace_slice_defaults(slice_):
    if slice_.start is None:
        slice_.start = 0
    
    if slice_.step is None:
        slice_.step = 1
        
    return slice_
