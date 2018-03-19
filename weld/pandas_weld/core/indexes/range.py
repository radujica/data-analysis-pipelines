import numpy as np
from grizzly.encoders import numpy_to_weld_type
from weld.types import WeldLong
from lazy_data import LazyData
from base import Index
from pandas_weld.core.utils import replace_slice_defaults, subset
from pandas_weld.weld import weld_range, weld_filter


class RangeIndex(LazyData):
    """ Weld-ed 1-d Index over a range

    Parameters
    ----------
    start : int
    stop : int
    step : int

    Returns
    -------
    RangeIndex

    """
    def __init__(self, start=0, stop=0, step=1):
        super(RangeIndex, self).__init__(weld_range(start, stop, step), WeldLong(), 1)

        self.start = start
        self.stop = stop
        self.step = step

    @property
    def data(self):
        return self.expr

    def __repr__(self):
        return "RangeIndex with start=%d, stop=%d, step=%d" % (self.start, self.stop, self.step)

    def __getitem__(self, item):
        """ Retrieve a portion of the RangeIndex

        Parameters
        ----------
        item : slice or LazyData
            if slice, returns a sliced Index;
            if LazyData, returns a filtered Index only with the labels corresponding to
            True in the Series

        Returns
        -------
        Index
            it will no longer be a RangeIndex

        """
        if isinstance(item, slice):
            item = replace_slice_defaults(item)

            # update func_args so that less data is read from file
            if isinstance(self, LazyData) and self.data_id is not None:
                index = self.input_mapping.data_ids.index(self.data_id)
                old_args = self.input_mapping.input_function_args[index]
                slice_as_tuple = (slice(item.start, item.stop, item.step),)
                new_args = old_args + (slice_as_tuple,)
                self.input_mapping.update_input_function_args(index, new_args)

            return Index(subset(self, item).expr,
                         np.dtype(np.int64))
        elif isinstance(item, LazyData):
            if item.weld_type != numpy_to_weld_type('bool'):
                raise ValueError('expected series of bool to filter Index elements')

            return Index(weld_filter(self.expr,
                                     item.expr,
                                     numpy_to_weld_type(np.dtype(np.int64))),
                         np.dtype(np.int64))
        else:
            raise TypeError('expected slice or Series of bool in Index.__getitem__')

    def evaluate(self, verbose=True, decode=True, passes=None, num_threads=1,
                 apply_experimental_transforms=False):
        return super(LazyData, self).evaluate(verbose, decode, passes,
                                              num_threads, apply_experimental_transforms)
