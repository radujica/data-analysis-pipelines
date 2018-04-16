import numpy as np
from grizzly.encoders import numpy_to_weld_type
from weld.types import WeldLong
from lazy_result import LazyResult
from base import Index
from pandas_weld.core.utils import replace_slice_defaults, subset
from pandas_weld.weld import weld_range, weld_filter


class RangeIndex(LazyResult):
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
        return "{}(start={}, stop={}, step={})".format(self.__class__.__name__,
                                                       self.start,
                                                       self.stop,
                                                       self.step)

    def __str__(self):
        return str(self.data)

    def __getitem__(self, item):
        """ Retrieve a portion of the RangeIndex

        Parameters
        ----------
        item : slice or LazyResult
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

            self.update_rows(item)

            return Index(subset(self, item).expr,
                         np.dtype(np.int64))
        elif isinstance(item, LazyResult):
            if str(item.weld_type) != str(numpy_to_weld_type('bool')):
                raise ValueError('expected series of bool to filter Index elements')

            return Index(weld_filter(self.expr,
                                     item.expr),
                         np.dtype(np.int64))
        else:
            raise TypeError('expected slice or Series of bool in Index.__getitem__')
