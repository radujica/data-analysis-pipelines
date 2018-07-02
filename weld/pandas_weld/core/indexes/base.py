from collections import OrderedDict

import numpy as np
from grizzly.encoders import numpy_to_weld_type
from weld.weldobject import WeldObject

from lazy_result import LazyResult
from pandas_weld.core.utils import replace_slice_defaults
from pandas_weld.weld import weld_filter


# TODO: this should be a superclass of the others to conform to pandas
# what is intended here is a 1d "MultiIndex" which could come from file (though currently is like a Series)
class Index(LazyResult):
    """ 1-d Weld-ed Index

    Parameters
    ----------
    data : np.ndarray or WeldObject
        this allows arbitrary indexes to be created
    dtype : np.dtype
        type of the data

    Returns
    -------
    Index

    """
    def __init__(self, data, dtype, name=None):
        if not isinstance(data, (np.ndarray, WeldObject)):
            raise TypeError('expected np.ndarray or WeldObject in Series.__init__')

        super(Index, self).__init__(data, numpy_to_weld_type(dtype), 1)

        self.dtype = dtype
        self.name = 'Index' if name is None else name

    @property
    def data(self):
        return self.expr

    def __repr__(self):
        return "{}(name={}, dtype={}, data={})".format(self.__class__.__name__,
                                                       self.name,
                                                       self.dtype,
                                                       repr(self.data))

    def __str__(self):
        return str(self.data)

    def evaluate(self, verbose=True, decode=True, passes=None, num_threads=1,
                 apply_experimental_transforms=False, as_dict=False):
        data = super(Index, self).evaluate(verbose, decode, passes, num_threads, apply_experimental_transforms)

        if as_dict:
            return OrderedDict({self.__class__.__name__: data})
        else:
            return Index(data, self.dtype, self.name)

    def __getitem__(self, item):
        """ Retrieve a portion of the Index

        Parameters
        ----------
        item : slice or LazyResult
            if slice, returns a sliced Index;
            if LazyResult, returns a filtered Index only with the labels corresponding to
            True in the Series

        Returns
        -------
        Index

        """
        if isinstance(item, slice):
            item = replace_slice_defaults(item)

            self.update_rows(item)

            return Index(self.expr,
                         self.dtype)
        elif isinstance(item, LazyResult):
            if str(item.weld_type) != str(numpy_to_weld_type('bool')):
                raise ValueError('expected LazyResult of bool to filter Index elements')

            return Index(weld_filter(self.expr,
                                     item.expr),
                         self.dtype)
        else:
            raise TypeError('expected slice or LazyResult of bool in Index.__getitem__')
