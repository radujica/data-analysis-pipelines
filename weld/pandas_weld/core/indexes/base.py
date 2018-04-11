import numpy as np
from grizzly.encoders import numpy_to_weld_type
from weld.weldobject import WeldObject
from lazy_data import LazyData
from pandas_weld.core.utils import replace_slice_defaults, subset
from pandas_weld.weld import weld_filter


# TODO: this should be a superclass of the others to conform to pandas
# what is intended here is a 1d "MultiIndex" which could come from file (though currently is like a Series)
class Index(LazyData):
    """ 1-d Weld-ed Index

    Parameters
    ----------
    data : np.array / WeldObject
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
        self.name = name

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

    def __getitem__(self, item):
        """ Retrieve a portion of the Index

        Parameters
        ----------
        item : slice or LazyData
            if slice, returns a sliced Index;
            if LazyData, returns a filtered Index only with the labels corresponding to
            True in the Series

        Returns
        -------
        Index

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
                         self.dtype)
        elif isinstance(item, LazyData):
            if str(item.weld_type) != str(numpy_to_weld_type('bool')):
                raise ValueError('expected LazyData of bool to filter Index elements')

            return Index(weld_filter(self.expr,
                                     item.expr),
                         self.dtype)
        else:
            raise TypeError('expected slice or LazyData of bool in Index.__getitem__')
