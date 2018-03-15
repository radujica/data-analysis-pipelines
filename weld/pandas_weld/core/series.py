from weld.types import WeldBit
from weld.weldobject import WeldObject
from lazy_data import LazyData
from pandas_weld.weld import weld_aggregate, weld_compare
from utils import subset, replace_slice_defaults
import numpy as np


class Series(LazyData):
    """ Weld-ed pandas Series

    Parameters
    ----------
    data : np.ndarray / WeldObject
        raw data or weld expression
    weld_type : WeldType
        of the elements
    data_id : str
        generated only by parsers to record the existence of new data from file; needs to be passed on
        to other LazyData children objects, e.g. when creating a pandas_weld.Series from netCDF4_weld.Variable

    See also
    --------
    pandas.Series

    """

    # TODO: should accept and store dtype instead; also see TODO @LazyData
    def __init__(self, data, weld_type, data_id=None):
        if not isinstance(data, (np.ndarray, WeldObject)):
            raise TypeError('expected np.ndarray or WeldObject in Series.__init__')
        super(Series, self).__init__(data, weld_type, 1, data_id)

    def __getitem__(self, item):
        """ Lazy operation to select a subset of the series

        Has consequences! Any previous and/or following operations on
        the data within will be done only on this subset of the data

        Parameters
        ----------
        item : slice
            a slice of the data for the number of desired rows; currently
            must contain a stop value and will not work as expected for
            start != 0 and stride != 1

        Returns
        -------
        Series

        """
        if not isinstance(item, slice):
            raise TypeError('expected a slice in Series.__getitem__')

        # TODO: add functionality for series, e.g. series[series < 10]

        item = replace_slice_defaults(item)

        # update func_args so that less data is read from file
        if isinstance(self, LazyData) and self.data_id is not None:
            index = self.input_mapping.data_ids.index(self.data_id)
            old_args = self.input_mapping.input_function_args[index]
            slice_as_tuple = (slice(item.start, item.stop, item.step),)
            new_args = old_args + (slice_as_tuple,)
            self.input_mapping.update_input_function_args(index, new_args)

        return Series(subset(self, item).expr,
                      self.weld_type,
                      self.data_id)

    def head(self, n=10):
        """ Eager operation to read first n rows

        This operation has no consequences, unlike getitem

        Parameters
        ----------
        n : int
            how many rows

        Returns
        -------
        np.array
            or other raw data

        """
        slice_ = replace_slice_defaults(slice(n))
        data = self.expr

        if self.data_id is not None:
            index = self.input_mapping.data_ids.index(self.data_id)
            old_args = self.input_mapping.input_function_args[index]
            slice_as_tuple = (slice_,)
            new_args = old_args + (slice_as_tuple,)
            data = self.input_mapping.input_functions[index](*new_args)

        if isinstance(data, WeldObject):
            data = self.evaluate(verbose=False)
        elif isinstance(data, np.ndarray):
            data = data[:n]
        else:
            raise TypeError('underlying data is neither LazyData nor np.ndarray')

        return data

    # all comparisons are assumed to be on scalars
    def _comparison(self, other, comparison):
        if not isinstance(other, (str, unicode, int, long, float, bool)):
            raise TypeError('can only compare with scalars')

        assert isinstance(comparison, (str, unicode))

        return Series(weld_compare(self.expr,
                                   other,
                                   comparison,
                                   self.weld_type),
                      WeldBit(),
                      self.data_id)

    def __lt__(self, other):
        return self._comparison(other, '<')

    def __le__(self, other):
        return self._comparison(other, '<=')

    def __eq__(self, other):
        return self._comparison(other, '==')

    def __ne__(self, other):
        return self._comparison(other, '!=')

    def __ge__(self, other):
        return self._comparison(other, '>=')

    def __gt__(self, other):
        return self._comparison(other, '>')

    def sum(self):
        """ Sums all the elements

        Returns
        -------
        Series

        """
        return LazyData(weld_aggregate(self.expr,
                                       "+",
                                       self.weld_type),
                        self.weld_type,
                        0)
