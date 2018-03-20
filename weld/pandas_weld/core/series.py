from grizzly.encoders import numpy_to_weld_type
from weld.types import WeldLong, WeldDouble
from weld.weldobject import WeldObject
from lazy_data import LazyData
from pandas_weld.weld import weld_aggregate, weld_compare, weld_filter, weld_element_wise_op, weld_count, weld_mean, \
    weld_standard_deviation
from utils import subset, replace_slice_defaults
import numpy as np


class Series(LazyData):
    """ Weld-ed pandas Series

    Parameters
    ----------
    data : np.ndarray / WeldObject
        raw data or weld expression
    dtype : np.dtype
        of the elements
    index : Index, RangeIndex, or MultiIndex
        index linked to the data; it is assumed to be of the same length
    name : str, optional
        name of the series
    data_id : str, optional
        generated only by parsers to record the existence of new data from file; needs to be passed on
        to other LazyData children objects, e.g. when creating a pandas_weld.Series from netCDF4_weld.Variable

    See also
    --------
    pandas.Series

    """

    def __init__(self, data, dtype, index, name=None, data_id=None):
        if not isinstance(data, (np.ndarray, WeldObject)):
            raise TypeError('expected np.ndarray or WeldObject in Series.__init__')

        super(Series, self).__init__(data, numpy_to_weld_type(dtype), 1, data_id)

        self.dtype = dtype
        self.index = index
        self.name = name

    @property
    def data(self):
        return self.expr

    def __repr__(self):
        return "Name:%s\n\tIndex:%s\n\tData:%s" % (self.name, repr(self.index), repr(self.expr))

    def __getitem__(self, item):
        """ Lazy operation to select a subset of the series

        Has consequences! When slicing, any previous and/or following operations on
        the data within will be done only on this subset of the data

        Parameters
        ----------
        item : slice or Series
            if slice, a slice of the data for the number of desired rows; currently
            must contain a stop value and will not work as expected for
            start != 0 and stride != 1;
            if Series, returns a filtered Series only with the elements corresponding to
            True in the item Series

        Returns
        -------
        Series

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

            new_index = self.index[item]

            return Series(subset(self, item).expr,
                          self.dtype,
                          new_index,
                          self.name,
                          self.data_id)
        elif isinstance(item, Series):
            if item.weld_type != numpy_to_weld_type(np.dtype(np.bool)):
                raise ValueError('expected series of bool to filter DataFrame rows')

            if isinstance(self.expr, LazyData):
                weld_type = self.expr.weld_type
                data_id = self.expr.data_id
            elif isinstance(self.expr, np.ndarray):
                weld_type = numpy_to_weld_type(self.expr.dtype)
                data_id = None
            else:
                raise TypeError('expected data in column to be of type LazyData or np.ndarray')

            new_index = self.index[item]

            return Series(weld_filter(self.expr,
                                      item.expr,
                                      weld_type),
                          self.dtype,
                          new_index,
                          self.name,
                          data_id)
        else:
            raise TypeError('expected a slice or a Series of bool in Series.__getitem__')

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

    # comparisons are limited to scalars
    def _comparison(self, other, comparison):
        if not isinstance(other, (str, unicode, int, long, float, bool)):
            raise TypeError('can only compare with scalars')

        assert isinstance(comparison, (str, unicode))

        return Series(weld_compare(self.expr,
                                   other,
                                   comparison,
                                   self.weld_type),
                      np.dtype(np.bool),
                      self.index,
                      self.name)

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

    # TODO: add type conversion(?); pandas works when e.g. column_of_ints - 2.0 => float result
    def _element_wise_operation(self, other, operation):
        if not isinstance(other, (str, unicode, int, long, float, bool)):
            raise TypeError('can only compare with scalars')

        assert isinstance(operation, (str, unicode))

        return Series(weld_element_wise_op(self.expr,
                                           other,
                                           operation,
                                           self.weld_type),
                      self.dtype,
                      self.index,
                      self.name)

    def __add__(self, other):
        return self._element_wise_operation(other, '+')

    def __sub__(self, other):
        return self._element_wise_operation(other, '-')

    def __mul__(self, other):
        return self._element_wise_operation(other, '*')

    def __div__(self, other):
        return self._element_wise_operation(other, '/')

    def _aggregate(self, operation):
        assert isinstance(operation, (str, unicode))

        return LazyData(weld_aggregate(self.expr,
                                       operation,
                                       self.weld_type),
                        self.weld_type,
                        0)

    def sum(self):
        return self._aggregate('+')

    def prod(self):
        return self._aggregate('*')

    def min(self):
        return self._aggregate('min')

    def max(self):
        return self._aggregate('max')

    def count(self):
        return LazyData(weld_count(self.expr),
                        WeldLong(),
                        0)

    # TODO: not safe against overflows, i.e. the sum in sum/length
    def mean(self):
        return LazyData(weld_mean(self.expr,
                                  self.weld_type),
                        WeldDouble(),
                        0)

    # TODO: same as mean
    def std(self):
        return LazyData(weld_standard_deviation(self.expr,
                                                self.weld_type),
                        WeldDouble(),
                        0)
